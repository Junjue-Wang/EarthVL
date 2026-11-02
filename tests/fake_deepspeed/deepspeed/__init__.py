"""A minimal CPU stand-in for the DeepSpeed API used by ``engine/trainer.py`` (TESTS ONLY, never imported otherwise).

The test fixture ``fake_deepspeed`` (tests/conftest.py) puts this directory first on ``sys.path``. It mimics
DeepSpeed 0.14:

* ``init_distributed()`` (a gloo process group) and ``initialize()`` -> ``(engine, optimizer, None, lr_scheduler)``;
* the optimizer of the config (``Adam`` / ``AdamW`` -> ``torch.optim.AdamW``), a config ``WarmupCosineLR`` or a client
  scheduler callable (called with the optimizer and stepped after every optimizer step);
* ``engine.backward`` / ``engine.step`` with gradient accumulation, ``global_steps``, ``get_lr``;
* ``save_checkpoint`` / ``load_checkpoint`` with DeepSpeed's layout ``<dir>/<tag>/mp_rank_00_model_states.pt`` (weights
  under ``'module'``, next to DeepSpeed's bookkeeping such as the set ``sparse_tensor_module_names``) plus
  ``<dir>/latest``, and DeepSpeed's failure modes: ``save_checkpoint`` ignores a failed ``makedirs`` and then fails in
  ``torch.save``; ``load_checkpoint`` raises a bare ``AssertionError`` when the tag has no model-states file, loads the
  module with ``load_module_strict``, and in bf16 / ZeRO mode ends a module-only load with
  ``optimizer._restore_from_bit16_weights()``, an ``AttributeError`` on an engine without optimizer;
* with ZeRO stage 2 / 3, ``save_checkpoint`` also stores the frozen parameters (``frozen_param_fragments``), and
  ``load_checkpoint`` copies every parameter that is frozen in the loading model from them after loading the module:
  a ``KeyError`` for a parameter that was trainable when the checkpoint was written;
* ``EMULATE_4BIT_STATE = True`` adds bitsandbytes-like quantization-state entries (``<weight>.absmax``, ...) for every
  2-D weight to the saved module states, as ``Linear4bit`` does in the state dict of a 4-bit (QLoRA) model; a strict
  ``load_state_dict`` rejects them as unexpected keys.

It rejects ``"auto"`` values and non-integer bucket sizes, which real DeepSpeed also refuses, so that the engine's
config sanitising is exercised.
"""
import glob
import math
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

__version__ = '0.0.0+fake'
CALLS = []  # (name, info) records inspected by the tests
EMULATE_4BIT_STATE = False


def init_distributed(dist_backend=None, **kwargs):
    CALLS.append(('init_distributed', dict(backend=dist_backend)))
    if not dist.is_initialized():
        dist.init_process_group(backend=dist_backend or 'gloo', init_method='env://')


def _check_config(node, path=''):
    if isinstance(node, dict):
        for key, value in node.items():
            _check_config(value, f'{path}.{key}' if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _check_config(value, f'{path}[{i}]')
    elif isinstance(node, str) and node.strip().lower() == 'auto':
        raise ValueError(f'fake deepspeed: "auto" is not a valid value for {path}')
    if path.endswith('bucket_size') and not isinstance(node, int):
        raise ValueError(f'fake deepspeed: {path} must be an int, got {node!r}')


class WarmupCosineLR(object):
    """Simplified DeepSpeed WarmupCosineLR: linear warm-up, then cosine decay to ``cos_min_ratio``."""

    def __init__(self, optimizer, total_num_steps, warmup_min_ratio=0.0, warmup_num_steps=1000,
                 cos_min_ratio=0.0001, warmup_type='log', last_batch_iteration=-1):
        self.optimizer = optimizer
        self.total_num_steps = max(int(total_num_steps), 1)
        self.warmup_min_ratio = warmup_min_ratio
        self.warmup_num_steps = int(warmup_num_steps)
        self.cos_min_ratio = cos_min_ratio
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.last_batch_iteration = last_batch_iteration

    def _ratio(self, it):
        if it < self.warmup_num_steps:
            return self.warmup_min_ratio + (1 - self.warmup_min_ratio) * it / max(self.warmup_num_steps, 1)
        progress = min((it - self.warmup_num_steps) / max(self.total_num_steps - self.warmup_num_steps, 1), 1.0)
        return self.cos_min_ratio + (1 - self.cos_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return [lr * self._ratio(max(self.last_batch_iteration, 0)) for lr in self.base_lrs]

    def step(self, last_batch_iteration=None):
        self.last_batch_iteration = self.last_batch_iteration + 1 if last_batch_iteration is None \
            else last_batch_iteration
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = lr

    def state_dict(self):
        return dict(last_batch_iteration=self.last_batch_iteration)

    def load_state_dict(self, state):
        self.last_batch_iteration = state['last_batch_iteration']


class DeepSpeedEngine(torch.nn.Module):
    def __init__(self, model, model_parameters, config, lr_scheduler=None):
        super().__init__()
        self.module = model
        self.param_names = {param: name for name, param in model.named_parameters()}
        self._config = config
        self.global_steps = 0
        self.micro_steps = 0
        self.gas = int(config.get('gradient_accumulation_steps', 1))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.bf16 = bool((config.get('bf16') or {}).get('enabled', False))
        self.zero_stage = int((config.get('zero_optimization') or {}).get('stage', 0))
        if self.bf16:
            self.module.to(torch.bfloat16)
        self.optimizer = None
        self.lr_scheduler = None
        opt_cfg = config.get('optimizer')
        if opt_cfg is not None:
            params = list(model_parameters) if model_parameters is not None else list(model.parameters())
            if not all(p.requires_grad for p in params):
                raise ValueError('fake deepspeed: frozen parameters were passed to the optimizer')
            kwargs = dict(opt_cfg.get('params', {}))
            kwargs.pop('adam_w_mode', None)
            kwargs.pop('torch_adam', None)
            if 'betas' in kwargs:
                kwargs['betas'] = tuple(kwargs['betas'])
            if opt_cfg['type'].lower() not in ('adam', 'adamw'):
                raise ValueError(f"fake deepspeed: unsupported optimizer {opt_cfg['type']}")
            self.optimizer = torch.optim.AdamW(params, **kwargs)
            sched_cfg = config.get('scheduler')
            if sched_cfg is not None:
                if sched_cfg['type'] != 'WarmupCosineLR':
                    raise ValueError(f"fake deepspeed: unsupported scheduler {sched_cfg['type']}")
                self.lr_scheduler = WarmupCosineLR(self.optimizer, **sched_cfg['params'])
            elif callable(lr_scheduler):
                self.lr_scheduler = lr_scheduler(self.optimizer)
            else:
                self.lr_scheduler = lr_scheduler

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, loss):
        CALLS.append(('backward', float(loss.detach())))
        (loss / self.gas).backward()
        return loss

    def step(self):
        if (self.micro_steps + 1) % self.gas == 0:
            if self.world_size > 1:
                for p in self.module.parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad)
                        p.grad /= self.world_size
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.global_steps += 1
        self.micro_steps += 1

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups] if self.optimizer is not None else []

    def save_checkpoint(self, save_dir, tag=None, client_state=None, save_latest=True):
        tag = tag or f'global_step{self.global_steps}'
        CALLS.append(('save_checkpoint', dict(tag=tag)))
        if not dist.is_initialized() or dist.get_rank() == 0:
            path = os.path.join(save_dir, tag)
            try:  # DeepSpeed's _create_checkpoint_file only logs a failed makedirs
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass
            module = self.module.state_dict()
            if EMULATE_4BIT_STATE:  # what bitsandbytes' Linear4bit._save_to_state_dict adds for its weight
                for name, value in list(module.items()):
                    if name.endswith('.weight') and value.dim() == 2:
                        module[name + '.absmax'] = torch.ones(1)
                        module[name + '.quant_map'] = torch.zeros(16)
                        module[name + '.quant_state.bitsandbytes__nf4'] = torch.zeros(8, dtype=torch.uint8)
            frozen = None  # DeepSpeed stores the frozen parameters separately when ZeRO partitions the gradients
            if self.zero_stage >= 2:
                frozen = OrderedDict((name, p.detach().cpu()) for name, p in self.module.named_parameters()
                                     if not p.requires_grad)
            state = dict(module=module,
                         buffer_names=[name for name, _ in self.module.named_buffers()],
                         optimizer=self.optimizer.state_dict() if self.optimizer is not None else None,
                         param_shapes=[OrderedDict((name, p.shape) for name, p in self.module.named_parameters()
                                                   if p.requires_grad)],
                         frozen_param_shapes=None if frozen is None else OrderedDict(
                             (name, p.shape) for name, p in frozen.items()),
                         frozen_param_fragments=frozen,
                         lr_scheduler=self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                         sparse_tensor_module_names=set(), skipped_steps=0, global_steps=self.global_steps,
                         dp_world_size=self.world_size, mp_world_size=1, ds_config=self._config,
                         ds_version=__version__)
            state.update(client_state or {})
            torch.save(state, os.path.join(path, 'mp_rank_00_model_states.pt'))
            if save_latest:
                with open(os.path.join(save_dir, 'latest'), 'w') as f:
                    f.write(tag)
        if dist.is_initialized():
            dist.barrier()
        return True

    def load_checkpoint(self, load_dir, tag=None, load_module_strict=True, load_optimizer_states=True,
                        load_lr_scheduler_states=True, load_module_only=False, custom_load_fn=None):
        CALLS.append(('load_checkpoint', dict(tag=tag, module_only=load_module_only, strict=load_module_strict)))
        if tag is None:
            latest = os.path.join(load_dir, 'latest')
            if not os.path.isfile(latest):  # DeepSpeed only logs a warning
                return None, None
            with open(latest) as f:
                tag = f.read().strip()
        # DeepSpeed globs the model-states files of the tag; its SDLoaderBase asserts that there is one (no message)
        files = sorted(glob.glob(os.path.join(load_dir, str(tag), 'mp_rank_*_model_states.pt')))
        assert len(files) > 0
        state = torch.load(files[0], map_location='cpu', weights_only=False)
        self.module.load_state_dict(state['module'], strict=load_module_strict)
        if state.get('frozen_param_fragments') is not None:  # as DeepSpeed's load_module_state_dict
            for param in self.module.parameters():
                if not param.requires_grad:
                    param.data.copy_(state['frozen_param_fragments'][self.param_names[param]].data)
        if not load_module_only:
            if load_optimizer_states and self.optimizer is not None and state.get('optimizer') is not None:
                self.optimizer.load_state_dict(state['optimizer'])
            if load_lr_scheduler_states and self.lr_scheduler is not None and state.get('lr_scheduler') is not None:
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            self.global_steps = state['global_steps']
        if (self.bf16 or self.zero_stage > 0) and (load_module_only or not load_optimizer_states) \
                and self.optimizer is None:
            # DeepSpeed then calls self.optimizer._restore_from_bit16_weights()
            raise AttributeError("'NoneType' object has no attribute '_restore_from_bit16_weights'")
        client_state = {k: v for k, v in state.items() if k not in ('module', 'optimizer', 'lr_scheduler')}
        return files[0], client_state


def initialize(args=None, model=None, optimizer=None, model_parameters=None, training_data=None, lr_scheduler=None,
               config=None, **kwargs):
    if not dist.is_initialized():
        init_distributed()
    if not isinstance(config, dict):
        raise TypeError(f'fake deepspeed: config must be a dict, got {type(config).__name__}')
    _check_config(config)
    if 'train_micro_batch_size_per_gpu' not in config and 'train_batch_size' not in config:
        raise ValueError('fake deepspeed: a batch size is required')
    engine = DeepSpeedEngine(model, model_parameters, config, lr_scheduler)
    CALLS.append(('initialize', dict(config=config)))
    return engine, engine.optimizer, None, engine.lr_scheduler

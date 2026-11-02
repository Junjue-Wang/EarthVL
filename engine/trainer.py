"""DeepSpeed training and evaluation engine for EarthVL.

The experiments of the paper were run with an internal build of ``ever`` whose DeepSpeed trainer was never released.
This module re-implements what the experiments rely on with public packages only: ``ever-beta`` (registry, config
files, model / data-loader builders), ``deepspeed`` (optimizer, mixed precision, ZeRO, checkpoints) and ``torch``.

Entry scripts (``train_generation.py``, ``train_seg.py``) register their modules and call :func:`main`::

    # training, e.g. on 4 GPUs
    deepspeed --include localhost:0,1,2,3 train_generation.py --config_path EarthVLNet --model_dir ./log/EarthVLNet \
        train.num_iters 40000 data.train.params.batch_size 6
    # evaluation of a checkpoint (one process; launch with torchrun / deepspeed to use several GPUs)
    python train_generation.py --config_path EarthVLNet --model_dir ./log/EarthVLNet \
        --eval_only --ckpt ./log/EarthVLNet/model-40000.pth

A config (``configs/<name>.py``, dots for sub-folders, or a path to a ``.py`` file) defines ``config = dict(...)``:

* ``model=dict(type, params)`` and ``data=dict(train=dict(type, params), test=dict(type, params))``, built through
  the ever registry;
* ``train=dict(num_iters, log_interval_step, save_ckpt_interval_step, eval_interval_step, eval_after_train,
  resume_from_last=True, seed=None)``; intervals count optimizer steps and ``0`` disables them (the legacy keys
  ``save_ckpt_interval_epoch`` / ``eval_interval_epoch`` + ``eval_per_epoch`` count epochs, as in ever);
* ``deepspeed={...}``, a DeepSpeed config: ``"auto"`` placeholders are resolved and
  ``train_micro_batch_size_per_gpu`` follows ``data.train.params.batch_size``;
* optionally ``learning_rate=dict(type='poly' | 'cosine', params=...)``, used when ``deepspeed`` has no
  ``scheduler`` (:class:`ClientLRScheduler`).

Trailing ``key value`` pairs on the command line override config entries, e.g. ``train.num_iters 1000``.

Conventions kept from ever's launcher, which the original experiments used:

* in training mode the model returns a loss tensor, or a dict whose entries with a key ending in ``loss`` are summed
  and back-propagated; every entry is logged (``NumericalLoss`` also returns statistics that must not be trained);
* the data sampler is re-seeded with the global step (``StepDistributedSampler.set_step``) before every batch;
* checkpoints are DeepSpeed directories ``<model_dir>/model-<step>.pth`` listed in ``checkpoint_info.json``, and a
  run restarted in the same ``model_dir`` resumes from the last one;
* ``evaluate_fn(ctx, dataloader)`` runs on every rank (distributed evaluation).

``--eval_only --ckpt`` accepts a DeepSpeed tag directory (``log/x/model-40000.pth``), a run directory (its last
checkpoint, named by ``latest`` or else by ``checkpoint_info.json``), an ever checkpoint, a ``.pth`` / ``.pt`` state
dict or a ``.safetensors`` file. A partial state dict, such as the released LLM-branch weights with the trainable
parameters only, must contain every trainable parameter; the frozen parameters keep the model's own (pre-trained)
values. ``torch.load`` runs with ``weights_only=True``; a file that needs full unpickling, which can execute code
stored in it, is loaded only when the environment variable ``EARTHVL_TRUST_CHECKPOINTS=1`` is set.
"""
import argparse
import contextlib
import glob
import importlib.util
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from ever.core import builder, registry
from ever.core.config import AttrDict, import_config
from ever.core.dist import all_gather, synchronize
from ever.core.to import to_device

__all__ = ['main', 'get_parser', 'EngineContext', 'ClientLRScheduler', 'sanitize_deepspeed_config',
           'load_model_weights', 'read_state_dict', 'resolve_deepspeed_checkpoint', 'TRUST_CHECKPOINTS_ENV']

LOGGER_NAME = 'EarthVL'
CKPT_INFO_NAME = 'checkpoint_info.json'
DS_MODEL_STATES = 'mp_rank_00_model_states.pt'
CKPT_TAG_PATTERN = re.compile(r'model-(\d+)\.pth')  # the tags written by save_checkpoint
# Set to 1 to allow full unpickling (weights_only=False) of checkpoints that torch.load(weights_only=True) refuses.
# Only do this for files you trust. module/seg/checkpoint_io.py follows the same policy.
TRUST_CHECKPOINTS_ENV = 'EARTHVL_TRUST_CHECKPOINTS'

# Values used for DeepSpeed "auto" placeholders (HF-Trainer style configs); these are DeepSpeed's own defaults.
_AUTO_DEFAULTS = {
    'reduce_bucket_size': 500_000_000,
    'allgather_bucket_size': 500_000_000,
    'sub_group_size': 1_000_000_000,
    'stage3_prefetch_bucket_size': 50_000_000,
    'stage3_param_persistence_threshold': 100_000,
    'stage3_max_live_parameters': 1_000_000_000,
    'stage3_max_reuse_distance': 1_000_000_000,
    'gradient_accumulation_steps': 1,
    'gradient_clipping': 0.0,
}
# DeepSpeed validates these as integers; configs often write them as floats (1e9).
_INT_KEYS = set(_AUTO_DEFAULTS) - {'gradient_clipping'} | {
    'train_micro_batch_size_per_gpu', 'train_batch_size', 'steps_per_print', 'stage'}


class EngineContext(object):
    """What ``evaluate_fn(ctx, dataloader)`` receives.

    Attributes: ``model`` (the unwrapped ``nn.Module``, in eval mode), ``model_dir``, ``global_step``, ``logger``
    (``logging.Logger``; silent on non-main ranks), ``is_main``, ``device``, ``config``, ``engine`` (DeepSpeed engine
    or ``None``), ``rank`` and ``world_size``.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return 'EngineContext({})'.format(', '.join(f'{k}={v!r}' for k, v in self.__dict__.items()
                                                    if k in ('model_dir', 'global_step', 'is_main', 'device')))


# ----------------------------------------------------------------------------------------------------------------
# command line, config, logging, distributed set-up
# ----------------------------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(description='Train or evaluate an EarthVL model with DeepSpeed.')
    parser.add_argument('--config_path', required=True,
                        help="config name under configs/ (e.g. 'EarthVLNet', 'seg.sfpn_convnext_large') "
                             "or a path to a .py config file")
    parser.add_argument('--model_dir', required=True, help='output directory: checkpoints, logs, predictions')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='set by the deepspeed launcher; LOCAL_RANK is used otherwise')
    parser.add_argument('--eval_only', action='store_true', help='evaluate --ckpt on data.test and exit')
    parser.add_argument('--ckpt', default=None,
                        help='checkpoint for --eval_only: a DeepSpeed tag directory (e.g. log/x/model-40000.pth) or '
                             'run directory (its last checkpoint), a .pth/.pt state dict, an ever checkpoint '
                             '{"model": state_dict, ...}, or a .safetensors file (e.g. trainable parameters only)')
    parser.add_argument('opts', nargs=argparse.REMAINDER,
                        help="config overrides as 'key value' pairs, e.g. train.num_iters 1000")
    return parser


def parse_args(argv=None):
    args = get_parser().parse_args(argv)
    opts = list(args.opts or [])
    if opts and opts[0] == '--':
        opts = opts[1:]
    if len(opts) % 2 != 0:
        raise SystemExit(f'config overrides must be "key value" pairs, got {opts}')
    keys = opts[0::2]
    flags = [k for k in keys if k.startswith('-')]
    if flags:
        raise SystemExit(f'options {flags} must come before the "key value" overrides')
    args.opts = opts
    if args.eval_only and not args.ckpt:
        raise SystemExit('--eval_only needs --ckpt')
    return args


def config_file_path(config_path):
    """The file behind a config name, resolved like ``ever.core.config.import_config``."""
    if config_path.endswith('.py'):
        return config_path
    segs = ['configs'] + config_path.split('.')
    segs[-1] += '.py'
    return os.path.join(os.path.curdir, *segs)


def load_config(config_path, opts=None):
    cfg = import_config(config_path)
    if opts:
        cfg.update_from_list(list(opts))
    for key in ('model', 'data'):
        if key not in cfg:
            raise KeyError(f'config {config_path!r} has no {key!r} entry')
    return cfg


def to_plain(obj):
    """AttrDict / tuple trees -> plain dicts and lists (deep copy)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def build_logger(model_dir, is_main, filename):
    """Console + ``<model_dir>/<filename>`` logger on the main rank, a silent logger elsewhere."""
    logger = logging.Logger(LOGGER_NAME)  # not attached to the root logger
    logger.setLevel(logging.INFO)
    if is_main:
        fmt = logging.Formatter('%(asctime)s, %(levelname)s:%(name)s:%(message)s', '%Y-%m-%d %H:%M:%S')
        handlers = [logging.StreamHandler(sys.stdout)]
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            handlers.append(logging.FileHandler(os.path.join(model_dir, filename), mode='a', encoding='utf-8'))
        for h in handlers:
            h.setFormatter(fmt)
            logger.addHandler(h)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def deepspeed_available():
    return importlib.util.find_spec('deepspeed') is not None


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def init_distributed(use_deepspeed):
    """Set up torch.distributed and return whether a process group exists.

    Launched by ``deepspeed`` / ``torchrun`` (``WORLD_SIZE`` set): join the process group (NCCL on GPUs). Without a
    launcher, evaluation runs as a plain single process; DeepSpeed itself needs a process group, so a one-process
    group is created when ``use_deepspeed``.
    """
    if dist.is_available() and dist.is_initialized():
        return True
    if 'WORLD_SIZE' not in os.environ:
        if not use_deepspeed:
            return False
        os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(_free_port()))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    if use_deepspeed:
        import deepspeed
        deepspeed.init_distributed()
    else:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    return True


def get_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def current_device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------
# model / data
# ----------------------------------------------------------------------------------------------------------------
def _ensure_registered(type_name, registries, what):
    if any(type_name in r for r in registries):
        return
    # entry scripts normally register their modules; fall back to ever's walk of ./data, ./module and ./callback
    registry.register_all()
    if not any(type_name in r for r in registries):
        raise KeyError(f'{what} type {type_name!r} is not registered: import the module that defines it '
                       f'(the registry is filled by ever.registry.register_all() run from the repository root)')


def build_model(cfg):
    _ensure_registered(cfg.model['type'], [registry.MODEL], 'model')
    return builder.make_model(cfg.model)


def build_dataloader(data_cfg):
    _ensure_registered(data_cfg['type'], [registry.DATALOADER, registry.DATASET], 'data loader')
    return builder.make_dataloader(data_cfg)


def count_parameters(model):
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# ----------------------------------------------------------------------------------------------------------------
# DeepSpeed config and learning rate
# ----------------------------------------------------------------------------------------------------------------
def sanitize_deepspeed_config(ds_cfg, micro_batch=None, world_size=1, logger=None):
    """Return a plain-dict copy of ``ds_cfg`` that ``deepspeed.initialize`` accepts.

    * ``"auto"`` placeholders (filled by the HF Trainer, not by DeepSpeed) get DeepSpeed's defaults
      (``reduce_bucket_size`` 5e8, ...) or the loader batch size; unknown ones raise;
    * integer-valued floats of integer fields (``1e9``) become ``int``;
    * ``train_micro_batch_size_per_gpu`` is the loader batch size ``micro_batch`` (the batch actually fed),
      ``gradient_accumulation_steps`` is kept (default 1) and an inconsistent ``train_batch_size`` is dropped.
    """
    log = logger.info if logger is not None else (lambda *a: None)
    warn = logger.warning if logger is not None else (lambda *a: None)
    ds = to_plain(ds_cfg or {})

    def walk(node, path):
        if isinstance(node, dict):
            for key in list(node.keys()):
                value = node[key]
                where = f'{path}.{key}' if path else key
                if isinstance(value, str) and value.strip().lower() == 'auto':
                    if key == 'train_micro_batch_size_per_gpu' and micro_batch is not None:
                        node[key] = int(micro_batch)
                    elif key == 'train_batch_size':
                        del node[key]  # derived by DeepSpeed from micro batch x accumulation x world size
                    elif key in _AUTO_DEFAULTS:
                        node[key] = _AUTO_DEFAULTS[key]
                    else:
                        raise ValueError(f'deepspeed.{where} = "auto" has no default here; set a value in the config')
                    log(f'deepspeed.{where}: "auto" -> {node.get(key, "(derived)")}')
                elif isinstance(value, (dict, list)):
                    walk(value, where)
                elif key in _INT_KEYS and isinstance(value, float) and value.is_integer():
                    node[key] = int(value)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f'{path}[{i}]')

    walk(ds, '')

    if micro_batch is not None:
        micro_batch = int(micro_batch)
        cur = ds.get('train_micro_batch_size_per_gpu')
        if cur is not None and int(cur) != micro_batch:
            warn(f'deepspeed.train_micro_batch_size_per_gpu={cur} differs from data.train.params.batch_size='
                 f'{micro_batch}; using {micro_batch}, the batch size the data loader produces')
        ds['train_micro_batch_size_per_gpu'] = micro_batch
    ds['gradient_accumulation_steps'] = int(ds.get('gradient_accumulation_steps', 1) or 1)
    if 'train_batch_size' in ds and micro_batch is not None:
        expected = micro_batch * ds['gradient_accumulation_steps'] * world_size
        if int(ds['train_batch_size']) != expected:
            warn(f'deepspeed.train_batch_size={ds["train_batch_size"]} != micro batch x accumulation x world size '
                 f'= {expected}; dropping it (DeepSpeed derives it)')
            del ds['train_batch_size']
    return ds


class ClientLRScheduler(object):
    """ever-style ``poly`` / ``cosine`` learning-rate schedules with an optional linear warm-up.

    The learning rate of optimizer step ``t`` (0-based) is::

        warm-up (t <= warmup_step): warmup_init_lr + t * (base_lr - warmup_init_lr) / warmup_step
        poly:   base_lr * (1 - t / max_iters) ** power
        cosine: eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t / max_iters))

    with ``t`` clipped to ``max_iters``. ``cosine`` with a warm-up is ``cosine_warm`` of the original code base
    (the cosine phase keeps counting from step 0). The object is handed to ``deepspeed.initialize``, which steps it
    after every optimizer step and stores its state in the checkpoints.
    """

    def __init__(self, optimizer, type='poly', base_lr=None, max_iters=None, power=0.9, eta_min=0.0,
                 warmup_step=0, warmup_init_lr=0.0):
        if type not in ('poly', 'cosine'):
            raise ValueError(f"learning_rate.type must be 'poly' or 'cosine', got {type!r}")
        if not max_iters or max_iters <= 0:
            raise ValueError('learning_rate.params.max_iters must be > 0')
        self.optimizer = optimizer
        self.type = type
        self.base_lr = float(base_lr if base_lr is not None else optimizer.param_groups[0]['lr'])
        self.max_iters = int(max_iters)
        self.power = float(power)
        self.eta_min = float(eta_min)
        self.warmup_step = int(warmup_step or 0)
        self.warmup_init_lr = float(warmup_init_lr or 0.0)
        self.last_step = 0
        self._last_lr = []
        self._apply()

    def lr_at(self, step):
        if self.warmup_step > 0 and step <= self.warmup_step:
            return self.warmup_init_lr + step * (self.base_lr - self.warmup_init_lr) / self.warmup_step
        t = min(step, self.max_iters)
        if self.type == 'poly':
            return self.base_lr * (1.0 - t / self.max_iters) ** self.power
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * t / self.max_iters))

    def _apply(self):
        lr = self.lr_at(self.last_step)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self._last_lr = [lr for _ in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        self.last_step += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def get_lr(self):
        return self.get_last_lr()

    def state_dict(self):
        return dict(last_step=self.last_step, type=self.type, base_lr=self.base_lr, max_iters=self.max_iters)

    def load_state_dict(self, state):
        self.last_step = int(state['last_step'])
        self._apply()


def _lr_schedule_params(cfg, ds_cfg, logger):
    lr_cfg = cfg.get('learning_rate', None)
    if not lr_cfg:
        return None
    if 'scheduler' in ds_cfg:
        logger.warning('both deepspeed.scheduler and learning_rate are set; using deepspeed.scheduler')
        return None
    params = dict(to_plain(lr_cfg.get('params', {})))
    params['type'] = lr_cfg['type']
    params.setdefault('max_iters', int(cfg.train['num_iters']))
    return params


def _check_schedule_length(ds_cfg, lr_params, num_iters, logger):
    total = ds_cfg.get('scheduler', {}).get('params', {}).get('total_num_steps', None)
    if total is not None and int(total) != num_iters:
        logger.warning(f'deepspeed.scheduler.params.total_num_steps={total} != train.num_iters={num_iters}')
    if lr_params is not None and int(lr_params['max_iters']) != num_iters:
        logger.warning(f'learning_rate.params.max_iters={lr_params["max_iters"]} != train.num_iters={num_iters}')


# ----------------------------------------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _allow_sets():
    """Let ``torch.load(weights_only=True)`` build sets, which torch 2.4 refuses by default.

    DeepSpeed stores ``sparse_tensor_module_names`` as a set in every model-states file. The previous allowlist is
    restored afterwards.
    """
    serialization = torch.serialization
    if not hasattr(serialization, 'add_safe_globals'):  # torch < 2.4
        yield
        return
    saved = list(serialization.get_safe_globals())
    serialization.add_safe_globals([set])
    try:
        yield
    finally:
        serialization.clear_safe_globals()
        serialization.add_safe_globals(saved)


def _torch_load(path, logger=None):
    """``torch.load(weights_only=True)``; full unpickling only when EARTHVL_TRUST_CHECKPOINTS=1."""
    try:
        with _allow_sets():
            return torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as e:
        if os.environ.get(TRUST_CHECKPOINTS_ENV, '0') != '1':
            raise pickle.UnpicklingError(
                f'{path} cannot be loaded with torch.load(weights_only=True): it stores Python objects other than '
                f'tensors and plain containers. Loading it with full unpickling can execute arbitrary code stored in '
                f'the file. If you trust the file, set {TRUST_CHECKPOINTS_ENV}=1 and run again.') from e
        (logger or logging.getLogger(__name__)).warning(
            f'{TRUST_CHECKPOINTS_ENV}=1: loading {path} with full unpickling (weights_only=False); this can execute '
            f'code stored in the file.')
        return torch.load(path, map_location='cpu', weights_only=False)


def _strip_prefixes(state_dict, prefixes=('module.', '_orig_mod.')):
    out = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p):]
                    changed = True
        out[key] = value
    return out


def resolve_deepspeed_checkpoint(path):
    """``(load_dir, tag)`` of the DeepSpeed checkpoint at ``path``.

    ``path`` is a tag directory holding ``mp_rank_00_model_states.pt`` (e.g. ``log/x/model-40000.pth``) or a run
    directory, whose last checkpoint is named by its ``latest`` file, or else by ``checkpoint_info.json``.
    """
    path = os.path.normpath(path)
    if os.path.isfile(os.path.join(path, DS_MODEL_STATES)):
        return os.path.split(path)
    tags = []
    latest = os.path.join(path, 'latest')
    if os.path.isfile(latest):
        with open(latest) as f:
            tags.append(f.read().strip())
    tags.append((read_checkpoint_info(path) or {}).get('last', {}).get('name', ''))
    for tag in tags:
        if tag and os.path.isfile(os.path.join(path, tag, DS_MODEL_STATES)):
            return path, tag
    raise FileNotFoundError(f'no DeepSpeed checkpoint in {path}: it has no {DS_MODEL_STATES} (a tag directory such as '
                            f'log/x/model-40000.pth), and neither its latest file nor {CKPT_INFO_NAME} names a tag '
                            f'directory that has one')


def _deepspeed_states_file(path):
    return os.path.join(*resolve_deepspeed_checkpoint(path), DS_MODEL_STATES)


def read_state_dict(path, logger=None):
    """Read model weights from a DeepSpeed checkpoint directory (tag or run directory), an ever checkpoint, a plain
    state dict or a ``.safetensors`` file.

    Returns ``(state_dict, meta)``: keys without ``module.`` / ``_orig_mod.`` prefixes, and the remaining top-level
    scalar entries of the file (``global_step`` / ``global_steps`` ...), or the metadata (strings) of a
    ``.safetensors`` file.
    """
    path = os.fspath(path)
    if os.path.isdir(path):
        path = _deepspeed_states_file(path)
    if path.lower().endswith('.safetensors'):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, framework='pt') as f:
            meta = dict(f.metadata() or {})
        return _strip_prefixes(load_file(path, device='cpu')), meta
    obj = _torch_load(path, logger)
    meta = {}
    state = obj
    if isinstance(obj, dict):
        for key in ('module', 'model', 'state_dict'):  # DeepSpeed, ever, generic
            if isinstance(obj.get(key, None), dict):
                state = obj[key]
                meta = {k: v for k, v in obj.items() if k != key and not isinstance(v, (dict, list))}
                break
    if not isinstance(state, dict):
        raise TypeError(f'unsupported checkpoint content in {path}: {type(state).__name__}')
    return _strip_prefixes(state), meta


def _report_keys(logger, source, missing, unexpected, limit=20):
    if not missing and not unexpected:
        logger.info(f'loaded {source}: all keys matched')
        return
    logger.warning(f'loaded {source} non-strictly: {len(missing)} missing and {len(unexpected)} unexpected keys')
    if missing:
        logger.warning(f'missing keys (first {limit}): {missing[:limit]}')
    if unexpected:
        logger.warning(f'unexpected keys (first {limit}): {unexpected[:limit]}')


def load_model_weights(model, path, logger, strict=False, limit=5):
    """``model.load_state_dict`` from any supported checkpoint, checking the keys; returns the checkpoint's metadata.

    A full state dict (all parameters of the model present) is loaded non-strictly, and missing / unexpected keys are
    logged. A partial one, such as the released LLM-branch weights with the trainable parameters only, must hold every
    parameter with ``requires_grad=True`` and nothing the model does not have; the frozen parameters and buffers
    missing from it keep the model's own values (e.g. the pre-trained language model).
    """
    state, meta = read_state_dict(path, logger)
    result = model.load_state_dict(state, strict=strict)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    params = dict(model.named_parameters(remove_duplicate=False))
    if not any(k in params for k in missing):
        _report_keys(logger, path, missing, unexpected)
        return meta
    buffers = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    not_frozen = [k for k in missing if k not in buffers and not (k in params and not params[k].requires_grad)]
    if unexpected or not_frozen:
        problems = []
        if unexpected:
            problems.append(f'{len(unexpected)} keys that the model does not have (first {limit}: '
                            f'{unexpected[:limit]})')
        if not_frozen:
            problems.append(f'no weights for {len(not_frozen)} trainable parameters (first {limit}: '
                            f'{not_frozen[:limit]})')
        raise RuntimeError(f'{path} holds part of the model weights, but it has ' + ' and '.join(problems)
                           + '; is it a checkpoint of this model and config?')
    logger.info(f'loaded {len(state)} tensors from {path} (a partial state dict); {len(missing)} frozen parameters '
                f'and buffers are not in it and keep their values from the model initialisation')
    return meta


def step_from_checkpoint(path, meta=None):
    """The global step of a checkpoint, used to name ``prediction-<step>.json``.

    In this order: ``model-<step>`` in the name, an integer ``global_step(s)`` stored in the checkpoint, a trailing
    ``_<step>`` / ``-<step>`` in the name (released weights, e.g. ``earthvl_count_40000.pth`` or
    ``earthvlnet_llm_lora_40000.safetensors``), the ``global_step(s)`` metadata string of a ``.safetensors`` file, 0.
    """
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r'model-(\d+)', name)
    if m:
        return int(m.group(1))
    meta = meta or {}
    for key in ('global_step', 'global_steps'):
        if isinstance(meta.get(key), int):
            return meta[key]
    m = re.search(r'[-_](\d+)$', os.path.splitext(name)[0])
    if m:
        return int(m.group(1))
    for key in ('global_step', 'global_steps'):
        if isinstance(meta.get(key), str) and meta[key].isdigit():
            return int(meta[key])
    return 0


def read_checkpoint_info(model_dir):
    path = os.path.join(model_dir, CKPT_INFO_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_checkpoint_info(model_dir, info):
    path = os.path.join(model_dir, CKPT_INFO_NAME)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(info, f)
    os.replace(tmp, path)


def save_checkpoint(engine, model_dir, step, ckpt_info, logger, is_main):
    """DeepSpeed checkpoint ``<model_dir>/model-<step>.pth`` (all ranks) + ``checkpoint_info.json`` (rank 0)."""
    tag = f'model-{step}.pth'
    engine.save_checkpoint(model_dir, tag=tag, client_state=dict(global_step=step))
    if is_main:
        ckpt_info[str(step)] = tag
        # the checkpoint just written is the one to resume from, also when a run that was not resumed left a later one
        ckpt_info['last'] = dict(step=step, name=tag)
        write_checkpoint_info(model_dir, ckpt_info)
        logger.info(f'{tag} has been saved.')
    synchronize()


def check_checkpoint_names(model_dir, start_step, num_iters, save_interval):
    """Fail before training when a file that is not a directory has the name of a checkpoint this run will write.

    The run saves ``model-<step>.pth`` every ``save_interval`` steps after ``start_step`` and at ``num_iters``.
    DeepSpeed would only fail when it saves there, e.g. after the whole run for a state dict ``model-<num_iters>.pth``.
    """
    if not os.path.isdir(model_dir):
        return
    clashes = []
    for name in sorted(os.listdir(model_dir)):
        m = CKPT_TAG_PATTERN.fullmatch(name)
        if m is None or os.path.isdir(os.path.join(model_dir, name)):
            continue
        step = int(m.group(1))
        if start_step < step <= num_iters and (step == num_iters or (save_interval > 0 and step % save_interval == 0)):
            clashes.append(name)
    if clashes:
        raise FileExistsError(f'{model_dir} holds {clashes}, files named like the DeepSpeed checkpoint directories '
                              f'that this run will save; move them elsewhere (e.g. a state dict for --eval_only) or '
                              f'train in another --model_dir')


def _load_error_keys(error):
    """``(missing, unexpected)`` keys listed by a ``load_state_dict(strict=True)`` RuntimeError (empty otherwise)."""
    keys = {}
    for line in str(error).splitlines():
        m = re.match(r'\s*(Missing|Unexpected) key\(s\) in state_dict: (.*)', line)
        if m:
            keys[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return keys.get('Missing', []), keys.get('Unexpected', [])


def _frozen_restore_failure(error, model):
    """The parameter named by ``error`` if it is the KeyError of DeepSpeed's restore of frozen parameters, else None.

    After loading the module of a ZeRO-2 / ZeRO-3 checkpoint, DeepSpeed (0.14) copies every parameter that is frozen in
    ``model`` from the checkpoint's ``frozen_param_fragments``, which hold the parameters that were frozen when the
    checkpoint was written. A parameter frozen now but trainable then, such as ``image_newline`` in the authors'
    original EarthVLNet runs, raises a KeyError with its name.
    """
    if isinstance(error, KeyError) and len(error.args) == 1:
        if error.args[0] in {name for name, p in model.named_parameters() if not p.requires_grad}:
            return error.args[0]
    return None


def try_resume(engine, model_dir, logger, limit=5):
    """Resume from the last DeepSpeed checkpoint of ``checkpoint_info.json``; return the global step.

    DeepSpeed loads the module strictly. If that fails on missing / unexpected keys, e.g. on the quantization state
    that a 4-bit (bitsandbytes) model adds to its state dict, the checkpoint is loaded again with
    ``load_module_strict=False`` (the optimizer, LR scheduler and step are restored as usual); a checkpoint without
    weights for trainable parameters still fails, and so does one written while a parameter that is frozen now was
    trainable (its optimizer state does not match).
    """
    info = read_checkpoint_info(model_dir)
    tag = (info or {}).get('last', {}).get('name', '')
    if not tag:
        return 0
    path = os.path.join(model_dir, tag)
    if not glob.glob(os.path.join(glob.escape(path), '*model_states.pt')):
        logger.warning(f'{path} is not a DeepSpeed checkpoint directory; not resuming')
        return 0
    try:
        load_path, _ = engine.load_checkpoint(model_dir, tag=tag)
    except KeyError as e:
        name = _frozen_restore_failure(e, engine.module)
        if name is None:
            raise
        raise RuntimeError(f'cannot resume from {path}: {name} is frozen in this model but was trainable when the '
                           f'checkpoint was written; train in another --model_dir') from e
    except RuntimeError as e:
        missing, unexpected = _load_error_keys(e)
        if not missing and not unexpected:
            raise
        params = dict(engine.module.named_parameters(remove_duplicate=False))
        trainable = [k for k in missing if k in params and params[k].requires_grad]
        if trainable:
            raise RuntimeError(f'cannot resume from {path}: it has no weights for {len(trainable)} trainable '
                               f'parameters (first {limit}: {trainable[:limit]})') from e
        logger.warning(f'strict loading of {path} failed on {len(unexpected)} unexpected and {len(missing)} missing '
                       f'keys (first {limit}: {(unexpected + missing)[:limit]}), e.g. the quantization state that a '
                       f'4-bit (bitsandbytes) model keeps in its state dict; loading it again with '
                       f'load_module_strict=False')
        load_path, _ = engine.load_checkpoint(model_dir, tag=tag, load_module_strict=False)
    if load_path is None:
        raise RuntimeError(f'could not resume from {path}')
    logger.info(f'{tag} has been restored (global step {engine.global_steps}).')
    return int(engine.global_steps)


# ----------------------------------------------------------------------------------------------------------------
# training helpers
# ----------------------------------------------------------------------------------------------------------------
class StepIterator(object):
    """Endless iterator over a data loader; reseeds the sampler with the global step like ever's launcher."""

    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0
        self._it = None

    def _sampler(self):
        batch_sampler = getattr(self.loader, 'batch_sampler', None)
        if batch_sampler is not None and getattr(batch_sampler, 'sampler', None) is not None:
            return batch_sampler.sampler
        return getattr(self.loader, 'sampler', None)

    def set_step(self, step):
        sampler = self._sampler()
        if hasattr(sampler, 'set_step'):
            sampler.set_step(step)
        elif hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(step)

    def next(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            return next(self._it)
        except StopIteration:
            self.epoch += 1
            self._it = iter(self.loader)
            return next(self._it)


def split_losses(output):
    """Loss to back-propagate and the scalars to log, from a model's training output."""
    if isinstance(output, torch.Tensor):
        loss_terms = {'loss': output}
        extra = {}
    elif isinstance(output, dict):
        loss_terms = {k: v for k, v in output.items() if k.endswith('loss')}
        extra = {k: v for k, v in output.items() if not k.endswith('loss')}
        if not loss_terms:
            raise ValueError(f'the training output {list(output)} has no entry whose key ends with "loss"')
    else:
        raise TypeError(f'the model must return a loss tensor or a dict of tensors in training mode, '
                        f'got {type(output).__name__}')
    total = sum(loss_terms.values())
    if not isinstance(total, torch.Tensor) or total.numel() != 1:
        raise ValueError('the training loss must be a single-element tensor')
    total = total.reshape(())
    scalars = {'total_loss': total.detach()} if len(loss_terms) > 1 or isinstance(output, dict) else {}
    scalars.update({k: v.detach() for k, v in loss_terms.items()})
    for k, v in extra.items():
        scalars[k] = v.detach() if isinstance(v, torch.Tensor) else v
    return total, scalars


def _as_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean())
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    return None


class Meter(object):
    """Averages of the logged scalars and timings since the last log line, reduced over ranks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums, self.counts = {}, {}
        self.steps = 0
        self.time = 0.0
        self.data_time = 0.0

    def add(self, scalars):
        for k, v in scalars.items():
            v = _as_float(v)
            if v is None:
                continue
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def summary(self):
        local = {k: (self.sums[k], self.counts[k]) for k in self.sums}
        gathered = all_gather(local)  # every rank calls this at the same steps
        sums, counts = {}, {}
        for part in gathered:
            for k, (s, c) in part.items():
                sums[k] = sums.get(k, 0.0) + s
                counts[k] = counts.get(k, 0) + c
        return {k: sums[k] / max(counts[k], 1) for k in self.sums if k in sums}


def _format_eta(seconds):
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def current_lr(engine):
    try:
        lrs = engine.get_lr()
        if lrs:
            return float(lrs[0])
    except Exception:  # scheduler not started yet, no optimizer
        pass
    opt = getattr(engine, 'optimizer', None)
    if opt is not None and getattr(opt, 'param_groups', None):
        return float(opt.param_groups[0]['lr'])
    return float('nan')


def run_evaluation(evaluate_fn, ctx, test_loader):
    """Call ``evaluate_fn(ctx, loader)`` on every rank with the model in eval mode."""
    logger = ctx.logger
    if evaluate_fn is None:
        logger.warning('no evaluate_fn was given; skipping evaluation')
        return None
    if test_loader is None:
        logger.warning('the config has no data.test; skipping evaluation')
        return None
    model = ctx.model
    was_training = model.training
    model.eval()
    logger.info(f'evaluation at step {ctx.global_step} on {_safe_len(test_loader)} batches per rank')
    start = time.time()
    try:
        result = evaluate_fn(ctx, test_loader)
    finally:
        model.train(was_training)
    synchronize()
    logger.info(f'evaluation at step {ctx.global_step} finished in {time.time() - start:.1f} s')
    return result


def _interval(train_cfg, key, legacy_key, steps_per_epoch, logger, legacy_enabled=True):
    """``train.<key>`` in optimizer steps; otherwise ever's legacy ``train.<legacy_key>``, which counts epochs."""
    if key in train_cfg and train_cfg[key] is not None:
        return int(train_cfg[key])
    epochs = train_cfg.get(legacy_key, None)
    if legacy_enabled and epochs is not None and int(epochs) > 0:
        steps = int(epochs) * steps_per_epoch
        logger.warning(f'train.{legacy_key}={epochs} (epochs, as in the original ever trainer) -> every {steps} '
                       f'optimizer steps; prefer train.{key}')
        return steps
    return 0


# ----------------------------------------------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------------------------------------------
def _safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return 0


def train(args, cfg, evaluate_fn=None):
    if not deepspeed_available():
        raise ImportError('training needs DeepSpeed: pip install deepspeed (evaluation with --eval_only does not)')
    distributed = init_distributed(use_deepspeed=True)
    rank, world_size = get_rank_world()
    is_main = rank == 0
    device = current_device()
    model_dir = args.model_dir
    if is_main:
        os.makedirs(model_dir, exist_ok=True)
        try:
            shutil.copy(config_file_path(args.config_path), os.path.join(model_dir, 'config.py'))
        except shutil.SameFileError:
            pass
    synchronize()
    logger = build_logger(model_dir, is_main, 'train.log')

    tcfg = cfg.train
    num_iters = int(tcfg['num_iters'])
    log_interval = max(int(tcfg.get('log_interval_step', 50) or 50), 1)
    eval_after_train = bool(tcfg.get('eval_after_train', False))
    if tcfg.get('seed', None) is not None:
        seed_everything(int(tcfg['seed']))

    logger.info(f'config: {args.config_path} | overrides: {args.opts}')
    logger.info(f'world size {world_size} | device {device} | model_dir {model_dir}')

    model = build_model(cfg)
    total, trainable = count_parameters(model)
    logger.info(f'model {type(model).__name__}: {total:,} parameters, {trainable:,} trainable')

    micro_batch = cfg.data.train.params.get('batch_size', None) if 'params' in cfg.data.train else None
    ds_config = sanitize_deepspeed_config(cfg.get('deepspeed', {}), micro_batch, world_size, logger)
    lr_params = _lr_schedule_params(cfg, ds_config, logger)
    _check_schedule_length(ds_config, lr_params, num_iters, logger)
    import deepspeed
    init_kwargs = dict(model=model, model_parameters=[p for p in model.parameters() if p.requires_grad],
                       config=ds_config)
    if lr_params is not None:
        init_kwargs['lr_scheduler'] = lambda optimizer: ClientLRScheduler(optimizer, **lr_params)
        logger.info(f'learning rate schedule: {lr_params}')
    engine, _, _, _ = deepspeed.initialize(**init_kwargs)
    gas = int(ds_config['gradient_accumulation_steps'])
    logger.info(f'deepspeed config: {json.dumps(ds_config, default=str)}')

    ckpt_info = (read_checkpoint_info(model_dir) if is_main else None) or {'last': dict(step=0, name='')}
    step = try_resume(engine, model_dir, logger) if tcfg.get('resume_from_last', True) else 0

    train_loader = build_dataloader(cfg.data.train)
    steps_per_epoch = max(_safe_len(train_loader) // gas, 1)
    save_interval = _interval(tcfg, 'save_ckpt_interval_step', 'save_ckpt_interval_epoch', steps_per_epoch, logger)
    eval_interval = _interval(tcfg, 'eval_interval_step', 'eval_interval_epoch', steps_per_epoch, logger,
                              legacy_enabled=bool(tcfg.get('eval_per_epoch', False)))
    check_checkpoint_names(model_dir, step, num_iters, save_interval)
    need_eval = eval_after_train or eval_interval > 0
    test_loader = build_dataloader(cfg.data.test) if need_eval and 'test' in cfg.data else None
    global_batch = (micro_batch or 0) * gas * world_size
    num_samples = _safe_len(getattr(train_loader, 'dataset', None))
    logger.info(f'micro batch {micro_batch} x accumulation {gas} x world size {world_size} = global batch '
                f'{global_batch} | {_safe_len(train_loader)} batches per epoch per rank | num_iters {num_iters}'
                + (f' (~{num_iters * global_batch / num_samples:.1f} epochs)' if num_samples and global_batch else ''))

    def make_ctx(current_step):
        return EngineContext(model=engine.module, model_dir=model_dir, global_step=current_step, logger=logger,
                             is_main=is_main, device=device, config=cfg, engine=engine, rank=rank,
                             world_size=world_size)

    data_iter = StepIterator(train_loader)
    meter = Meter()
    saved_step = None
    eval_result = None
    if step >= num_iters:
        logger.info(f'global step {step} >= num_iters {num_iters}: nothing to train')
        saved_step = step
    else:
        logger.info(f'start training at step {step}')
    engine.train()
    while step < num_iters:
        start = time.time()
        data_iter.set_step(step)
        batch = data_iter.next()
        data_time = time.time() - start
        batch = to_device(batch, device)
        output = engine(*batch) if isinstance(batch, (list, tuple)) else engine(batch)
        loss, scalars = split_losses(output)
        engine.backward(loss)
        engine.step()
        meter.add(scalars)
        meter.time += time.time() - start
        meter.data_time += data_time
        if int(engine.global_steps) == step:  # gradient accumulation micro step
            continue
        step = int(engine.global_steps)
        meter.steps += 1

        if step % log_interval == 0 or step == num_iters:
            values = meter.summary()
            sec = meter.time / max(meter.steps, 1)
            if is_main:
                losses = ''.join(f'{k} = {v:.6f}, ' for k, v in values.items())
                logger.info(f'{losses}lr = {current_lr(engine):.6g}, step: {step}({data_iter.epoch}), '
                            f'({sec:.3f} sec / step, data: {meter.data_time / max(meter.steps, 1):.3f} sec, '
                            f'eta: {_format_eta((num_iters - step) * sec)})')
            meter.reset()
        if save_interval > 0 and step % save_interval == 0:
            save_checkpoint(engine, model_dir, step, ckpt_info, logger, is_main)
            saved_step = step
        if eval_interval > 0 and step % eval_interval == 0 and not (eval_after_train and step == num_iters):
            eval_result = run_evaluation(evaluate_fn, make_ctx(step), test_loader)
            engine.train()

    if saved_step != step:
        save_checkpoint(engine, model_dir, step, ckpt_info, logger, is_main)
    if eval_after_train:
        eval_result = run_evaluation(evaluate_fn, make_ctx(step), test_loader)
    logger.info('done')
    return dict(global_step=step, model_dir=model_dir, eval_result=eval_result, distributed=distributed)


# DeepSpeed config of the --eval_only engine. bf16 / fp16 stay off: with them, DeepSpeed's load_checkpoint (0.14) ends
# with optimizer._restore_from_bit16_weights(), which fails on an engine without optimizer. evaluate_checkpoint casts
# the model to the training precision itself (_training_dtype).
_EVAL_DEEPSPEED_CONFIG = dict(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1)


def _training_dtype(cfg):
    """``torch.float16`` / ``torch.bfloat16`` if DeepSpeed's fp16 / bf16 mode (it casts the model) is on, else None."""
    ds = to_plain(cfg.get('deepspeed', {}) or {})
    for key, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16), ('bfloat16', torch.bfloat16)):
        value = ds.get(key, None)
        if (value.get('enabled', False) if isinstance(value, dict) else value) is True:
            return dtype
    return None


def _describe(error):
    """``Type: first line of the message`` of an exception (DeepSpeed raises bare AssertionErrors)."""
    lines = str(error).strip().splitlines()
    return f'{type(error).__name__}: {lines[0]}' if lines else type(error).__name__


def evaluate_checkpoint(args, cfg, evaluate_fn=None):
    """``--eval_only``: load ``--ckpt`` and run ``evaluate_fn`` on ``data.test``."""
    ckpt = os.path.normpath(args.ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    run_dir = None
    if os.path.isdir(ckpt):  # a run directory is evaluated at its last tag, with and without DeepSpeed alike
        load_dir, tag = resolve_deepspeed_checkpoint(ckpt)
        if load_dir == ckpt:
            run_dir = ckpt
        ckpt = os.path.join(load_dir, tag)
    use_deepspeed = os.path.isdir(ckpt) and deepspeed_available()
    init_distributed(use_deepspeed=use_deepspeed)
    rank, world_size = get_rank_world()
    is_main = rank == 0
    device = current_device()
    logger = build_logger(args.model_dir, is_main, 'eval.log')
    logger.info(f'config: {args.config_path} | overrides: {args.opts}')
    logger.info(f'evaluating {ckpt}' + (f' (the last checkpoint of {run_dir})' if run_dir else '')
                + f' | world size {world_size} | device {device}')

    model = build_model(cfg)
    dtype = _training_dtype(cfg)
    if dtype is torch.float16:  # evaluate in the precision of training, as DeepSpeed casts the model
        model.half()
    elif dtype is torch.bfloat16:
        model.bfloat16()
    engine = None
    meta = {}
    if use_deepspeed:
        import deepspeed
        engine, _, _, _ = deepspeed.initialize(model=model, config=dict(_EVAL_DEEPSPEED_CONFIG))
        load_dir, tag = os.path.split(ckpt)
        failure = None
        try:
            load_path, meta = engine.load_checkpoint(load_dir, tag=tag, load_module_only=True,
                                                     load_optimizer_states=False, load_lr_scheduler_states=False)
            if load_path is None:
                raise FileNotFoundError(f'DeepSpeed found no checkpoint at {ckpt}')
            logger.info(f'loaded {ckpt} with DeepSpeed: all keys matched')
        except (RuntimeError, AssertionError) as e:  # strict key check failed, or no checkpoint DeepSpeed can read
            failure = _describe(e)
        except KeyError as e:  # DeepSpeed's restore of the frozen parameters failed (see _frozen_restore_failure)
            if _frozen_restore_failure(e, engine.module) is None:
                raise
            failure = f'{_describe(e)}, frozen here but trainable when the checkpoint was written'
        if failure is not None:  # outside the except clauses, whose traceback keeps what DeepSpeed read in memory
            logger.warning(f'strict DeepSpeed load failed ({failure}); loading non-strictly')
            meta = load_model_weights(engine.module, ckpt, logger, strict=False)
        model = engine.module
        meta = {k: v for k, v in (meta or {}).items() if not isinstance(v, (dict, list))}
    else:
        meta = load_model_weights(model, ckpt, logger, strict=False)
        model.to(device)
    step = step_from_checkpoint(ckpt, meta)

    test_loader = build_dataloader(cfg.data.test)
    ctx = EngineContext(model=model, model_dir=args.model_dir, global_step=step, logger=logger, is_main=is_main,
                        device=device, config=cfg, engine=engine, rank=rank, world_size=world_size)
    result = run_evaluation(evaluate_fn, ctx, test_loader)
    return dict(global_step=step, model_dir=args.model_dir, eval_result=result)


def main(evaluate_fn=None, argv=None):
    """Parse the command line (``argv`` defaults to ``sys.argv[1:]``) and train or evaluate.

    ``evaluate_fn(ctx, dataloader)`` is called on every rank for evaluation (see :class:`EngineContext`).
    """
    args = parse_args(argv)
    if args.local_rank is not None and 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if hasattr(sys.stdout, 'reconfigure'):  # metric tables contain non-ASCII characters
        try:
            sys.stdout.reconfigure(errors='replace')
        except (ValueError, OSError):
            pass
    cfg = load_config(args.config_path, args.opts)
    if args.eval_only:
        return evaluate_checkpoint(args, cfg, evaluate_fn)
    if 'train' not in cfg or 'num_iters' not in cfg.train:
        raise KeyError('training needs train.num_iters in the config')
    return train(args, cfg, evaluate_fn)


if __name__ == '__main__':
    main()

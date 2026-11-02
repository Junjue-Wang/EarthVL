"""engine/trainer.py: DeepSpeed config sanitising, LR schedules, checkpoint reading, and train / save / resume /
evaluate round trips.

The round trips run ``train_generation.py``'s evaluation with the real counting-branch config, model and loader
(shrunk to a tiny width) on synthetic data, with the test-only DeepSpeed stand-in of tests/fake_deepspeed, which also
emulates DeepSpeed's failure modes (strict module loading, missing tags, bf16 module-only loads).
"""
import argparse
import json
import logging
import os
import pickle
from collections import OrderedDict

import pytest
import torch

import _synthetic


# ---------------------------------------------------------------------------------------------------------------
# helpers of the engine
# ---------------------------------------------------------------------------------------------------------------
def test_sanitize_deepspeed_config():
    from engine.trainer import sanitize_deepspeed_config
    ds = sanitize_deepspeed_config({
        'train_micro_batch_size_per_gpu': 'auto', 'train_batch_size': 'auto', 'gradient_accumulation_steps': 'auto',
        'zero_optimization': {'stage': 2.0, 'reduce_bucket_size': 'auto', 'sub_group_size': 1e9},
    }, micro_batch=4, world_size=2)
    assert ds['train_micro_batch_size_per_gpu'] == 4 and 'train_batch_size' not in ds
    assert ds['gradient_accumulation_steps'] == 1
    assert ds['zero_optimization'] == {'stage': 2, 'reduce_bucket_size': 500_000_000, 'sub_group_size': 1_000_000_000}
    assert all(isinstance(v, int) for v in ds['zero_optimization'].values())
    # an inconsistent train_batch_size is dropped (DeepSpeed derives it)
    assert 'train_batch_size' not in sanitize_deepspeed_config({'train_batch_size': 64}, micro_batch=4, world_size=2)
    with pytest.raises(ValueError):
        sanitize_deepspeed_config({'optimizer': {'params': {'lr': 'auto'}}})


def test_client_lr_scheduler():
    from engine.trainer import ClientLRScheduler
    param = torch.nn.Parameter(torch.zeros(1))

    poly = ClientLRScheduler(torch.optim.SGD([param], lr=1.0), type='poly', base_lr=1e-4, max_iters=30000, power=0.9)
    assert poly.lr_at(0) == pytest.approx(1e-4)
    assert poly.lr_at(15000) == pytest.approx(1e-4 * 0.5 ** 0.9)
    assert poly.lr_at(30000) == 0.0

    # the counting-branch schedule (configs/EarthVLCount.py); the released checkpoint stored lr 1.5421257e-12
    opt = torch.optim.SGD([param], lr=1.0)
    cos = ClientLRScheduler(opt, type='cosine', base_lr=1e-3, max_iters=40000, eta_min=0., warmup_step=1000,
                            warmup_init_lr=1e-8)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-8)
    assert cos.lr_at(500) == pytest.approx(1e-8 + 500 * (1e-3 - 1e-8) / 1000)
    assert cos.lr_at(39999) == pytest.approx(1.5421257e-12, rel=1e-6)
    for _ in range(3):
        cos.step()
    assert cos.get_last_lr() == [cos.lr_at(3)] and cos.state_dict()['last_step'] == 3


def test_split_losses_trains_only_loss_entries():
    from engine.trainer import split_losses
    out = dict(numerical_loss=torch.tensor([2.0], requires_grad=True), scaler=torch.tensor([5.0]),
               ohem_num=torch.tensor([3.0]))
    total, scalars = split_losses(out)
    assert total.item() == 2.0 and total.dim() == 0
    assert set(scalars) == {'total_loss', 'numerical_loss', 'scaler', 'ohem_num'}
    total, scalars = split_losses(torch.tensor(1.5))
    assert total.item() == 1.5 and set(scalars) == {'loss'}
    with pytest.raises(ValueError):
        split_losses(dict(accuracy=torch.tensor(1.0)))


def test_step_from_checkpoint_names():
    from engine.trainer import step_from_checkpoint
    assert step_from_checkpoint('log/EarthVLCount/model-40000.pth') == 40000
    assert step_from_checkpoint('EarthVL-pretrained/earthvl_count_40000.pth') == 40000
    assert step_from_checkpoint('EarthVL-pretrained/earthvlnet_llm_lora_40000.safetensors') == 40000
    assert step_from_checkpoint('weights.pth', dict(global_step=123)) == 123
    # .safetensors metadata are strings; the file name comes first
    assert step_from_checkpoint('earthvlnet_llm_lora_40000.safetensors', dict(global_steps='39000')) == 40000
    assert step_from_checkpoint('weights.safetensors', dict(global_steps='40000')) == 40000
    assert step_from_checkpoint('weights.pth') == 0
    assert step_from_checkpoint('model_v2.pth') == 0


# ---------------------------------------------------------------------------------------------------------------
# reading checkpoints
# ---------------------------------------------------------------------------------------------------------------
class TinyNet(torch.nn.Module):
    """A frozen part (like the pre-trained language model), a buffer and a trainable head."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.backbone.requires_grad_(False)
        self.register_buffer('scale', torch.ones(4))
        self.head = torch.nn.Linear(4, 2)


def capture_logger(name='test_engine'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def test_torch_load_policy(tmp_path, monkeypatch, caplog):
    from engine.trainer import TRUST_CHECKPOINTS_ENV, read_state_dict
    monkeypatch.delenv(TRUST_CHECKPOINTS_ENV, raising=False)
    weight = torch.arange(4.)

    # DeepSpeed model states: a set and torch.Size entries load with weights_only=True, and torch's allowlist is
    # restored afterwards
    tag = tmp_path / 'run' / 'model-5.pth'
    tag.mkdir(parents=True)
    torch.save(dict(module={'module.w': weight}, sparse_tensor_module_names=set(),
                    param_shapes=[OrderedDict(w=weight.shape)], global_steps=5), tag / 'mp_rank_00_model_states.pt')
    state, meta = read_state_dict(tag)
    assert list(state) == ['w'] and torch.equal(state['w'], weight) and meta['global_steps'] == 5
    assert set not in torch.serialization.get_safe_globals()

    # a pickled Python object needs full unpickling: refused unless EARTHVL_TRUST_CHECKPOINTS=1
    path = tmp_path / 'ever_style.pth'
    torch.save(dict(model={'w': weight}, global_step=7, args=argparse.Namespace(lr=0.1)), path)
    with pytest.raises(pickle.UnpicklingError, match=TRUST_CHECKPOINTS_ENV):
        read_state_dict(path)
    monkeypatch.setenv(TRUST_CHECKPOINTS_ENV, '1')
    with caplog.at_level(logging.WARNING):
        state, meta = read_state_dict(path, capture_logger())
    assert torch.equal(state['w'], weight) and meta['global_step'] == 7
    assert any(r.levelno == logging.WARNING and 'full unpickling' in r.getMessage() for r in caplog.records)


def test_load_model_weights_full_and_partial(tmp_path, caplog):
    safetensors_torch = pytest.importorskip('safetensors.torch')
    from engine.trainer import load_model_weights
    logger = capture_logger()
    torch.manual_seed(0)
    trained = TinyNet()
    trainable = {k: v for k, v in trained.state_dict().items() if k.startswith('head.')}

    # trainable parameters only (.safetensors): the frozen parameters and the buffer keep the model's values
    path = tmp_path / 'tiny_trainable_3.safetensors'
    safetensors_torch.save_file(trainable, str(path), metadata={'global_steps': '3'})
    model = TinyNet()
    backbone = model.backbone.weight.clone()
    with caplog.at_level(logging.INFO):
        meta = load_model_weights(model, path, logger)
    assert meta == {'global_steps': '3'}
    assert torch.equal(model.head.weight, trained.head.weight) and torch.equal(model.backbone.weight, backbone)
    assert 'loaded 2 tensors' in caplog.text and '3 frozen parameters and buffers' in caplog.text

    # a partial state dict must not have keys the model lacks, nor lack trainable parameters
    safetensors_torch.save_file(dict(trainable, **{'head2.weight': torch.zeros(2, 4)}), str(path))
    with pytest.raises(RuntimeError, match='head2.weight'):
        load_model_weights(TinyNet(), path, logger)
    safetensors_torch.save_file({'head.weight': trainable['head.weight']}, str(path))
    with pytest.raises(RuntimeError, match='head.bias'):
        load_model_weights(TinyNet(), path, logger)

    # a full state dict keeps the non-strict load with a key report
    full = tmp_path / 'tiny_full.pth'
    torch.save(dict(trained.state_dict(), extra=torch.zeros(1)), full)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        load_model_weights(TinyNet(), full, logger)
    assert "unexpected keys (first 20): ['extra']" in caplog.text
    safetensors_torch.save_file(trained.state_dict(), str(path))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        load_model_weights(TinyNet(), path, logger)
    assert 'all keys matched' in caplog.text


def test_resolve_deepspeed_checkpoint(tmp_path):
    from engine.trainer import resolve_deepspeed_checkpoint
    run = tmp_path / 'run'
    for step in (2, 4):
        (run / f'model-{step}.pth').mkdir(parents=True)
        torch.save({}, run / f'model-{step}.pth' / 'mp_rank_00_model_states.pt')
    tag_dir = str(run / 'model-2.pth')
    assert resolve_deepspeed_checkpoint(tag_dir) == (str(run), 'model-2.pth')
    (run / 'checkpoint_info.json').write_text(json.dumps({'last': dict(step=4, name='model-4.pth')}))
    assert resolve_deepspeed_checkpoint(str(run)) == (str(run), 'model-4.pth')  # no latest: checkpoint_info.json
    (run / 'latest').write_text('model-2.pth')
    assert resolve_deepspeed_checkpoint(str(run)) == (str(run), 'model-2.pth')  # latest comes first
    (run / 'latest').write_text('model-6.pth')
    assert resolve_deepspeed_checkpoint(str(run)) == (str(run), 'model-4.pth')  # unless it names no checkpoint
    with pytest.raises(FileNotFoundError, match='no DeepSpeed checkpoint'):
        resolve_deepspeed_checkpoint(str(tmp_path))


def test_eval_deepspeed_engine_leaves_precision_to_the_engine(tmp_path, fake_deepspeed):
    import engine.trainer as trainer
    trainer.init_distributed(use_deepspeed=True)
    saved, _, _, _ = fake_deepspeed.initialize(model=torch.nn.Linear(3, 2),
                                               config=dict(train_micro_batch_size_per_gpu=1))
    saved.save_checkpoint(str(tmp_path), tag='model-1.pth')
    load = dict(tag='model-1.pth', load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)
    # like DeepSpeed 0.14, a module-only load in bf16 mode fails on an engine without optimizer ...
    bf16, _, _, _ = fake_deepspeed.initialize(model=torch.nn.Linear(3, 2),
                                              config=dict(train_micro_batch_size_per_gpu=1, bf16=dict(enabled=True)))
    with pytest.raises(AttributeError, match='_restore_from_bit16_weights'):
        bf16.load_checkpoint(str(tmp_path), **load)
    # ... so the --eval_only engine runs without DeepSpeed precision modes; evaluate_checkpoint casts the model
    model = torch.nn.Linear(3, 2)
    engine, _, _, _ = fake_deepspeed.initialize(model=model, config=dict(trainer._EVAL_DEEPSPEED_CONFIG))
    assert engine.load_checkpoint(str(tmp_path), **load)[0] is not None
    assert torch.equal(model.weight, saved.module.weight)
    assert trainer._training_dtype(dict(deepspeed=dict(bf16=dict(enabled=True)))) is torch.bfloat16
    assert trainer._training_dtype(dict(deepspeed=dict(bf16=dict(enabled=False)))) is None


# ---------------------------------------------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------------------------------------------
TINY_COUNT = [  # 1536 feature channels + a 16-channel mask embedding, width 32, one Transformer layer
    'model.params.text_hidden_size', '32', 'model.params.mask_emb_dim', '16', 'model.params.vision_channel', '1552',
    'model.params.count_cfg.dim', '32', 'model.params.count_cfg.depth', '1', 'model.params.count_cfg.heads', '2',
    'model.params.count_cfg.mlp_dim', '32', 'model.params.count_cfg.dim_head', '16',
]


def count_argv(model_dir, vqa_data, tokenizer_dir, num_iters, extra=()):
    return [
        '--config_path', 'EarthVLCount', '--model_dir', str(model_dir),
        *extra,
        'model.params.pretrained_model_dir', repr(tokenizer_dir.replace(os.sep, '/')),
        *TINY_COUNT,
        'data.train.params.qa_path', _synthetic.pylist(vqa_data.qa['Train']),
        'data.train.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Train']),
        'data.test.params.qa_path', _synthetic.pylist(vqa_data.qa['Test']),
        'data.test.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Test']),
        'data.train.params.batch_size', '2', 'data.train.params.num_workers', '0',
        'data.test.params.batch_size', '2', 'data.test.params.num_workers', '0',
        'train.num_iters', str(num_iters), 'train.save_ckpt_interval_step', '2', 'train.log_interval_step', '1',
        'learning_rate.params.max_iters', str(num_iters), 'learning_rate.params.warmup_step', '1',
    ]


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_count_branch_train_resume_and_evaluate(tmp_path, vqa_data, t5_tokenizer_dir, fake_deepspeed, monkeypatch):
    pytest.importorskip('albumentations')
    import engine.trainer as trainer
    import train_generation

    run = tmp_path / 'run'
    # 1) train 4 steps from scratch: checkpoints at 2 and 4, evaluation after training
    out = trainer.main(train_generation.evaluate_cls_fn, count_argv(run, vqa_data, t5_tokenizer_dir, 4))
    assert out['global_step'] == 4
    for step in (2, 4):
        assert (run / f'model-{step}.pth' / 'mp_rank_00_model_states.pt').is_file()
    info = read_json(run / 'checkpoint_info.json')
    assert info['last'] == dict(step=4, name='model-4.pth') and info['2'] == 'model-2.pth'
    assert (run / 'config.py').is_file()
    log = (run / 'train.log').read_text(encoding='utf-8')
    assert 'numerical_loss' in log and 'model-4.pth has been saved.' in log
    pred4 = read_json(run / 'prediction-4.json')
    per_image = sum(t in _synthetic.COUNT_SELECT for t, _, _ in _synthetic.QA_TEMPLATES)
    n_test = per_image * len(vqa_data.images['Test'])
    assert sum(len(v) for v in pred4.values()) == n_test - n_test % 2  # drop_last with batch 2
    assert set(pred4) <= set(vqa_data.images['Test'])
    first = next(iter(pred4.values()))[0]
    assert set(first) == {'Type', 'Question', 'Answer', 'Correct', 'GT_Answer'}
    # DeepSpeed got the sanitised config (micro batch of the loader) without a scheduler: the engine's LR schedule
    init = [info for name, info in fake_deepspeed.CALLS if name == 'initialize'][0]
    assert init['config']['train_micro_batch_size_per_gpu'] == 2 and 'scheduler' not in init['config']

    # 2) a restart in the same model_dir resumes from model-4.pth and trains to step 6
    fake_deepspeed.CALLS.clear()
    out = trainer.main(train_generation.evaluate_cls_fn, count_argv(run, vqa_data, t5_tokenizer_dir, 6))
    assert out['global_step'] == 6
    assert sum(1 for name, _ in fake_deepspeed.CALLS if name == 'backward') == 2  # steps 5 and 6 only
    assert read_json(run / 'checkpoint_info.json')['last'] == dict(step=6, name='model-6.pth')
    assert 'model-4.pth has been restored (global step 4)' in (run / 'train.log').read_text(encoding='utf-8')
    pred6 = read_json(run / 'prediction-6.json')

    # 3) --eval_only from the DeepSpeed checkpoint directory reproduces the in-training predictions
    eval_ds = tmp_path / 'eval_ds'
    out = trainer.main(train_generation.evaluate_cls_fn,
                       count_argv(eval_ds, vqa_data, t5_tokenizer_dir, 6,
                                  extra=['--eval_only', '--ckpt', str(run / 'model-6.pth')]))
    assert out['global_step'] == 6
    assert ('load_checkpoint', dict(tag='model-6.pth', module_only=True, strict=True)) in fake_deepspeed.CALLS
    assert read_json(eval_ds / 'prediction-6.json') == pred6

    # 4) --eval_only without DeepSpeed, from a plain state dict and from an ever-style checkpoint
    monkeypatch.setattr(trainer, 'deepspeed_available', lambda: False)
    states = torch.load(run / 'model-6.pth' / 'mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
    plain = tmp_path / 'earthvl_count_6.pth'
    torch.save(states['module'], plain)
    ever_style = tmp_path / 'model-6.pth'
    torch.save(dict(model={'module.' + k: v for k, v in states['module'].items()}, global_step=6), ever_style)
    for name, ckpt in (('eval_plain', plain), ('eval_ever', ever_style)):
        out = trainer.main(train_generation.evaluate_cls_fn,
                           count_argv(tmp_path / name, vqa_data, t5_tokenizer_dir, 6,
                                      extra=['--eval_only', '--ckpt', str(ckpt)]))
        assert out['global_step'] == 6, name
        assert read_json(tmp_path / name / 'prediction-6.json') == pred6, name
        assert 'all keys matched' in (tmp_path / name / 'eval.log').read_text(encoding='utf-8')


NO_EVAL = ['train.eval_after_train', 'False']


def calls(fake_deepspeed, name):
    return [info for call, info in fake_deepspeed.CALLS if call == name]


def test_resume_after_a_strict_load_failure(tmp_path, vqa_data, t5_tokenizer_dir, fake_deepspeed):
    pytest.importorskip('albumentations')
    import engine.trainer as trainer

    run = tmp_path / 'run'
    fake_deepspeed.EMULATE_4BIT_STATE = True  # saved module states carry bitsandbytes-like quantization state
    trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 2) + NO_EVAL)
    states = torch.load(run / 'model-2.pth' / 'mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
    assert any(k.endswith('.weight.absmax') for k in states['module'])

    # the strict load fails on the unexpected keys; the second, non-strict load restores the run at step 2
    fake_deepspeed.CALLS.clear()
    out = trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 4) + NO_EVAL)
    assert out['global_step'] == 4
    assert [info['strict'] for info in calls(fake_deepspeed, 'load_checkpoint')] == [True, False]
    assert len(calls(fake_deepspeed, 'backward')) == 2  # steps 3 and 4 only
    log = (run / 'train.log').read_text(encoding='utf-8')
    assert 'WARNING' in log and 'strict loading of' in log and 'load_module_strict=False' in log
    assert '.weight.absmax' in log and 'model-2.pth has been restored (global step 2)' in log

    # without the weights of a trainable parameter, the checkpoint is not resumed non-strictly
    states_file = run / 'model-4.pth' / 'mp_rank_00_model_states.pt'
    states = torch.load(states_file, map_location='cpu', weights_only=False)
    del states['module']['mm_projector.linear_1.weight']
    torch.save(states, states_file)
    with pytest.raises(RuntimeError, match='mm_projector.linear_1.weight'):
        trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 6) + NO_EVAL)


def test_eval_only_from_a_run_directory(tmp_path, vqa_data, t5_tokenizer_dir, fake_deepspeed, monkeypatch):
    pytest.importorskip('albumentations')
    import engine.trainer as trainer
    import train_generation

    run = tmp_path / 'run'
    trainer.main(train_generation.evaluate_cls_fn, count_argv(run, vqa_data, t5_tokenizer_dir, 4))
    pred4 = read_json(run / 'prediction-4.json')

    def evaluate(name, ckpt):
        out = trainer.main(train_generation.evaluate_cls_fn,
                           count_argv(tmp_path / name, vqa_data, t5_tokenizer_dir, 4,
                                      extra=['--eval_only', '--ckpt', str(ckpt)]))
        assert out['global_step'] == 4, name
        assert read_json(tmp_path / name / 'prediction-4.json') == pred4, name
        return (tmp_path / name / 'eval.log').read_text(encoding='utf-8')

    # DeepSpeed gets the tag named by the run directory's latest file (not the run directory's name)
    fake_deepspeed.CALLS.clear()
    log = evaluate('eval_latest', run)
    assert calls(fake_deepspeed, 'load_checkpoint') == [dict(tag='model-4.pth', module_only=True, strict=True)]
    assert f'the last checkpoint of {run}' in log
    assert not {'bf16', 'fp16'} & set(calls(fake_deepspeed, 'initialize')[0]['config'])
    # without a latest file (as in the authors' run directories) or when it names no checkpoint: checkpoint_info.json
    (run / 'latest').write_text('model-404.pth')
    evaluate('eval_stale_latest', run)
    (run / 'latest').unlink()
    fake_deepspeed.CALLS.clear()
    evaluate('eval_info', run)
    assert calls(fake_deepspeed, 'load_checkpoint')[0]['tag'] == 'model-4.pth'

    # DeepSpeed failing with its bare AssertionError: the engine's own reader loads the checkpoint
    def assertion_error(self, *args, **kwargs):
        raise AssertionError  # DeepSpeed's assert without a message

    with monkeypatch.context() as m:
        m.setattr(fake_deepspeed.DeepSpeedEngine, 'load_checkpoint', assertion_error)
        log = evaluate('eval_assert', run / 'model-4.pth')
    assert 'strict DeepSpeed load failed (AssertionError)' in log and 'all keys matched' in log

    # the same run directory without DeepSpeed
    monkeypatch.setattr(trainer, 'deepspeed_available', lambda: False)
    evaluate('eval_no_deepspeed', run)

    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='no DeepSpeed checkpoint'):
        trainer.main(train_generation.evaluate_cls_fn,
                     count_argv(tmp_path / 'eval_empty', vqa_data, t5_tokenizer_dir, 4,
                                extra=['--eval_only', '--ckpt', str(empty)]))


def test_training_fails_fast_on_a_file_named_like_a_checkpoint(tmp_path, vqa_data, t5_tokenizer_dir, fake_deepspeed):
    pytest.importorskip('albumentations')
    import engine.trainer as trainer

    # a state dict where the run will save its last checkpoint, listed as 'last' like the authors' ever checkpoints
    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'w': torch.zeros(1)}, run / 'model-4.pth')
    (run / 'checkpoint_info.json').write_text(json.dumps({'last': dict(step=4, name='model-4.pth')}))
    with pytest.raises(FileExistsError, match='model-4.pth'):
        trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 4) + NO_EVAL)
    assert not calls(fake_deepspeed, 'backward') and (run / 'model-4.pth').is_file()
    assert 'model-4.pth is not a DeepSpeed checkpoint directory; not resuming' in \
        (run / 'train.log').read_text(encoding='utf-8')

    # a file at a step this run does not save is no obstacle, and the checkpoints of a run that was not resumed
    # replace a stale 'last' entry, so that the next restart resumes from them
    (run / 'model-4.pth').rename(run / 'model-3.pth')
    (run / 'checkpoint_info.json').write_text(json.dumps({'last': dict(step=40000, name='model-40000.pth')}))
    assert trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 4) + NO_EVAL)['global_step'] == 4
    assert read_json(run / 'checkpoint_info.json')['last'] == dict(step=4, name='model-4.pth')
    fake_deepspeed.CALLS.clear()
    assert trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 6) + NO_EVAL)['global_step'] == 6
    assert len(calls(fake_deepspeed, 'backward')) == 2


FROZEN = ('counting_input_embedding.', 'mask_emb.')  # frozen here, like the pre-trained language model


def test_eval_only_from_trainable_parameters_safetensors(tmp_path, vqa_data, t5_tokenizer_dir, fake_deepspeed,
                                                         monkeypatch):
    pytest.importorskip('albumentations')
    safetensors_torch = pytest.importorskip('safetensors.torch')
    import engine.trainer as trainer
    import train_generation

    build_model = trainer.build_model

    def build_partly_frozen(cfg):
        torch.manual_seed(0)  # the frozen weights come from the initialisation, as pre-trained weights would
        model = build_model(cfg)
        for name, param in model.named_parameters():
            if name.startswith(FROZEN):
                param.requires_grad_(False)
        return model

    monkeypatch.setattr(trainer, 'build_model', build_partly_frozen)
    run = tmp_path / 'run'
    trainer.main(train_generation.evaluate_cls_fn, count_argv(run, vqa_data, t5_tokenizer_dir, 4))
    pred4 = read_json(run / 'prediction-4.json')  # evaluation of the full model after training

    # the trainable parameters only, like the released LLM-branch weights
    states = torch.load(run / 'model-4.pth' / 'mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
    trainable = {k: v for k, v in states['module'].items() if not k.startswith(FROZEN)}
    assert 0 < len(trainable) < len(states['module'])
    weights = tmp_path / 'earthvl_count_trainable_4.safetensors'
    safetensors_torch.save_file(trainable, str(weights), metadata={'global_steps': '4'})

    eval_dir = tmp_path / 'eval_safetensors'
    out = trainer.main(train_generation.evaluate_cls_fn,
                       count_argv(eval_dir, vqa_data, t5_tokenizer_dir, 4,
                                  extra=['--eval_only', '--ckpt', str(weights)]))
    assert out['global_step'] == 4  # the trailing _4 of the file name
    assert read_json(eval_dir / 'prediction-4.json') == pred4
    log = (eval_dir / 'eval.log').read_text(encoding='utf-8')
    n_frozen = len(states['module']) - len(trainable)
    assert f'loaded {len(trainable)} tensors' in log and f'{n_frozen} frozen parameters and buffers' in log


def test_checkpoint_written_while_a_frozen_parameter_was_trainable(tmp_path, vqa_data, t5_tokenizer_dir,
                                                                    fake_deepspeed, monkeypatch):
    """Like image_newline, trainable in the authors' EarthVLNet runs and frozen here: DeepSpeed's restore of the frozen
    parameters of a ZeRO-2 checkpoint fails with a KeyError. --eval_only then loads the checkpoint with the engine's
    reader, and resuming such a checkpoint fails with a clear error."""
    pytest.importorskip('albumentations')
    import engine.trainer as trainer
    import train_generation

    run = tmp_path / 'run'
    zero2 = ['deepspeed.zero_optimization.stage', '2']
    trainer.main(train_generation.evaluate_cls_fn, count_argv(run, vqa_data, t5_tokenizer_dir, 2) + zero2)
    pred2 = read_json(run / 'prediction-2.json')
    states = torch.load(run / 'model-2.pth' / 'mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
    assert states['frozen_param_fragments'] == {}  # nothing was frozen in training

    build_model = trainer.build_model

    def build_with_frozen_mask_emb(cfg):
        model = build_model(cfg)
        model.mask_emb.requires_grad_(False)
        return model

    monkeypatch.setattr(trainer, 'build_model', build_with_frozen_mask_emb)

    def evaluate(name):
        return trainer.main(train_generation.evaluate_cls_fn,
                            count_argv(tmp_path / name, vqa_data, t5_tokenizer_dir, 2,
                                       extra=['--eval_only', '--ckpt', str(run / 'model-2.pth')]))

    assert evaluate('eval')['global_step'] == 2
    assert read_json(tmp_path / 'eval' / 'prediction-2.json') == pred2
    log = (tmp_path / 'eval' / 'eval.log').read_text(encoding='utf-8')
    assert "strict DeepSpeed load failed (KeyError: 'mask_emb." in log and 'all keys matched' in log

    with pytest.raises(RuntimeError, match='mask_emb.* was trainable when the checkpoint was written'):
        trainer.main(None, count_argv(run, vqa_data, t5_tokenizer_dir, 4) + zero2 + NO_EVAL)

    # any other KeyError of DeepSpeed is raised
    def key_error(self, *args, **kwargs):
        raise KeyError('module')

    monkeypatch.setattr(fake_deepspeed.DeepSpeedEngine, 'load_checkpoint', key_error)
    with pytest.raises(KeyError, match='module'):
        evaluate('eval_key_error')

"""Stage-1 pipeline: checkpoint loading and the stage 1 -> stage 2 file contract.

tools/extract_seg_features.py writes what the VQA loader reads. Checkpoints are read with
torch.load(weights_only=True); full unpickling needs EARTHVL_TRUST_CHECKPOINTS=1.
"""
import argparse
import json
import logging
import os
import pickle
import shutil

import numpy as np
import pytest
import torch

import _synthetic

pytest.importorskip('albumentations')
h5py = pytest.importorskip('h5py')

# Semantic-FPN with a tiny ConvNeXt: the code paths of ConvNeXt-L at a fraction of the memory
TINY_ARCH = dict(depths=[1, 1, 1, 1], channels=[8, 16, 32, 64])
TINY_SEG_CONFIG = '''
import copy
from configs.seg.sfpn_convnext_large import config as _base

config = copy.deepcopy(_base)
config['model']['params']['encoder']['convnext'].update(arch=dict(depths=[1, 1, 1, 1], channels=[8, 16, 32, 64]),
                                                        init_cfg=None)
config['model']['params']['fpn']['in_channels_list'] = (8, 16, 32, 64)
'''


def _save_checkpoint(model, tmp_path, ckpt_format):
    """The authors' ever file, or a run directory of engine/trainer.py in DeepSpeed's layout."""
    state = model.state_dict()
    if ckpt_format == 'ever':
        path = tmp_path / 'model-30000.pth'
        torch.save(dict(model={'module.' + k: v for k, v in state.items()}, global_step=30000), path)
        return path
    # DeepSpeed also stores a set (sparse_tensor_module_names), which torch 2.4 refuses with weights_only=True
    run = tmp_path / 'run'
    (run / 'model-30000.pth').mkdir(parents=True)
    torch.save(dict(module=state, optimizer=None, sparse_tensor_module_names=set(), global_steps=30000,
                    ds_config=dict(train_micro_batch_size_per_gpu=8), ds_version='0.14.0', global_step=30000),
               run / 'model-30000.pth' / 'mp_rank_00_model_states.pt')
    (run / 'latest').write_text('model-30000.pth')
    return run


@pytest.mark.parametrize('ckpt_format', ['ever', 'deepspeed'])
def test_extract_features_then_vqa_loader(tmp_path, monkeypatch, ckpt_format):
    from ever.core.config import import_config
    from data.earthvl_mc import EarthVLDataset
    from module.seg.checkpoint_io import TRUST_CHECKPOINTS_ENV
    from module.seg.semantic_fpn import SemanticFPN
    from tools import extract_seg_features

    monkeypatch.delenv(TRUST_CHECKPOINTS_ENV, raising=False)  # both formats load with weights_only=True
    config_path = tmp_path / 'tiny_seg.py'
    config_path.write_text(TINY_SEG_CONFIG, encoding='utf-8')
    torch.manual_seed(0)
    model = SemanticFPN(import_config(str(config_path))['model']['params'])
    ckpt = _save_checkpoint(model, tmp_path, ckpt_format)

    data_root = tmp_path / 'EarthVLSet'
    rng = np.random.default_rng(0)
    for name in ('4191.png', '4192.png'):
        _synthetic.write_png(str(data_root / 'EarthVQA' / 'Test' / 'images_png' / name),
                             rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / 'features'
    argv = ['--config', str(config_path), '--ckpt', str(ckpt), '--data_root', str(data_root), '--out_dir',
            str(out_dir), '--subsets', 'EarthVQA', '--splits', 'Test', '--batch_size', '2', '--num_workers', '0',
            '--device', 'cpu']
    extract_seg_features.main(argv)

    feat_dir = out_dir / 'EarthVQA' / 'Test'
    with h5py.File(feat_dir / '4191.hdf5', 'r') as f:
        feature, pred_mask = f['feature'][()], f['pred_mask'][()]
    assert feature.shape == (64, 2, 2) and feature.dtype == np.float32  # [C, H/32, W/32] of the last stage
    assert pred_mask.shape == (64, 64) and pred_mask.dtype == np.uint8
    assert pred_mask.min() >= 1 and pred_mask.max() <= 8
    mtime = os.path.getmtime(feat_dir / '4191.hdf5')
    extract_seg_features.main(argv)  # finished files are skipped
    assert os.path.getmtime(feat_dir / '4191.hdf5') == mtime

    qa_path = tmp_path / 'Test_QA.json'
    qa_path.write_text(json.dumps({'4191.png': [dict(Type='Basic Judging', Question='Are there any roads in this '
                                                     'scene?', Answer='Yes')]}), encoding='utf-8')
    ds = EarthVLDataset(qa_path=str(qa_path), image_feat_dir=str(feat_dir), image_load=False, mask_load=False,
                        feat_load=True, pred_mask_load=True, encoding=False)
    image, target = ds[0]
    np.testing.assert_array_equal(image, feature)
    np.testing.assert_array_equal(target['seg_mask'], pred_mask.astype(np.int64) - 1)


def test_extract_default_splits():
    from tools import extract_seg_features

    # the VQA configs read only the Train and Test features
    assert extract_seg_features.parse_args(['--ckpt', 'model.pth']).splits == ['Train', 'Test']


def test_checkpoint_trust_policy(tmp_path, monkeypatch, caplog):
    from module.seg.checkpoint_io import TRUST_CHECKPOINTS_ENV, load_state_dict_any

    monkeypatch.delenv(TRUST_CHECKPOINTS_ENV, raising=False)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    keys = model.state_dict().keys()

    # a set (as in DeepSpeed files) is allowed only during the load
    get_safe_globals = getattr(torch.serialization, 'get_safe_globals', list)
    allowlist = list(get_safe_globals())
    torch.save(dict(module=model.state_dict(), sparse_tensor_module_names=set()), tmp_path / 'ds.pt')
    assert load_state_dict_any(str(tmp_path / 'ds.pt')).keys() == keys
    assert list(get_safe_globals()) == allowlist

    # an object that needs full unpickling is refused unless the variable is set
    path = tmp_path / 'with_args.pth'
    torch.save(dict(model=model.state_dict(), args=argparse.Namespace(lr=1e-4)), path)
    with pytest.raises(pickle.UnpicklingError, match=TRUST_CHECKPOINTS_ENV):
        load_state_dict_any(str(path))
    monkeypatch.setenv(TRUST_CHECKPOINTS_ENV, '1')
    with caplog.at_level(logging.WARNING, logger='module.seg.checkpoint_io'):
        assert load_state_dict_any(str(path)).keys() == keys
    assert any(r.levelno == logging.WARNING and 'full unpickling' in r.getMessage() for r in caplog.records)


def _fair_names(state_dict):
    """mmcls ConvNeXt names -> names of the official FAIR release (dwconv/pwconv, one final 'norm')."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(('norm0.', 'norm1.', 'norm2.')):
            continue
        k = k.replace('.depthwise_conv.', '.dwconv.').replace('.pointwise_conv1.', '.pwconv1.')
        out[k.replace('.pointwise_conv2.', '.pwconv2.').replace('norm3.', 'norm.')] = v
    return out


def test_convnext_fair_checkpoint_from_url(tmp_path, monkeypatch):
    """A file like the official FAIR ImageNet release ({'model': state_dict}) loads through torch.hub with
    weights_only=True."""
    from module.seg.checkpoint_io import TRUST_CHECKPOINTS_ENV
    from module.seg.convnext import ConvNeXt, load_pretrained_convnext

    monkeypatch.delenv(TRUST_CHECKPOINTS_ENV, raising=False)
    torch.manual_seed(0)
    src = ConvNeXt(arch=TINY_ARCH)
    fair = _fair_names(src.state_dict())
    fair.update({'head.weight': torch.randn(10, TINY_ARCH['channels'][-1]), 'head.bias': torch.zeros(10)})
    torch.save({'model': fair}, tmp_path / 'convnext_tiny_22k_224.pth')
    torch.save({'model': fair, 'args': argparse.Namespace(lr=1e-4)}, tmp_path / 'with_args.pth')
    # no network: the "download" copies the file into $TORCH_HOME/hub/checkpoints
    monkeypatch.setenv('TORCH_HOME', str(tmp_path / 'torch_home'))
    monkeypatch.setattr(torch.hub, 'download_url_to_file',
                        lambda url, dst, *args, **kwargs: shutil.copyfile(tmp_path / os.path.basename(url), dst))

    dst = ConvNeXt(arch=TINY_ARCH)
    result = load_pretrained_convnext(dst, 'http://127.0.0.1:9/convnext_tiny_22k_224.pth')
    assert sorted(result.missing_keys) == sorted(f'norm{i}.{p}' for i in range(3) for p in ('weight', 'bias'))
    assert result.unexpected_keys == []
    for k, v in src.state_dict().items():
        if not k.startswith(('norm0.', 'norm1.', 'norm2.')):
            assert torch.equal(dst.state_dict()[k], v), k
    with pytest.raises(pickle.UnpicklingError, match=TRUST_CHECKPOINTS_ENV):
        load_pretrained_convnext(dst, 'http://127.0.0.1:9/with_args.pth')

# Copyright (c) OpenMMLab. All rights reserved.
# ConvNeXt backbone of the authors' segmentation code (an mmcls 0.x port),
# rewritten without mmcv. Module and parameter names are unchanged, so the
# authors' trained Semantic-FPN checkpoints load by key. Apache-2.0 (OpenMMLab).
# The mmcls ConvNeXt is modified from the official implementation
# (https://github.com/facebookresearch/ConvNeXt, MIT) and timm (Apache-2.0);
# see THIRD_PARTY_NOTICES.md, items 9, 11 and 12.
"""ConvNeXt backbone (`A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_).

Differences to the mmcv-based original:

* ``init_cfg`` is interpreted here (TruncNormal / Constant / Pretrained), with
  the same order and semantics as ``mmcv.runner.BaseModule.init_weights``.
* ``Pretrained`` checkpoints are mapped explicitly to these parameter names.
  The official FAIR checkpoints, timm checkpoints (``hf-hub:timm/...``) and
  mmcls/mmpretrain checkpoints are supported. mmcv's non-strict
  ``load_checkpoint`` only logged a warning when the key names did not match;
  here loading fails if no tensor matches.
* ``torch.load`` uses ``weights_only=True``; full unpickling needs
  ``EARTHVL_TRUST_CHECKPOINTS=1`` (see ``module.seg.checkpoint_io``).
"""
import logging
import os
import re
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.seg.checkpoint_io import safe_load, torch_load

logger = logging.getLogger(__name__)

# Public ImageNet-22k ConvNeXt-L weights (official FAIR release). timm hosts the
# same weights as 'hf-hub:timm/convnext_large.fb_in22k'.
CONVNEXT_LARGE_IN22K = 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth'

# Set to 1 to skip the ``Pretrained`` initializer, e.g. when a trained
# Semantic-FPN checkpoint overwrites the backbone anyway.
SKIP_PRETRAINED_ENV = 'EARTHVL_SKIP_PRETRAINED'


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


def drop_path(x, drop_prob=0., training=False):
    """Stochastic depth per sample (same as ``mmcv.cnn.bricks.DropPath``)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def build_norm_layer(cfg, num_features):
    """Subset of ``mmcv.cnn.build_norm_layer``: LN2d (default), LN and BN."""
    cfg = dict(cfg)
    layer_type = cfg.pop('type', 'LN2d')
    if layer_type == 'LN2d':
        return LayerNorm2d(num_features, **cfg)
    if layer_type == 'LN':
        return nn.LayerNorm(num_features, **cfg)
    if layer_type == 'BN':
        cfg.pop('requires_grad', None)
        return nn.BatchNorm2d(num_features, **cfg)
    raise ValueError(f'Unsupported norm type {layer_type!r}')


def build_activation_layer(cfg):
    cfg = dict(cfg)
    layer_type = cfg.pop('type', 'GELU')
    if layer_type == 'GELU':
        return nn.GELU(**cfg)
    if layer_type == 'ReLU':
        return nn.ReLU(**cfg)
    raise ValueError(f'Unsupported activation type {layer_type!r}')


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt backbone returning the LayerNorm'd output of each stage in ``out_indices``.

    Args:
        arch (str | dict): 'tiny', 'small', 'base', 'large', 'xlarge' or a dict
            with ``depths`` and ``channels``.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
        norm_cfg (dict): Norm layers of the stem, the blocks and the outputs.
            The norms inside the downsample layers always use ``eps=1e-5``,
            as in the original code.
        act_cfg (dict): Activation between the pointwise convolutions.
        linear_pw_conv (bool): Whether to use linear layers as pointwise convs.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Init value for Layer Scale.
        out_indices (Sequence | int): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed).
        gap_before_final_norm (bool): Accepted for config compatibility. It is
            unused, as in the original segmentation code.
        pretrained (str, optional): Deprecated shortcut for
            ``init_cfg=dict(type='Pretrained', checkpoint=pretrained)``.
        init_cfg (dict | list[dict], optional): mmcv-style initializers,
            applied in order: ``TruncNormal``/``Normal``/``Constant`` (with
            ``layer`` class names) and ``Pretrained`` (``checkpoint`` = local
            file, http(s) URL or ``hf-hub:<repo_id>``; ``None`` skips it).
    """
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.init_cfg = init_cfg

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()
        self.init_weights()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # The output of LayerNorm2d may be discontiguous, which
                # may cause some problem in the downstream tasks
                outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()
        return self

    # ------------------------------------------------------------------ init
    def init_weights(self):
        if not self.init_cfg:
            return
        cfgs = self.init_cfg if isinstance(self.init_cfg, (list, tuple)) else [self.init_cfg]
        for cfg in cfgs:
            cfg = dict(cfg)
            init_type = cfg.pop('type')
            if init_type == 'Pretrained':
                checkpoint = cfg.get('checkpoint', None)
                if not checkpoint:
                    continue
                if os.environ.get(SKIP_PRETRAINED_ENV, '0') == '1':
                    logger.info('%s=1: skip loading pretrained weights from %s', SKIP_PRETRAINED_ENV, checkpoint)
                    continue
                load_pretrained_convnext(self, checkpoint, prefix=cfg.get('prefix', None))
            else:
                _apply_layer_init(self, init_type, cfg)


def _layer_matches(module, layers):
    names = [module.__class__.__name__] + [b.__name__ for b in module.__class__.__bases__]
    return bool(set(layers) & set(names))


def _apply_layer_init(model, init_type, cfg):
    """mmcv's TruncNormalInit / NormalInit / ConstantInit restricted to ``layer`` class names."""
    layers = cfg.get('layer', None)
    if isinstance(layers, str):
        layers = [layers]
    bias = cfg.get('bias', 0.)
    for m in model.modules():
        if layers is not None and not _layer_matches(m, layers):
            continue
        weight = getattr(m, 'weight', None)
        if not isinstance(weight, torch.Tensor):
            continue
        if init_type == 'TruncNormal':
            nn.init.trunc_normal_(weight, cfg.get('mean', 0.), cfg.get('std', 1.), cfg.get('a', -2.), cfg.get('b', 2.))
        elif init_type == 'Normal':
            nn.init.normal_(weight, cfg.get('mean', 0.), cfg.get('std', 1.))
        elif init_type == 'Constant':
            nn.init.constant_(weight, cfg.get('val', 1.))
        else:
            raise ValueError(f'Unsupported init type {init_type!r}')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, bias)


# ----------------------------------------------------------- pretrained weights
def _download_once(fn):
    """Run ``fn`` (a cached download) on local rank 0 first, then on the other ranks (cache hits)."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return fn()
    first = int(os.environ.get('LOCAL_RANK', 0)) == 0
    if not first:
        dist.barrier()
    result = fn()
    if first:
        dist.barrier()
    return result


def read_checkpoint(checkpoint, map_location='cpu'):
    """Read a checkpoint from a local path, an http(s) URL or ``hf-hub:<repo_id>[/<file>]``."""
    if checkpoint.startswith(('http://', 'https://', 'hf-hub:')):
        return _download_once(lambda: _read_checkpoint(checkpoint, map_location))
    return _read_checkpoint(checkpoint, map_location)


def _read_checkpoint(checkpoint, map_location='cpu'):
    if checkpoint.startswith(('http://', 'https://')):
        # downloaded once into the torch hub cache; a retry with weights_only=False reads the cached file
        return safe_load(lambda weights_only: torch.hub.load_state_dict_from_url(
            checkpoint, map_location=map_location, progress=True, weights_only=weights_only), checkpoint)
    if checkpoint.startswith('hf-hub:'):
        from huggingface_hub import hf_hub_download
        parts = checkpoint[len('hf-hub:'):].split('/')
        repo_id = '/'.join(parts[:2])
        filename = '/'.join(parts[2:]) or None
        if filename is None:
            try:
                path = hf_hub_download(repo_id, 'model.safetensors')
            except Exception:  # repositories without safetensors
                path = hf_hub_download(repo_id, 'pytorch_model.bin')
        else:
            path = hf_hub_download(repo_id, filename)
        return read_checkpoint(path, map_location)
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(checkpoint, device=str(map_location))
    return torch_load(checkpoint, map_location)


def _unwrap(state):
    for key in ('state_dict', 'model', 'module'):
        if isinstance(state, dict) and isinstance(state.get(key, None), dict):
            state = state[key]
    return state


def convnext_state_dict_to_mmcls(state_dict, prefix=None):
    """Map FAIR / timm / mmcls ConvNeXt checkpoints to the parameter names used here."""
    sd = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        for p in ('module.', '_orig_mod.'):
            if k.startswith(p):
                k = k[len(p):]
        sd[k] = v
    if prefix is None:
        # mmcls/mmseg checkpoints keep the backbone under 'backbone.', Semantic-FPN under 'en.'.
        for p in ('backbone.', 'en.'):
            if any(k.startswith(p) for k in sd):
                prefix = p
                break
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    keys = list(sd.keys())
    out = {}
    if any(k.startswith('stem.') for k in keys) or any('.blocks.' in k for k in keys):  # timm
        for k, v in sd.items():
            if k.startswith('head.') and not k.startswith('head.norm.'):
                continue
            nk = re.sub(r'^stem\.(\d+)\.', r'downsample_layers.0.\1.', k)
            nk = re.sub(r'^stages\.(\d+)\.downsample\.(\d+)\.', r'downsample_layers.\1.\2.', nk)
            nk = re.sub(r'^stages\.(\d+)\.blocks\.(\d+)\.', r'stages.\1.\2.', nk)
            nk = nk.replace('.conv_dw.', '.depthwise_conv.')
            nk = nk.replace('.mlp.fc1.', '.pointwise_conv1.').replace('.mlp.fc2.', '.pointwise_conv2.')
            nk = re.sub(r'^head\.norm\.', 'norm3.', nk)
            out[nk] = v
    elif any('.dwconv.' in k for k in keys):  # official FAIR release
        for k, v in sd.items():
            if k.startswith('head.'):
                continue
            nk = k.replace('.dwconv.', '.depthwise_conv.')
            nk = nk.replace('.pwconv1.', '.pointwise_conv1.').replace('.pwconv2.', '.pointwise_conv2.')
            nk = re.sub(r'^norm\.', 'norm3.', nk)
            out[nk] = v
    else:  # mmcls / mmpretrain names (identical to this module)
        out = {k: v for k, v in sd.items() if not k.startswith('head.')}
    return out


def load_pretrained_convnext(model, checkpoint, prefix=None):
    """Load ImageNet weights into ``model`` (non-strict), log what did not load and return the result."""
    state = convnext_state_dict_to_mmcls(_unwrap(read_checkpoint(checkpoint)), prefix=prefix)
    own = model.state_dict()
    matched, mismatched = {}, []
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            mismatched.append(f'{k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}')
        else:
            matched[k] = v
    result = model.load_state_dict(matched, strict=False)
    loaded = len([k for k in matched if k in own])
    msg = (f'ConvNeXt pretrained weights from {checkpoint}: loaded {loaded}/{len(own)} tensors; '
           f'missing {len(result.missing_keys)} {result.missing_keys}; '
           f'unexpected {len(result.unexpected_keys)} {result.unexpected_keys[:20]}; '
           f'shape mismatch {len(mismatched)} {mismatched[:20]}')
    logger.info(msg)
    if loaded == 0:
        raise RuntimeError('No ConvNeXt weights could be loaded from {}. {}'.format(checkpoint, msg))
    return result

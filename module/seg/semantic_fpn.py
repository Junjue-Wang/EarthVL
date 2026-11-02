"""Semantic-FPN with a ConvNeXt encoder: stage 1 of EarthVLNet.

Ported from the authors' segmentation code. Module and parameter names are
unchanged (``en``, ``fpn``, ``decoder``, ``cls_pred_conv``), so their trained
checkpoints load by key once a ``module.`` prefix is stripped.

* Training mode: ``model(image, dict(mask=...))`` returns ``{'seg_loss': ...}``
  (cross-entropy, label -1 ignored).
* Eval mode: ``model(image)`` returns ``(probs, feature)``. ``probs`` are the
  8-class softmax probabilities at input resolution and ``feature`` is the
  LayerNorm'd last ConvNeXt stage (``[N, 1536, H/32, W/32]`` for ConvNeXt-L).
  EarthVLNet consumes ``feature`` and ``argmax(probs) + 1``.
"""
import contextlib
import logging

import torch
import torch.nn as nn
from ever import registry
from ever.interface import ERModule

from module.seg.base import AssymetricDecoder, FPN, default_conv_block
from module.seg.convnext import ConvNeXt

logger = logging.getLogger(__name__)


@registry.MODEL.register('SemanticFPN')
class SemanticFPN(ERModule):
    def __init__(self, config):
        # Keep the literal ``super().__init__``: ever-beta >= 0.6 rewraps any
        # ERModule.__init__ without it and drops set_default_config().
        super().__init__(config)
        if 'convnext' in self.config.encoder:
            self.en = ConvNeXt(**self.config.encoder.convnext)
        else:
            raise ValueError('Only the ConvNeXt encoder is supported in this release, got encoder={}'.format(
                list(self.config.encoder.keys())))
        self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.loss.ignore_index)

        if self.config.sync_bn and torch.cuda.is_available():
            # The original ever trainer converted BN to SyncBN (train.sync_bn=True).
            # SyncBatchNorm falls back to plain BN without a process group or in eval mode;
            # state_dict keys are unchanged.
            self.fpn = nn.SyncBatchNorm.convert_sync_batchnorm(self.fpn)
            self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
        self._amp_warned = False

    def _autocast(self, x):
        """bf16 autocast for the training forward on CUDA (the authors used fp16 AMP with a GradScaler)."""
        amp = self.config.amp
        if not (self.training and amp.enabled and x.is_cuda):
            return contextlib.nullcontext()
        dtype = getattr(torch, amp.dtype)
        if dtype == torch.float16:
            raise ValueError('amp.dtype float16 would need loss scaling (a GradScaler), which this training does '
                             'not do; use bfloat16, or amp.enabled=False for fp32.')
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            if not self._amp_warned:
                logger.warning('This GPU does not support bfloat16; training SemanticFPN in fp32.')
                self._amp_warned = True
            return contextlib.nullcontext()
        return torch.autocast(device_type='cuda', dtype=dtype)

    def forward(self, x, y=None):
        with self._autocast(x):
            feat_list = self.en(x)
            if len(feat_list) == 5:
                feat_list = feat_list[1:]
            fpn_feat_list = self.fpn(feat_list)
            final_feat = self.decoder(fpn_feat_list)
            cls_pred = self.cls_pred_conv(final_feat)
            cls_pred = self.upsample4x_op(cls_pred)
        if self.training:
            if y is None or 'mask' not in y:
                raise ValueError('SemanticFPN needs y=dict(mask=...) in training mode.')
            cls_true = y['mask'].to(device=cls_pred.device).long()
            return dict(seg_loss=self.ce_loss(cls_pred.float(), cls_true))
        return cls_pred.softmax(dim=1), feat_list[-1]

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                convnext=dict(
                    arch='large',
                    out_indices=(0, 1, 2, 3),
                    drop_path_rate=0.5,
                    init_cfg=None,
                ),
            ),
            fpn=dict(
                in_channels_list=(192, 384, 768, 1536),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            classes=8,
            loss=dict(
                ignore_index=-1,
            ),
            # Convert the BN layers of the FPN/decoder to SyncBatchNorm when CUDA is available.
            sync_bn=False,
            # Mixed precision of the training forward (parameters and optimizer stay fp32).
            # Only bfloat16: float16 would need a GradScaler.
            amp=dict(enabled=False, dtype='bfloat16'),
        ))

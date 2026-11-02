"""Stage 1 of EarthVLNet: Semantic-FPN + ConvNeXt-L land-cover segmentation on EarthVLSet.

Paper setting (Experiments, land-cover semantic segmentation): AdamW (0.9, 0.999),
weight decay 0.05, lr 1e-4 with a poly schedule (power 0.9), batch 16 (2 GPUs x 8),
30k iterations, random scale {0.5..2.0} + 512x512 crops + flips/rotations/colour jitter.

Train:  scripts/train_seg.sh  (train_seg.py --config_path seg.sfpn_convnext_large ...)
Features for the VQA stage: scripts/extract_features.sh

Notes on fidelity to the authors' segmentation code:
* Images are normalised with mean = std = 0.5 on [0, 255] input, i.e. x / 127.5 - 1,
  as in the authors' EarthVLSet segmentation config (their older LoveDA-only
  config used ImageNet mean/std). The config of the authors' stage-1 checkpoint,
  whose features the released VQA weights were trained on, was not available;
  check MEAN/STD against it once it is. Features must be extracted with the
  normalisation the checkpoint was trained with.
* Mixed precision: the authors trained stage 1 with fp16 AMP (torch autocast with
  a GradScaler). This config uses bf16 autocast instead, a deviation that can
  shift the segmentation accuracy and the extracted features slightly.
  ``model.params.amp.enabled False`` trains in fp32.
* The original ``Transpose(0.5)`` is read by albumentations <= 1.4.3 as
  ``always_apply=0.5``, so the transpose was always applied; ``p=1.0`` keeps that.
* Rotterdam (Netherlands): the paper lists it under Val, but the Hugging Face
  EarthVL-GLOBAL Train split contains it (image ids 9118-9485). The authors'
  segmentation run also trained on Rotterdam, so stage 1 uses the HF Train
  folders as they are. The VQA configs leave it out of VQA training
  (``ROTTERDAM_IMAGES`` in configs/EarthVL_cls.py and configs/EarthVL_count.py).
"""
from albumentations import (Compose, HorizontalFlip, Normalize, RandomBrightnessContrast, RandomCrop,
                            RandomRotate90, Transpose, VerticalFlip)
from ever.preprocess.albu import RandomDiscreteScale, ToTensor

DATA_ROOT = './EarthVLSet'
SUBSETS = ['EarthVQA', 'EarthVL-GLOBAL']
# ImageNet-22k ConvNeXt-L (official FAIR release). Same weights on the HF hub:
# 'hf-hub:timm/convnext_large.fb_in22k'. Set to None to train from scratch.
CONVNEXT_PRETRAINED = 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth'

MAX_ITERS = 30000
MICRO_BATCH = 8  # per GPU; 2 GPUs -> global batch 16 (paper)
BASE_LR = 1e-4
# Not confirmed for the authors' stage-1 checkpoint (see the notes above).
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

config = dict(
    model=dict(
        type='SemanticFPN',
        params=dict(
            encoder=dict(
                convnext=dict(
                    arch='large',
                    out_indices=(0, 1, 2, 3),
                    drop_path_rate=0.5,
                    init_cfg=[
                        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                        dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                        dict(type='Pretrained', checkpoint=CONVNEXT_PRETRAINED),
                    ]),
            ),
            fpn=dict(
                in_channels_list=(192, 384, 768, 1536),
                out_channels=256,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                num_groups_gn=None,
            ),
            classes=8,
            loss=dict(ignore_index=-1),
            sync_bn=True,
            # bf16 autocast in the training forward on GPUs that support it; weights and
            # AdamW states stay fp32. Deviation: the authors used fp16 AMP with a GradScaler.
            amp=dict(enabled=True, dtype='bfloat16'),
        )),
    data=dict(
        train=dict(
            type='EarthVLSegLoader',
            params=dict(
                data_root=DATA_ROOT,
                subsets=SUBSETS,
                splits=['Train'],
                transforms=Compose([
                    RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], p=0.5),
                    RandomCrop(512, 512),
                    RandomBrightnessContrast(p=0.5),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                    Transpose(p=1.0),
                    RandomRotate90(p=0.5),
                    RandomRotate90(p=0.5),
                    Normalize(mean=MEAN, std=STD, max_pixel_value=255, always_apply=True),
                    ToTensor(),
                ]),
                training=True,
                batch_size=MICRO_BATCH,
                num_workers=4,
            ),
        ),
        # Test masks are withheld (Global-LoveDA benchmark on Codabench), so mIoU is reported on Val.
        test=dict(
            type='EarthVLSegLoader',
            params=dict(
                data_root=DATA_ROOT,
                subsets=SUBSETS,
                splits=['Val'],
                transforms=Compose([
                    Normalize(mean=MEAN, std=STD, max_pixel_value=255, always_apply=True),
                    ToTensor(),
                ]),
                training=False,
                batch_size=4,
                num_workers=4,
            ),
        ),
    ),
    train=dict(
        num_iters=MAX_ITERS,
        log_interval_step=50,
        save_ckpt_interval_step=10000,
        eval_interval_step=10000,
        eval_after_train=True,
    ),
    # DeepSpeed has no poly scheduler; the engine applies this one.
    learning_rate=dict(
        type='poly',
        params=dict(base_lr=BASE_LR, power=0.9, max_iters=MAX_ITERS),
    ),
    deepspeed={
        'train_micro_batch_size_per_gpu': MICRO_BATCH,
        'gradient_accumulation_steps': 1,
        'optimizer': {
            'type': 'AdamW',
            # torch_adam: torch.optim.AdamW as in the original (no fused-kernel build needed)
            'params': {'lr': BASE_LR, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.05,
                       'torch_adam': True},
        },
        'gradient_clipping': 0.0,
        'zero_optimization': {'stage': 0},
        'fp16': {'enabled': False},
        'bf16': {'enabled': False},
        'steps_per_print': 1000000,
        'wall_clock_breakdown': False,
    },
)

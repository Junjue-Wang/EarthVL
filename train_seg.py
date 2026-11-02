"""Train or evaluate the stage-1 segmentation model of EarthVLNet (Semantic-FPN + ConvNeXt-L).

Training (2 GPUs x 8 images, see scripts/train_seg.sh):
    deepspeed --num_gpus 2 train_seg.py --config_path=seg.sfpn_convnext_large \
        --model_dir=./log/seg/sfpn_convnext_large

Evaluation (mIoU on the Val split of EarthVQA + EarthVL-GLOBAL):
    python train_seg.py --config_path=seg.sfpn_convnext_large --model_dir=./log/seg/eval \
        --eval_only --ckpt ./log/seg/sfpn_convnext_large/model-30000.pth

Trailing ``key value`` pairs override the config, e.g. ``data.test.params.batch_size 2``.
"""
import json
import os
import sys

import numpy as np
import prettytable as pt
import torch
import torch.distributed as dist
from tqdm import tqdm

# Register only the segmentation model and loader (the LLM modules are not needed here).
import data.earthvl_seg  # noqa: F401
import module.seg.semantic_fpn  # noqa: F401
from data.earthvl_seg import CLASS_NAMES
from module.seg.convnext import SKIP_PRETRAINED_ENV


def segmentation_scores(confusion_matrix, class_names=CLASS_NAMES):
    """IoU/F1/precision/recall per class, mIoU, mF1 and OA from a [gt, pred] confusion matrix."""
    cm = confusion_matrix.astype(np.float64)
    eps = 1e-7
    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    iou = tp / (gt + pred - tp + eps)
    precision = tp / (pred + eps)
    recall = tp / (gt + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    names = list(class_names) if len(class_names) == len(tp) else [str(i) for i in range(len(tp))]
    return dict(
        mIoU=float(iou.mean()),
        mF1=float(f1.mean()),
        OA=float(tp.sum() / (cm.sum() + eps)),
        IoU={n: float(v) for n, v in zip(names, iou)},
        F1={n: float(v) for n, v in zip(names, f1)},
        precision={n: float(v) for n, v in zip(names, precision)},
        recall={n: float(v) for n, v in zip(names, recall)},
        num_pixels=int(cm.sum()),
    )


def format_scores(scores):
    table = pt.PrettyTable(['class', 'IoU', 'F1', 'precision', 'recall'])
    for name in scores['IoU']:
        table.add_row([name] + ['{:.4f}'.format(scores[k][name]) for k in ('IoU', 'F1', 'precision', 'recall')])
    table.add_row(['mean', '{:.4f}'.format(scores['mIoU']), '{:.4f}'.format(scores['mF1']), '', ''])
    return '{}\nmIoU {:.4f} | mF1 {:.4f} | OA {:.4f}'.format(table.get_string(), scores['mIoU'], scores['mF1'],
                                                            scores['OA'])


@torch.no_grad()
def evaluate_fn(ctx, dataloader):
    """mIoU over the loader's masks (label -1 ignored); counts are all-reduced across ranks."""
    model = ctx.model
    model.eval()
    num_classes = int(getattr(model, 'config', {}).get('classes', len(CLASS_NAMES)))
    cm = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=ctx.device)
    for img, target in tqdm(dataloader, disable=not ctx.is_main, desc='seg eval'):
        if 'mask' not in target:
            raise ValueError('Evaluation needs masks: point data.test.params.splits to a split with masks_png (Val).')
        probs = model(img.to(ctx.device, non_blocking=True))
        if isinstance(probs, (tuple, list)):
            probs = probs[0]
        pred = probs.argmax(dim=1)
        gt = target['mask'].to(ctx.device, non_blocking=True)
        valid = (gt >= 0) & (gt < num_classes)
        cm += torch.bincount(gt[valid] * num_classes + pred[valid], minlength=num_classes * num_classes)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(cm)
    scores = segmentation_scores(cm.view(num_classes, num_classes).cpu().numpy())
    scores['global_step'] = int(ctx.global_step)
    if ctx.is_main:
        ctx.logger.info('Segmentation results at step %d:\n%s', ctx.global_step, format_scores(scores))
        os.makedirs(ctx.model_dir, exist_ok=True)
        with open(os.path.join(ctx.model_dir, 'miou-{}.json'.format(ctx.global_step)), 'w') as f:
            json.dump(scores, f, indent=2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--eval_only' in argv:
        # the trained checkpoint replaces the ImageNet initialisation anyway
        os.environ.setdefault(SKIP_PRETRAINED_ENV, '1')
    from engine.trainer import main as engine_main
    return engine_main(evaluate_fn=evaluate_fn, argv=argv)


if __name__ == '__main__':
    main()

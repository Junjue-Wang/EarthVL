"""Extract the stage-1 inputs of EarthVLNet with a trained Semantic-FPN (ConvNeXt-L).

For every image this writes ``<out_dir>/<subset>/<split>/<image_id>.hdf5`` with

* ``feature``:   float32 ``[1536, H/32, W/32]`` (``[1536, 32, 32]`` for 1024x1024 images),
                 the LayerNorm'd last ConvNeXt stage;
* ``pred_mask``: uint8 ``[H, W]``, ``argmax + 1`` (1..8 = background, building, road,
                 water, barren, forest, agriculture, playground).

This is the format read by the VQA loaders (``feat['feature']``, ``feat['pred_mask'] - 1``).
Existing files are skipped, so an interrupted run can be resumed. No masks are needed.
The VQA configs read only the Train and Test features (the default ``--splits``); Val is optional.

Example (see scripts/extract_features.sh):
    python tools/extract_seg_features.py --config seg.sfpn_convnext_large \
        --ckpt ./log/seg/sfpn_convnext_large/model-30000.pth \
        --data_root ./EarthVLSet --out_dir ./features/sfpn_convnext_large \
        --subsets EarthVQA EarthVL-GLOBAL --splits Train Test --batch_size 4

``--ckpt`` accepts a DeepSpeed checkpoint directory written by train_seg.py, the
authors' ever checkpoint (``{'model': ...}``) or a plain state dict.
"""
import argparse
import contextlib
import copy
import logging
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ever.core.config import import_config  # noqa: E402

from data.earthvl_seg import SUBSETS, EarthVLSegDataset  # noqa: E402
from module.seg.checkpoint_io import load_weights  # noqa: E402
from module.seg.semantic_fpn import SemanticFPN  # noqa: E402

logger = logging.getLogger('extract_seg_features')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--config', default='seg.sfpn_convnext_large',
                   help="config name under configs/ (e.g. 'seg.sfpn_convnext_large') or a .py path")
    p.add_argument('--ckpt', required=True, help='trained Semantic-FPN checkpoint (DeepSpeed dir, ever .pth, state dict)')
    p.add_argument('--data_root', default='./EarthVLSet', help='EarthVLSet root (Hugging Face layout)')
    p.add_argument('--out_dir', default='./features/sfpn_convnext_large')
    p.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    p.add_argument('--splits', nargs='+', default=['Train', 'Test'],
                   help='the VQA configs read only Train and Test; Val is optional')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--fp16', action='store_true',
                   help='fp16 autocast on CUDA (faster; the saved feature is still float32)')
    p.add_argument('--device', default=None, help="'cuda', 'cuda:1' or 'cpu' (default: cuda if available)")
    p.add_argument('--num_shards', type=int, default=1, help='split the work over several processes/GPUs')
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--compress', action='store_true', help='gzip pred_mask (about 20x smaller, slower to write)')
    p.add_argument('--non_strict', action='store_true', help='allow missing/unexpected checkpoint keys')
    return p.parse_args(argv)


def resolve_config(name_or_path):
    if name_or_path.endswith('.py'):
        path = name_or_path
    else:
        path = os.path.join(REPO_ROOT, 'configs', *name_or_path.split('.')) + '.py'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Config not found: {path}')
    return import_config(os.path.abspath(path))


def build_model(cfg):
    if cfg['model']['type'] != 'SemanticFPN':
        raise ValueError(f"Expected a SemanticFPN config, got model.type={cfg['model']['type']}")
    params = copy.deepcopy(cfg['model']['params'])
    convnext = params.get('encoder', {}).get('convnext', None)
    if convnext is not None and convnext.get('init_cfg', None):
        # the checkpoint overwrites the backbone: skip the ImageNet download
        init_cfg = convnext['init_cfg']
        init_cfg = init_cfg if isinstance(init_cfg, (list, tuple)) else [init_cfg]
        convnext['init_cfg'] = [c for c in init_cfg if c.get('type') != 'Pretrained']
    params['sync_bn'] = False
    return SemanticFPN(params)


def output_path(out_dir, item):
    stem = os.path.splitext(item['imagen'])[0]
    return os.path.join(out_dir, item['subset'], item['split'], stem + '.hdf5')


def write_hdf5(path, feature, pred_mask, compress=False):
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('feature', data=feature)
        f.create_dataset('pred_mask', data=pred_mask, **(dict(compression='gzip', compression_opts=4) if compress else {}))
    os.replace(tmp, path)  # never leave a truncated .hdf5 behind


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('--shard_id must be in [0, --num_shards)')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_fp16 = args.fp16 and device.type == 'cuda'
    if args.fp16 and not use_fp16:
        logger.warning('--fp16 is ignored on %s', device)

    cfg = resolve_config(args.config)
    transforms = cfg['data']['test']['params']['transforms']  # same normalisation as training
    model = build_model(cfg)
    load_weights(model, args.ckpt, strict=not args.non_strict)
    model.to(device).eval()

    total_written, total_skipped = 0, 0
    t0 = time.time()
    for subset in args.subsets:
        for split in args.splits:
            dataset = EarthVLSegDataset(data_root=args.data_root, subsets=[subset], splits=[split],
                                        transforms=transforms, load_mask=False, return_meta=True)
            items = dataset.items[args.shard_id::args.num_shards]
            todo = [it for it in items if not os.path.exists(output_path(args.out_dir, it))]
            total_skipped += len(items) - len(todo)
            logger.info('%s/%s: %d images, %d to extract, %d already done', subset, split, len(items), len(todo),
                        len(items) - len(todo))
            if not todo:
                continue
            os.makedirs(os.path.join(args.out_dir, subset, split), exist_ok=True)
            dataset.items = todo
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=device.type == 'cuda')
            for img, meta in tqdm(loader, desc=f'{subset}/{split}'):
                if img.shape[-1] % 32 or img.shape[-2] % 32:
                    raise ValueError(f'Image size {tuple(img.shape[-2:])} must be divisible by 32')
                img = img.to(device, non_blocking=True)
                amp = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()
                with torch.inference_mode(), amp:
                    probs, feature = model(img)
                pred_mask = (probs.argmax(dim=1) + 1).to(torch.uint8).cpu().numpy()
                feature = feature.float().cpu().numpy()
                for i in range(feature.shape[0]):
                    item = dict(imagen=meta['imagen'][i], subset=subset, split=split)
                    write_hdf5(output_path(args.out_dir, item), feature[i], pred_mask[i], compress=args.compress)
                    total_written += 1
    logger.info('Done: %d written, %d skipped, %.1f s -> %s', total_written, total_skipped, time.time() - t0,
                os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()

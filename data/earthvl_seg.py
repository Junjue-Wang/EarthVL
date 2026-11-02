"""Land-cover segmentation data of EarthVLSet (Hugging Face layout).

    <data_root>/EarthVQA/{Train,Val,Test}/images_png/<id>.png
    <data_root>/EarthVQA/{Train,Val}/masks_png/<id>.png
    <data_root>/EarthVL-GLOBAL/{Train,Val,Test}/images_png/<id>.png
    <data_root>/EarthVL-GLOBAL/{Train,Val}/masks_png/<id>.png

Mask values: 0 = no-data (ignored), 1..8 = background, building, road, water,
barren, forest, agriculture, playground. They are remapped to training labels
0..7 with -1 ignored; the LoveDA value 9 (pool) maps to water, as in the
authors' LoveDA loader.
"""
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

import ever as er
from ever.data.distributed import DistributedNonOverlapSeqSampler, StepDistributedSampler
from ever.interface import ConfigurableMixin

logger = logging.getLogger(__name__)

SUBSETS = ('EarthVQA', 'EarthVL-GLOBAL')
SPLITS = ('Train', 'Val', 'Test')
IMAGE_DIRNAME = 'images_png'
MASK_DIRNAME = 'masks_png'

CLASS_NAMES = ('Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural', 'Playground')

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
    Playground=(165, 0, 165),
)

# mask value -> training label (same as LABEL_MAP of the authors' LoveDA loader)
LABEL_MAP = OrderedDict(
    IGNORE=-1,
    Background=0,
    Building=1,
    Road=2,
    River=3,
    Barren=4,
    Forest=5,
    Agricultural=6,
    Playground=7,
    Pool=3,
)
_LUT = np.full(256, -1, dtype=np.int64)
_LUT[:len(LABEL_MAP)] = list(LABEL_MAP.values())
_LUT_T = torch.from_numpy(_LUT)


def reclassify(mask):
    """Map raw mask values (uint8, numpy or torch) to int64 labels with -1 = ignore."""
    if isinstance(mask, torch.Tensor):
        return _LUT_T[mask.long().clamp(0, 255)]
    return _LUT[np.clip(np.asarray(mask), 0, 255).astype(np.int64)]


def _sort_key(fname):
    stem = os.path.splitext(fname)[0]
    return (0, int(stem), '') if stem.isdigit() else (1, 0, stem)


def read_image(path):
    """RGB uint8 array; RGBA/palette/grey PNGs are converted (alpha is dropped, as ``[:, :, :3]`` did)."""
    with Image.open(path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.asarray(img).copy()


def read_mask(path):
    with Image.open(path) as m:
        mask = np.array(m)
    if mask.ndim != 2:
        raise ValueError(f'{path}: expected a single-channel mask, got shape {mask.shape}')
    return mask


class EarthVLSegDataset(Dataset):
    """Images (and masks) of the given subsets and splits.

    Args:
        data_root: EarthVLSet root in the Hugging Face layout.
        subsets: e.g. ('EarthVQA', 'EarthVL-GLOBAL').
        splits: e.g. ('Train',).
        transforms: albumentations transform applied to ``image`` (and ``mask``).
        load_mask: True requires masks, False never loads them, None loads them when the folder exists.
        return_meta: add ``imagen``/``subset``/``split`` strings to the target dict.
        image_dir / mask_dir: optional explicit folders (lists) instead of the layout above.
    """

    def __init__(self, data_root='./EarthVLSet', subsets=SUBSETS, splits=('Train',), transforms=None,
                 load_mask=None, return_meta=True, image_dir=None, mask_dir=None):
        self.transforms = transforms
        self.return_meta = return_meta
        self.items = []
        if image_dir is not None:
            image_dirs = [image_dir] if isinstance(image_dir, str) else list(image_dir)
            mask_dirs = mask_dir if isinstance(mask_dir, (list, tuple)) else [mask_dir] * len(image_dirs)
            for img_d, mask_d in zip(image_dirs, mask_dirs):
                self._add_dir(img_d, mask_d, load_mask, subset='', split='')
        else:
            subsets = [subsets] if isinstance(subsets, str) else list(subsets)
            splits = [splits] if isinstance(splits, str) else list(splits)
            for subset in subsets:
                for split in splits:
                    root = os.path.join(data_root, subset, split)
                    self._add_dir(os.path.join(root, IMAGE_DIRNAME), os.path.join(root, MASK_DIRNAME),
                                  load_mask, subset=subset, split=split)
        if len(self.items) == 0:
            raise FileNotFoundError(f'No images found for data_root={data_root}, subsets={subsets}, splits={splits}, '
                                    f'image_dir={image_dir}')

    def _add_dir(self, image_dir, mask_dir, load_mask, subset, split):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'Image folder not found: {image_dir}')
        has_masks = mask_dir is not None and os.path.isdir(mask_dir)
        if load_mask and not has_masks:
            raise FileNotFoundError(f'Mask folder not found: {mask_dir}')
        use_masks = has_masks if load_mask is None else bool(load_mask)
        names = sorted((f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.tif', '.tiff'))),
                       key=_sort_key)
        for name in names:
            mask_path = None
            if use_masks:
                mask_path = os.path.join(mask_dir, name)
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(f'Missing mask for {os.path.join(image_dir, name)}: {mask_path}')
            self.items.append(dict(image_path=os.path.join(image_dir, name), mask_path=mask_path,
                                   imagen=name, subset=subset, split=split))
        logger.info('%s: %d images (masks: %s)', image_dir, len(names), use_masks)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = read_image(item['image_path'])
        mask = read_mask(item['mask_path']) if item['mask_path'] is not None else None
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask) if mask is not None else self.transforms(image=image)
            image = blob['image']
            mask = blob.get('mask', mask)
        target = dict()
        if mask is not None:
            # remap after the geometric transforms: they see raw uint8 values (nearest interpolation)
            target['mask'] = reclassify(mask)
        if self.return_meta:
            target.update(imagen=item['imagen'], subset=item['subset'], split=item['split'])
        return image, target


class _AutoStepSampler(Sampler):
    """Single-process stand-in for ever's StepDistributedSampler (which needs torch.distributed)."""

    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        self.seed = seed
        self.step = 0

    def set_step(self, step):
        self.step = step

    def set_epoch(self, epoch):
        self.step = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.step)
        self.step += 1  # reshuffle on the next pass even if nobody calls set_step()
        return iter(torch.randperm(len(self.dataset), generator=g).tolist())

    def __len__(self):
        return len(self.dataset)


class _AutoStepDistributedSampler(StepDistributedSampler):
    """ever's StepDistributedSampler that also reshuffles on every new pass over the data."""

    def set_epoch(self, epoch):
        self.step = epoch

    def __iter__(self):
        indices = list(super().__iter__())
        self.step += 1
        return iter(indices)


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


@er.registry.DATALOADER.register()
class EarthVLSegLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        cfg = self.config
        dataset = EarthVLSegDataset(
            data_root=cfg.data_root, subsets=cfg.subsets, splits=cfg.splits, transforms=cfg.transforms,
            load_mask=True if cfg.training else cfg.load_mask,
            return_meta=not cfg.training, image_dir=cfg.image_dir, mask_dir=cfg.mask_dir)
        if cfg.training:
            sampler = _AutoStepDistributedSampler(dataset, seed=cfg.seed) if _dist_ready() \
                else _AutoStepSampler(dataset, seed=cfg.seed)
        else:
            # every rank evaluates a disjoint, ordered shard; evaluate_fn all-reduces the counts
            sampler = DistributedNonOverlapSeqSampler(dataset) if _dist_ready() else SequentialSampler(dataset)
        super(EarthVLSegLoader, self).__init__(dataset,
                                               cfg.batch_size,
                                               sampler=sampler,
                                               num_workers=cfg.num_workers,
                                               pin_memory=torch.cuda.is_available(),
                                               drop_last=False)

    def set_default_config(self):
        self.config.update(dict(
            data_root='./EarthVLSet',
            subsets=list(SUBSETS),
            splits=['Train'],
            image_dir=None,
            mask_dir=None,
            load_mask=None,
            transforms=None,
            training=False,
            batch_size=4,
            num_workers=4,
            seed=0,
        ))

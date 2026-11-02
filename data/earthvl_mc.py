"""Multiple-choice VQA data for EarthVLSet (EarthVQA + EarthVL-GLOBAL).

QA files use the Hugging Face EarthVLSet format::

    {"7541.png": [{"Type": "Basic Judging", "Question": "Are there any roads in this scene?", "Answer": "Yes."}, ...]}

With ``feat_load=True`` every image is read from ``<image_feat_dir>/<image_id>.hdf5``, written by the stage-1
segmentation network: dataset ``feature`` ([1536, 32, 32] float32) and ``pred_mask`` ([1024, 1024] uint8, class
index + 1, i.e. 1..8).

QAs without an answer ('' or null) keep the answer '': the EarthVQA Test labels are withheld for the EarthVQA
challenge on Codabench, so such QAs are predicted but not scored (``module/metric.py``). Training data
(``training=True``) must have an answer for every QA. ``exclude_images`` lists image file names (the keys of the QA
files, e.g. ``'9118.png'``) whose QAs are not loaded.
"""
import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch.distributed as dist
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import ever as er
from ever.core.dist import is_main_process
from ever.data.distributed import StepDistributedSampler
from ever.interface import ConfigurableMixin

from data import vocab

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


def reclassify(cls):
    """Map the values of a ground-truth mask PNG (0 = ignore, 1..8 = classes) to -1..7 via LABEL_MAP."""
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls) * label, new_cls)
    return new_cls


def _as_list(value, n):
    """A single directory is shared by all `n` QA files; a list must give one entry per QA file."""
    if isinstance(value, str):
        return [value] * n if n > 1 else [value]
    value = list(value)
    if len(value) != n:
        raise ValueError(f'Expected one directory per QA file ({n}), got {len(value)}: {value}')
    return value


class EarthVLDataset(Dataset):
    # Kept as class attributes for backward compatibility (EarthVLDataset.QUESTION_TYPES etc.).
    QUESTIONS = vocab.QUESTIONS
    QUESTION_VOC = vocab.QUESTION_VOC
    QUESTION_TYPES = vocab.QUESTION_TYPES
    ANSWER_VOC = vocab.ANSWER_VOC

    def __init__(self, image_dir='', mask_dir='', qa_path='', image_feat_dir='', encoding=True, vqa_transforms=None,
                 image_load=True, mask_load=True, feat_load=False, pred_mask_load=False, feat_shuffle=False,
                 select_type=None, training=True, exclude_images=None):
        self.item_list = []
        self.question_voc = []
        self.ans_voc = []
        self.encoding = encoding
        self.vqa_transforms = vqa_transforms
        self.feat_shuffle = feat_shuffle  # unused, kept for config compatibility
        self.pred_mask_load = pred_mask_load
        self.training = training  # every training QA needs an answer
        if isinstance(exclude_images, str):
            exclude_images = [exclude_images]
        self.exclude_images = set(exclude_images or ())
        self.excluded_images = set()  # images of exclude_images found in the QA files
        self.num_excluded_qas = 0
        if isinstance(qa_path, str):
            qa_path = [qa_path]
        image_dir = _as_list(image_dir, len(qa_path))
        mask_dir = _as_list(mask_dir, len(qa_path))
        image_feat_dir = _as_list(image_feat_dir, len(qa_path))
        for image_d, mask_d, qa_p, image_f in zip(image_dir, mask_dir, qa_path, image_feat_dir):
            self.generate_pairs(image_d, mask_d, qa_p, image_f, select_type)

        self.question_voc = list(set(self.question_voc))
        self.ans_voc = list(set(self.ans_voc))
        self.image_load = image_load
        self.mask_load = mask_load
        self.feat_load = feat_load
        if is_main_process():
            if self.exclude_images:
                not_found = len(self.exclude_images - self.excluded_images)
                print(f'Excluded {len(self.excluded_images)} images ({self.num_excluded_qas} QAs) listed in '
                      f'exclude_images' + (f'; {not_found} listed images are not in the QA files' if not_found else ''))
            print(f'The {len(self.item_list)} QAs are loaded!')
            num_unanswered = sum(not item['ans'] for item in self.item_list)
            if num_unanswered:
                print(f'{num_unanswered} QAs have no answer: they are predicted but not scored.')
            print(f'The question voc has {len(self.question_voc)} words!')
            print(f'The answer voc has {len(self.ans_voc)} words!')

    def generate_pairs(self, image_dir, mask_dir, qa_path, image_feat_dir=None, select_type=None):
        with open(qa_path, 'r', encoding='utf-8') as f:
            qas_dict = json.load(f)
        for imagen, qas_list in qas_dict.items():
            if imagen in self.exclude_images:
                self.excluded_images.add(imagen)
                self.num_excluded_qas += sum(select_type is None or qa['Type'] in select_type for qa in qas_list)
                continue
            for qa_dict in qas_list:
                questype, ques, ans = qa_dict['Type'], qa_dict['Question'], qa_dict['Answer']
                if select_type is not None and questype not in select_type:
                    continue
                words = ques.strip('?').split(' ')
                self.question_voc += words
                ans = '' if ans is None else str(ans)
                if not ans.strip():
                    # no ground-truth answer (e.g. the withheld EarthVQA Test labels): kept as '', not scored
                    if self.training:
                        raise ValueError(f'{qa_path}: the question {ques!r} of {imagen} has no answer; training '
                                         'needs an answer for every QA (QAs without answers can only be evaluated, '
                                         'with training=False)')
                    ans = ''
                else:
                    # Answers are compared and generated with a trailing period ('Yes.', '3.', '10%-20%.').
                    if not ans.endswith('.'):
                        ans = ans + '.'
                    self.ans_voc.append(ans)
                self.item_list.append(dict(image_path=os.path.join(image_dir, imagen),
                                           mask_path=os.path.join(mask_dir, imagen),
                                           feat_path=os.path.join(image_feat_dir, imagen.replace('png', 'hdf5')),
                                           questype=questype,
                                           ques=words,
                                           ans=ans,
                                           ))

    def __getitem__(self, idx):
        item = self.item_list[idx]
        image_path, mask_path, feat_path = item['image_path'], item['mask_path'], item['feat_path']
        questype, ques, ans = item['questype'], item['ques'], item['ans']
        mask = []
        vqa_image = []
        if self.image_load:
            vqa_image = imread(image_path)[:, :, :3]

        if self.mask_load:
            mask = reclassify(imread(mask_path).astype(np.int64))

        if self.feat_load:
            with h5py.File(feat_path, 'r') as feat:
                vqa_image = np.array(feat['feature']).transpose([1, 2, 0])  # C, H, W -> H, W, C
                if self.pred_mask_load:
                    mask = np.array(feat['pred_mask']).astype(np.int64) - 1  # 1..8 -> 0..7

        if self.vqa_transforms is not None:
            # the transforms also rewrite direction words of the answer (module/data_augs.py)
            if self.pred_mask_load:
                blob = self.vqa_transforms(image=vqa_image, answer=ans, mask=mask)
                vqa_image, ans, mask = blob['image'], blob['answer'], blob['mask']
            else:
                blob = self.vqa_transforms(image=vqa_image, answer=ans)
                vqa_image, ans = blob['image'], blob['answer']
        if self.feat_load:
            vqa_image = vqa_image.transpose([2, 0, 1])

        question_len = len(ques)
        if self.encoding:
            encoded_ques = np.zeros(len(self.QUESTION_VOC)).astype(np.int64)
            for i, word in enumerate(ques):
                encoded_ques[i] = EarthVLDataset.QUESTION_VOC.index(word)
            ques = encoded_ques
            ans = EarthVLDataset.ANSWER_VOC.index(ans) if ans else -1  # -1: no ground-truth answer
        else:
            ques = ' '.join(ques) + '?'
            ans = str(ans)

        return vqa_image, dict(imagen=os.path.basename(image_path), vqaimagen=os.path.basename(image_path),
                               seg_mask=mask, question=ques, questype=questype, question_len=question_len, answer=ans)

    def __len__(self):
        return len(self.item_list)


class _SingleProcessStepSampler(StepDistributedSampler):
    """StepDistributedSampler with one replica, usable without an initialised torch.distributed process group."""

    def __init__(self, dataset, seed=0):
        DistributedSampler.__init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=False)
        self.step = 0


def build_step_sampler(dataset):
    """Sampler of the original training code, used for both the train and the test loader.

    ``StepDistributedSampler`` shuffles with seed ``step`` (set by the trainer) and pads the index list so that all
    ranks get the same number of samples (together with ``drop_last=True`` in the loader, a few test QAs can be
    repeated or skipped). It needs an initialised process group; without one, a single-replica sampler yielding
    exactly the indices of a world-size-1 run is returned.
    """
    if dist.is_available() and dist.is_initialized():
        return StepDistributedSampler(dataset)
    return _SingleProcessStepSampler(dataset)


@er.registry.DATALOADER.register()
class EarthVLLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        dataset = EarthVLDataset(self.config.image_dir, self.config.mask_dir,
                                 self.config.qa_path, self.config.image_feat_dir,
                                 self.config.encoding, self.config.vqa_transforms,
                                 self.config.image_load, self.config.mask_load, self.config.feat_load,
                                 self.config.pred_mask_load, self.config.feat_shuffle, self.config.select_type,
                                 training=self.config.training,
                                 exclude_images=self.config.exclude_images,
                                 )
        sampler = build_step_sampler(dataset)

        super(EarthVLLoader, self).__init__(dataset,
                                            self.config.batch_size,
                                            sampler=sampler,
                                            num_workers=self.config.num_workers,
                                            pin_memory=True,
                                            drop_last=True
                                            )

    def set_default_config(self):
        self.config.update(dict(
            encoding=True,
            image_load=True,
            mask_load=False,
            feat_load=False,
            feat_shuffle=False,
            pred_mask_load=False,
            vqa_transforms=None,
            common_transforms=None,
            image_dir='',
            mask_dir='',
            qa_path='',
            image_feat_dir='',
            image_transforms=None,
            batch_size=3,
            num_workers=0,
            training=True,
            select_type=None,
            exclude_images=None,
        ))

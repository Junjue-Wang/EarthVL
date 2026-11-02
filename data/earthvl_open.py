"""Open-ended VQA data for EarthVL-GLOBAL (EarthVLSet ``earthvl_global_openended``).

Together with the open-ended metrics (``module.metric.Metric``: BLEU, ROUGE-L and CIDEr, with the coco-caption code in
``score/``), this loader is an evaluation tool for the open-ended split. This release contains no open-ended model,
config or training script.

QA files use the Hugging Face EarthVLSet format (several reference answers per question)::

    {"7541.png": {"Describe and give me some advice about the greening renovation.": ["answer 1", "answer 2", ...]}}

In training mode every (question, reference answer) pair is one sample; in test mode one sample holds all references
of a question, joined by ``////next sentence////``.

Sentences are tokenised with spaCy (``en_core_web_sm``). spaCy is imported lazily, only when the word vocabulary is
built or a sentence is encoded (``qa_encoding`` / ``ans_encoding``), so the multiple-choice pipeline never needs it::

    pip install spacy && python -m spacy download en_core_web_sm

The vocabulary is cached in ``vocabulary_path`` (default ``./vocabulary.pkl``): it is built from the loaded answers
and questions (words with frequency >= 5) when the file does not exist, and loaded otherwise. Build it from the
training split and reuse it for evaluation.
"""
import json
import logging
import os
import pickle
from collections import Counter, defaultdict

import h5py
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, RandomSampler

import ever as er
from ever.core.dist import is_main_process
from ever.interface import ConfigurableMixin

from data.earthvl_mc import build_step_sampler, reclassify

logger = logging.getLogger(__name__)

SPACY_MODEL = 'en_core_web_sm'
# Maximum encoded answer length (in tokens, including <bos>/<eos>) assumed when a cached vocabulary is loaded. It is
# the value of the original EarthVL-GLOBAL training set; a freshly built vocabulary computes it from the data instead.
CACHED_VOCAB_ANSWER_MAXLEN = 223
# Questions are encoded to a fixed length of 16 ids (<bos> + at most 14 tokens + <eos>). All EarthVL-GLOBAL questions
# have at most 13 spaCy tokens; longer questions are truncated (the original code raised an IndexError).
QUESTION_MAXLEN = 16

_spacy_nlp = None


def spacy_tokenize(text):
    """Token strings of `text` from the spaCy English tokenizer (the model is loaded on first use)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError('The open-ended VQA loader needs spaCy: '
                              f'pip install spacy && python -m spacy download {SPACY_MODEL}') from e
        _spacy_nlp = spacy.load(SPACY_MODEL)
    return [tok.text for tok in _spacy_nlp.tokenizer(text)]


def _as_list(value, n):
    if isinstance(value, str):
        return [value] * n if n > 1 else [value]
    value = list(value)
    if len(value) != n:
        raise ValueError(f'Expected one directory per QA file ({n}), got {len(value)}: {value}')
    return value


class EarthVLOpenQADataset(Dataset):
    def __init__(self, image_dir='', mask_dir='', qa_path='', image_feat_dir='', qa_encoding=True, ans_encoding=True,
                 vqa_transforms=None, image_load=True, mask_load=True, feat_load=False, pred_mask_load=False,
                 feat_shuffle=False, select_type=None, training=True, vocabulary_path='./vocabulary.pkl',
                 question_maxlen=QUESTION_MAXLEN):
        self.item_list = []
        self.question = []
        self.question_voc = []
        self.ans_voc = []
        self.qa_encoding = qa_encoding
        self.ans_encoding = ans_encoding
        self.vqa_transforms = vqa_transforms
        self.feat_shuffle = feat_shuffle  # unused, kept for config compatibility
        self.pred_mask_load = pred_mask_load
        self.vocabulary = defaultdict()
        self.vocabulary_path = vocabulary_path
        self.question_maxlen = question_maxlen
        self.sentence_maxlen = 0
        self.training = training
        self._warned_truncation = False
        if isinstance(qa_path, str):
            qa_path = [qa_path]
        image_dir = _as_list(image_dir, len(qa_path))
        mask_dir = _as_list(mask_dir, len(qa_path))
        image_feat_dir = _as_list(image_feat_dir, len(qa_path))

        for image_d, mask_d, qa_p, image_f in zip(image_dir, mask_dir, qa_path, image_feat_dir):
            self.generate_pairs(image_d, mask_d, qa_p, image_f, select_type)

        self.question_voc = list(set(self.question_voc))
        self.image_load = image_load
        self.mask_load = mask_load
        self.feat_load = feat_load
        self.get_vocabulary(self.question, self.ans_voc)
        if is_main_process():
            print(f'The {len(self.item_list)} QAs are loaded!')

    def generate_pairs(self, image_dir, mask_dir, qa_path, image_feat_dir=None, select_type=None):
        # select_type is ignored: open-ended questions have no question type
        with open(qa_path, 'r', encoding='utf-8') as f:
            qas_dict = json.load(f)
        for imagen, qas_list in qas_dict.items():
            for ques, ans in qas_list.items():
                words = ques.strip('.').split(' ')
                self.question_voc += words
                self.question.append(ques)
                self.ans_voc.extend(ans)
                sample = dict(image_path=os.path.join(image_dir, imagen),
                              mask_path=os.path.join(mask_dir, imagen),
                              feat_path=os.path.join(image_feat_dir, imagen.replace('png', 'hdf5')),
                              questype='openend',
                              ques=words)
                if self.training:
                    for ans_i in ans:
                        self.item_list.append(dict(sample, ans=ans_i))
                else:
                    self.item_list.append(dict(sample, ans=ans))

    def get_vocabulary(self, questions, answers):
        if os.path.exists(self.vocabulary_path):
            with open(self.vocabulary_path, 'rb') as fr:
                self.vocabulary = pickle.load(fr, encoding='bytes')
            self.sentence_maxlen = CACHED_VOCAB_ANSWER_MAXLEN
        else:
            counter = Counter()
            for sample in answers:
                words = spacy_tokenize(sample.strip('.').lower())
                counter.update(words)
                self.sentence_maxlen = max(self.sentence_maxlen, len(words))
            for q in questions:
                counter.update(spacy_tokenize(q.strip('.').lower()))
            itos = ['<pad>', '<bos>', '<eos>', '<unk>']
            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            for word, freq in words_and_frequencies:
                if freq < 5:  # min_freq
                    break
                itos.append(word)
            self.vocabulary.update({tok: i for i, tok in enumerate(itos)})
            self.sentence_maxlen += 2  # <bos>, <eos>
            with open(self.vocabulary_path, 'wb') as f:
                pickle.dump(self.vocabulary, f)
        if is_main_process():
            logger.info('Vocabulary: %d words (%s), max answer length %d',
                        len(self.vocabulary), self.vocabulary_path, self.sentence_maxlen)

    def word2vector(self, caption, encode_len):
        """Encode `caption` as [<bos>, word ids..., <eos>, <pad>...] of length `encode_len`."""
        vector = np.zeros(encode_len).astype(np.int64)
        element = spacy_tokenize(caption.strip('.').lower())
        if len(element) > encode_len - 2:
            if not self._warned_truncation:
                logger.warning('Truncating a sentence of %d tokens to %d tokens: %r',
                               len(element), encode_len - 2, caption)
                self._warned_truncation = True
            element = element[:encode_len - 2]
        for i, word in enumerate(element):
            if word in self.vocabulary.keys():
                vector[i + 1] = self.vocabulary[word]
            else:
                vector[i + 1] = self.vocabulary['<unk>']
        vector[0] = self.vocabulary['<bos>']
        vector[len(element) + 1] = self.vocabulary['<eos>']
        return vector

    def vector2word(self, predict_vect):
        """Decode word ids until <eos> (id 2), skipping <pad> (id 0)."""
        caption_list = []
        temp_dict = {val: key for key, val in self.vocabulary.items()}
        for vect_index in predict_vect.tolist():
            if vect_index == 2:
                break
            if vect_index != 0:
                caption_list.append(temp_dict[vect_index])
        return ' '.join(caption_list)

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
                mask = np.array(feat['pred_mask']).astype(np.int64) - 1  # 1..8 -> 0..7

        if self.vqa_transforms is not None:
            if self.pred_mask_load:
                blob = self.vqa_transforms(image=vqa_image, answer=ans, mask=mask)
                vqa_image, ans, mask = blob['image'], blob['answer'], blob['mask']
            else:
                blob = self.vqa_transforms(image=vqa_image, answer=ans)
                vqa_image, ans = blob['image'], blob['answer']
        if self.feat_load:
            vqa_image = vqa_image.transpose([2, 0, 1])

        question_len = len(ques)
        ques = ' '.join(ques)
        if self.training:
            if self.qa_encoding:
                ques = self.word2vector(ques, self.question_maxlen)
            if self.ans_encoding:
                ans = self.word2vector(ans, self.sentence_maxlen)
        else:
            ans = '////next sentence////'.join([caption.strip('.').lower() for caption in ans])
            if self.qa_encoding:
                ques = self.word2vector(ques, self.question_maxlen)
        return vqa_image, dict(imagen=os.path.basename(image_path), vqaimagen=os.path.basename(image_path),
                               seg_mask=mask, question=ques, questype=questype, question_len=question_len, answer=ans)

    def __len__(self):
        return len(self.item_list)


@er.registry.DATALOADER.register()
class EarthVLOpenQALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        dataset = EarthVLOpenQADataset(self.config.image_dir, self.config.mask_dir,
                                       self.config.qa_path, self.config.image_feat_dir,
                                       self.config.qa_encoding, self.config.ans_encoding, self.config.vqa_transforms,
                                       self.config.image_load, self.config.mask_load, self.config.feat_load,
                                       self.config.pred_mask_load, self.config.feat_shuffle, self.config.select_type,
                                       self.config.training,
                                       vocabulary_path=self.config.vocabulary_path,
                                       question_maxlen=self.config.question_maxlen,
                                       )
        if self.config.evaluate:
            sampler = RandomSampler(dataset)
        else:
            sampler = build_step_sampler(dataset)

        super(EarthVLOpenQALoader, self).__init__(dataset,
                                                  self.config.batch_size,
                                                  sampler=sampler,
                                                  num_workers=self.config.num_workers,
                                                  pin_memory=True,
                                                  drop_last=True
                                                  )

    def set_default_config(self):
        self.config.update(dict(
            qa_encoding=True,
            ans_encoding=False,
            image_load=False,
            mask_load=False,
            feat_load=True,
            feat_shuffle=False,
            pred_mask_load=False,
            vqa_transforms=None,
            common_transforms=None,
            image_dir='',
            mask_dir='',
            qa_path='',
            image_feat_dir='',
            image_transforms=None,
            batch_size=2,
            num_workers=0,
            training=False,
            select_type=None,
            evaluate=False,
            vocabulary_path='./vocabulary.pkl',
            question_maxlen=QUESTION_MAXLEN,
        ))

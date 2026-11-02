"""Synthetic stand-ins for EarthVLSet, the stage-1 features and the pre-trained tokenizers / language model.

Everything is generated into temporary directories by the fixtures of ``tests/conftest.py``; nothing is downloaded.
The QA files follow the Hugging Face EarthVLSet layout and the HDF5 files the stage-1 feature format
(``feature`` float32 [1536, 32, 32], ``pred_mask`` uint8 [1024, 1024] with values 1..8).
"""
import json
import os
from types import SimpleNamespace

import numpy as np

FEAT_CHANNELS = 1536
FEAT_SIZE = 32
MASK_SIZE = 1024

# (Type, Question, Answer) with real EarthVLSet questions, one or more per question type
QA_TEMPLATES = [
    ('Basic Judging', 'Are there any roads in this scene?', 'Yes'),
    ('Reasoning-based Judging', 'Are there any bridges in this scene?', 'No.'),
    ('Basic Counting', 'What is the area ratio of water?', '10%-20%'),
    ('Reasoning-based Counting', 'How many intersections are in this scene?', '{n}'),
    ('Object Situation Analysis', 'What are the road materials?', 'There are cement roads.'),
    ('Distribution Analysis', 'What are the distributions of the trees?', 'There are some trees along the roads.'),
    ('Direction Analysis', 'What are the directions of the main roads?',
     'Some roads run east-west. Some roads run southwest-northeast.'),
    ('Comprehensive Analysis', 'What are the comprehensive traffic situations in this scene?',
     'This is an important traffic area with {n} intersections.'),
    ('Comprehensive Analysis', 'What are the land use types in this scene?',
     'There are residential and construction areas.'),
]
COUNT_SELECT = ['Basic Counting', 'Reasoning-based Counting', 'Comprehensive Analysis']

# subset -> (QA file of a split, image ids of Train / Test)
_LAYOUT = {
    'EarthVQA': (lambda split: os.path.join('EarthVQA', f'{split}_QA.json'),
                 dict(Train=['1.png', '2.png'], Test=['4191.png', '4192.png'])),
    'EarthVL-GLOBAL': (lambda split: os.path.join('EarthVL-GLOBAL', 'earthvl_global_multiplechoice', f'{split}.json'),
                       dict(Train=['5001.png'], Test=['6001.png'])),
}


def qas_for_image(index):
    n = (2, 0, 7, 3, 5, 1)[index % 6]
    return [dict(Type=t, Question=q, Answer=a.format(n=n)) for t, q, a in QA_TEMPLATES]


def write_feature_file(path, rng, channels=FEAT_CHANNELS, size=FEAT_SIZE, mask_size=MASK_SIZE):
    import h5py
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feature = rng.standard_normal((channels, size, size)).astype(np.float32)
    pred_mask = rng.integers(1, 9, size=(mask_size, mask_size), dtype=np.uint8)
    with h5py.File(path, 'w') as f:
        f.create_dataset('feature', data=feature)
        f.create_dataset('pred_mask', data=pred_mask)
    return feature, pred_mask


def make_vqa_dataset(root, seed=0):
    """EarthVLSet QA files + stage-1 features for Train and Test of both subsets under ``root``."""
    rng = np.random.default_rng(seed)
    data_root = os.path.join(root, 'EarthVLSet')
    feat_root = os.path.join(root, 'features')
    out = SimpleNamespace(data_root=data_root, feat_root=feat_root, qa={}, feat={}, images={}, num_qas={})
    index = 0
    for split in ('Train', 'Test'):
        qa_files, feat_dirs, n = [], [], 0
        for subset, (qa_rel, ids) in _LAYOUT.items():
            qa_dict = {}
            for image_id in ids[split]:
                qa_dict[image_id] = qas_for_image(index)
                index += 1
                write_feature_file(os.path.join(feat_root, subset, split, image_id.replace('.png', '.hdf5')), rng)
                n += len(qa_dict[image_id])
            qa_path = os.path.join(data_root, qa_rel(split))
            os.makedirs(os.path.dirname(qa_path), exist_ok=True)
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(qa_dict, f)
            qa_files.append(qa_path)
            feat_dirs.append(os.path.join(feat_root, subset, split))
            out.images.setdefault(split, []).extend(ids[split])
        out.qa[split] = qa_files
        out.feat[split] = feat_dirs
        out.num_qas[split] = n
    return out


def write_unlabelled_copy(qa_path, out_path):
    """Copy of a QA file with every answer blank (''), like the Hugging Face EarthVQA Test_QA.json, whose labels are
    withheld for the EarthVQA challenge."""
    with open(qa_path, encoding='utf-8') as f:
        qa_dict = json.load(f)
    for qas in qa_dict.values():
        for qa in qas:
            qa['Answer'] = ''
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(qa_dict, f)
    return out_path


def pylist(paths):
    """A list literal for ``key value`` config overrides (forward slashes avoid escaping on Windows)."""
    return '[' + ', '.join(repr(p.replace(os.sep, '/')) for p in paths) + ']'


# ---------------------------------------------------------------------------------------------------------------
# tokenizers / language model
# ---------------------------------------------------------------------------------------------------------------
def _words(texts):
    from tokenizers import pre_tokenizers
    pre = pre_tokenizers.Whitespace()
    seen = {}
    for text in texts:
        for token, _ in pre.pre_tokenize_str(text):
            seen.setdefault(token, None)
    return list(seen)


def make_t5_like_tokenizer(path, texts):
    """Word-level tokenizer with T5's special ids (pad 0, </s> 1, unk 2); stands in for ./pretrain/flan-t5-xl."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ['<pad>', '</s>', '<unk>']
    vocab = {w: i for i, w in enumerate(specials + [w for w in _words(texts) if w not in specials])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>', vocab['</s>'])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='<pad>', eos_token='</s>', unk_token='<unk>',
                                   clean_up_tokenization_spaces=False)
    fast.save_pretrained(path)
    return path


def make_tiny_llava_dir(path, texts, hidden_size=64):
    """A tiny random Llama + word-level tokenizer with ``<image>`` / ``<pad>`` tokens (stands in for
    ./pretrain/vicuna-7b). Returns ``(image_token_index, pad_token_id, hidden_size)``."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = ['<unk>', '<s>', '</s>']
    words = [w for w in _words(texts) if w not in specials + ['<image>', '<pad>']]
    vocab = {w: i for i, w in enumerate(specials + words + ['<image>', '<pad>'])}
    image_id, pad_id = vocab['<image>'], vocab['<pad>']
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single='<s> $A', special_tokens=[('<s>', vocab['<s>'])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                   pad_token='<pad>', additional_special_tokens=['<image>'],
                                   clean_up_tokenization_spaces=False)
    fast.save_pretrained(path)

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=hidden_size, intermediate_size=2 * hidden_size,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048, bos_token_id=vocab['<s>'], eos_token_id=vocab['</s>'],
                      pad_token_id=pad_id, tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return image_id, pad_id, hidden_size


# ---------------------------------------------------------------------------------------------------------------
# segmentation images
# ---------------------------------------------------------------------------------------------------------------
def write_png(path, array):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)

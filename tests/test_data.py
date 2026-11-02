"""Data loaders on synthetic EarthVLSet files (Hugging Face layout + stage-1 HDF5 features)."""
import json
import os
import pickle

import numpy as np
import pytest

import _synthetic

h5py = pytest.importorskip('h5py')
pytest.importorskip('albumentations')
pytest.importorskip('skimage')

DIRECTIONS = {'east-west', 'southwest-northeast', 'north-south', 'southeast-northwest'}


def _dataset(vqa_data, split='Test', **kwargs):
    from data.earthvl_mc import EarthVLDataset
    params = dict(qa_path=vqa_data.qa[split], image_feat_dir=vqa_data.feat[split], image_load=False,
                  mask_load=False, feat_load=True, pred_mask_load=True, encoding=False, vqa_transforms=None)
    params.update(kwargs)
    return EarthVLDataset(**params)


def test_dataset_reads_qas_and_features(vqa_data):
    ds = _dataset(vqa_data)
    assert len(ds) == vqa_data.num_qas['Test']
    image, target = ds[0]
    item = ds.item_list[0]
    with h5py.File(item['feat_path'], 'r') as f:
        feature, pred_mask = f['feature'][()], f['pred_mask'][()]
    assert image.shape == (1536, 32, 32) and image.dtype == np.float32
    np.testing.assert_array_equal(image, feature)
    assert target['seg_mask'].shape == (1024, 1024) and target['seg_mask'].dtype == np.int64
    np.testing.assert_array_equal(target['seg_mask'], pred_mask.astype(np.int64) - 1)  # 1..8 -> 0..7
    assert target['imagen'] in vqa_data.images['Test']
    assert target['question'].endswith('?')
    assert target['answer'].endswith('.')  # 'Yes' -> 'Yes.', '10%-20%' -> '10%-20%.'
    answers = {ds[i][1]['answer'] for i in range(len(ds))}
    assert {'Yes.', 'No.', '10%-20%.'} <= answers


def test_select_type(vqa_data):
    ds = _dataset(vqa_data, select_type=_synthetic.COUNT_SELECT)
    types = {item['questype'] for item in ds.item_list}
    assert types == set(_synthetic.COUNT_SELECT)
    per_image = sum(t in _synthetic.COUNT_SELECT for t, _, _ in _synthetic.QA_TEMPLATES)
    assert len(ds) == per_image * len(vqa_data.images['Test'])


def test_qas_without_answers(vqa_data, unlabelled_test_qa):
    """The EarthVQA Test labels are withheld: for evaluation such QAs keep the answer '', training rejects them."""
    qa_path, unlabelled = unlabelled_test_qa
    ds = _dataset(vqa_data, qa_path=qa_path, training=False)
    assert len(ds) == vqa_data.num_qas['Test']
    for item in ds.item_list:
        if os.path.basename(item['image_path']) in unlabelled:
            assert item['ans'] == ''
        else:
            assert item['ans'].endswith('.')
    assert '' not in ds.ans_voc
    idx = next(i for i, item in enumerate(ds.item_list) if not item['ans'])
    assert ds[idx][1]['answer'] == ''
    ds.encoding = True  # answers as ANSWER_VOC indices: -1 without an answer
    assert ds[idx][1]['answer'] == -1

    with pytest.raises(ValueError, match='Test_QA.json'):
        _dataset(vqa_data, qa_path=qa_path, training=True)


def test_exclude_images(vqa_data, capsys):
    capsys.readouterr()
    ds = _dataset(vqa_data, split='Train', select_type=_synthetic.COUNT_SELECT,
                  exclude_images=['5001.png', '9118.png'])
    images = {os.path.basename(item['image_path']) for item in ds.item_list}
    assert images == set(vqa_data.images['Train']) - {'5001.png'}
    per_image = sum(t in _synthetic.COUNT_SELECT for t, _, _ in _synthetic.QA_TEMPLATES)
    assert len(ds) == per_image * (len(vqa_data.images['Train']) - 1)
    assert ds.excluded_images == {'5001.png'} and ds.num_excluded_qas == per_image
    out = capsys.readouterr().out
    assert (f'Excluded 1 images ({per_image} QAs) listed in exclude_images; 1 listed images are not in the QA files'
            in out)


def test_dataset_prints_on_rank_0_only(vqa_data, unlabelled_test_qa, capsys, monkeypatch):
    import torch.distributed as dist
    assert not (dist.is_available() and dist.is_initialized())  # ever's is_main_process() then reads RANK
    qa_path, unlabelled = unlabelled_test_qa
    num_unlabelled = len(unlabelled) * len(_synthetic.QA_TEMPLATES)
    capsys.readouterr()
    _dataset(vqa_data, qa_path=qa_path, training=False, exclude_images=['6001.png'])
    out = capsys.readouterr().out
    assert f'The {num_unlabelled} QAs are loaded!' in out
    assert f'{num_unlabelled} QAs have no answer' in out and 'Excluded 1 images' in out
    monkeypatch.setenv('RANK', '1')
    _dataset(vqa_data, qa_path=qa_path, training=False, exclude_images=['6001.png'])
    assert capsys.readouterr().out == ''


def test_open_ended_dataset(tmp_path, capsys, monkeypatch):
    from data.earthvl_open import EarthVLOpenQADataset
    qa_path = tmp_path / 'Test.json'
    qa_path.write_text(json.dumps({'6001.png': {'Describe and give me some advice about the greening renovation.':
                                                ['There are some trees.', 'Plant trees along the roads.']}}),
                       encoding='utf-8')
    vocabulary_path = tmp_path / 'vocabulary.pkl'  # a cached vocabulary: spaCy is not needed without encoding
    with open(vocabulary_path, 'wb') as f:
        pickle.dump({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}, f)
    params = dict(qa_path=str(qa_path), qa_encoding=False, ans_encoding=False, image_load=False, mask_load=False,
                  feat_load=False, vocabulary_path=str(vocabulary_path))
    capsys.readouterr()
    assert len(EarthVLOpenQADataset(training=True, **params)) == 2  # one sample per reference answer
    ds = EarthVLOpenQADataset(training=False, **params)
    assert len(ds) == 1
    assert ds[0][1]['answer'] == 'there are some trees////next sentence////plant trees along the roads'
    assert capsys.readouterr().out.count('QAs are loaded!') == 2
    monkeypatch.setenv('RANK', '1')
    EarthVLOpenQADataset(training=False, **params)
    assert capsys.readouterr().out == ''


def test_training_transforms_rewrite_directions(vqa_data):
    from engine.trainer import load_config
    cfg = load_config('EarthVLNet')
    ds = _dataset(vqa_data, split='Train', vqa_transforms=cfg.data.train.params.vqa_transforms,
                  select_type=['Direction Analysis'])
    assert len(ds) == len(vqa_data.images['Train'])
    for _ in range(10):
        image, target = ds[0]
        assert image.shape == (1536, 32, 32)
        assert target['seg_mask'].shape == (1024, 1024)
        words = {w.strip('.') for w in target['answer'].split() if '-' in w}
        assert len(words) == 2 and words <= DIRECTIONS, target['answer']


def test_loader_batches_without_process_group(vqa_data):
    import torch.distributed as dist
    from engine.trainer import build_dataloader, load_config

    assert not (dist.is_available() and dist.is_initialized())
    cfg = load_config('EarthVLCount', [
        'data.train.params.qa_path', _synthetic.pylist(vqa_data.qa['Train']),
        'data.train.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Train']),
        'data.train.params.batch_size', '2', 'data.train.params.num_workers', '0'])
    loader = build_dataloader(cfg.data.train)
    x, y = next(iter(loader))
    assert tuple(x.shape) == (2, 1536, 32, 32)
    assert tuple(y['seg_mask'].shape) == (2, 1024, 1024)
    assert len(y['question']) == 2 and all(isinstance(q, str) for q in y['question'])
    assert set(y['questype']) <= set(_synthetic.COUNT_SELECT)
    # the engine re-seeds the sampler with the global step before every batch
    sampler = loader.sampler
    sampler.set_step(3)
    first = list(sampler)
    sampler.set_step(3)
    assert list(sampler) == first
    sampler.set_step(4)
    assert sorted(sampler) == sorted(first)


def test_loaders_pass_training_and_exclude_images(vqa_data, unlabelled_test_qa):
    from engine.trainer import build_dataloader, load_config
    qa_path, unlabelled = unlabelled_test_qa
    cfg = load_config('EarthVLCount', [
        'data.test.params.qa_path', _synthetic.pylist(qa_path),
        'data.test.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Test']),
        'data.test.params.batch_size', '2', 'data.test.params.num_workers', '0',
        'data.train.params.qa_path', _synthetic.pylist(qa_path),
        'data.train.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Test'])])
    # test loader: QAs without answers are kept, with the answer ''
    loader = build_dataloader(cfg.data.test)
    assert not loader.dataset.training and not loader.dataset.exclude_images
    answers = [(image, answer) for _, y in loader for image, answer in zip(y['imagen'], y['answer'])]
    per_image = sum(t in _synthetic.COUNT_SELECT for t, _, _ in _synthetic.QA_TEMPLATES)
    assert len(answers) == per_image * len(vqa_data.images['Test'])
    assert all((answer == '') == (image in unlabelled) for image, answer in answers)
    # train loader: every QA needs an answer
    with pytest.raises(ValueError, match='Test_QA.json'):
        build_dataloader(cfg.data.train)

    cfg = load_config('EarthVLCount', [
        'data.train.params.qa_path', _synthetic.pylist(vqa_data.qa['Train']),
        'data.train.params.image_feat_dir', _synthetic.pylist(vqa_data.feat['Train']),
        'data.train.params.exclude_images', "['5001.png']", 'data.train.params.num_workers', '0'])
    dataset = build_dataloader(cfg.data.train).dataset
    assert dataset.training and dataset.exclude_images == {'5001.png'}
    assert {os.path.basename(item['image_path']) for item in dataset.item_list} == {'1.png', '2.png'}


def test_seg_dataset_rgba_and_label_remap(tmp_path):
    from data.earthvl_seg import EarthVLSegDataset, reclassify
    rng = np.random.default_rng(0)
    root = tmp_path / 'EarthVLSet'
    rgba = rng.integers(0, 256, size=(64, 64, 4), dtype=np.uint8)
    mask = np.tile(np.arange(10, dtype=np.uint8), (64, 7))[:, :64]  # values 0..9
    _synthetic.write_png(str(root / 'EarthVL-GLOBAL' / 'Val' / 'images_png' / '9100.png'), rgba)
    _synthetic.write_png(str(root / 'EarthVL-GLOBAL' / 'Val' / 'masks_png' / '9100.png'), mask)
    ds = EarthVLSegDataset(data_root=str(root), subsets=['EarthVL-GLOBAL'], splits=['Val'])
    image, target = ds[0]
    np.testing.assert_array_equal(image, rgba[:, :, :3])
    assert target['imagen'] == '9100.png'
    # 0 -> ignore (-1), 1..8 -> 0..7, LoveDA 9 (pool) -> water (3)
    np.testing.assert_array_equal(reclassify(np.arange(10)), [-1, 0, 1, 2, 3, 4, 5, 6, 7, 3])
    np.testing.assert_array_equal(target['mask'], reclassify(mask))

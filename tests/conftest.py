"""CPU smoke tests of the EarthVL release: no dataset, GPU, DeepSpeed or 7B weights needed.

Run from anywhere with ``python -m pytest tests -q``. The tests switch the working directory to the repository root,
because ``ever.registry.register_all()`` and the config names (``--config_path EarthVLCount``) are resolved from it.
Optional dependencies (transformers, peft, h5py, ...) are skipped cleanly when they are missing. GPUs are hidden
(the models place their inputs on ``cuda`` whenever it is available).
"""
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # before torch is imported

import pytest  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
FAKE_DEEPSPEED_DIR = os.path.join(TESTS_DIR, 'fake_deepspeed')

os.chdir(REPO_ROOT)
for _p in (TESTS_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('USE_LIBUV', '0')  # gloo rendezvous on PyTorch builds without libuv (Windows)

import _synthetic  # noqa: E402


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope='session')
def vqa_data(tmp_path_factory):
    """Synthetic EarthVLSet QA files + stage-1 HDF5 features (Train and Test of EarthVQA and EarthVL-GLOBAL)."""
    pytest.importorskip('h5py')
    return _synthetic.make_vqa_dataset(str(tmp_path_factory.mktemp('earthvlset')))


@pytest.fixture(scope='session')
def unlabelled_test_qa(vqa_data, tmp_path_factory):
    """Test QA files (same features as ``vqa_data``) as on Hugging Face: EarthVQA without answers (labels withheld
    for the EarthVQA challenge), EarthVL-GLOBAL with answers. Returns ``(qa_paths, images_without_answers)``."""
    qa_path = os.path.join(str(tmp_path_factory.mktemp('unlabelled')), 'EarthVQA', 'Test_QA.json')
    _synthetic.write_unlabelled_copy(vqa_data.qa['Test'][0], qa_path)
    with open(qa_path, encoding='utf-8') as f:
        images = set(json.load(f))
    return [qa_path, vqa_data.qa['Test'][1]], images


@pytest.fixture(scope='session')
def t5_tokenizer_dir(tmp_path_factory):
    """Word-level stand-in for the flan-t5-xl tokenizer of the counting branch."""
    pytest.importorskip('transformers')
    pytest.importorskip('tokenizers')
    from data.vocab import QUESTIONS
    return _synthetic.make_t5_like_tokenizer(str(tmp_path_factory.mktemp('flan-t5-tokenizer')), QUESTIONS)


def _purge_deepspeed_modules():
    return {name: sys.modules.pop(name) for name in list(sys.modules)
            if name == 'deepspeed' or name.startswith('deepspeed.')}


@pytest.fixture
def clean_distributed():
    """Restore the environment and destroy any process group created by the engine during the test."""
    import torch.distributed as dist
    env = dict(os.environ)
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    os.environ.clear()
    os.environ.update(env)


@pytest.fixture
def fake_deepspeed(clean_distributed):
    """Make ``import deepspeed`` load tests/fake_deepspeed (also when a real DeepSpeed is installed)."""
    saved = _purge_deepspeed_modules()
    sys.path.insert(0, FAKE_DEEPSPEED_DIR)
    import deepspeed
    assert deepspeed.__version__.endswith('+fake'), deepspeed.__file__
    deepspeed.CALLS.clear()
    try:
        yield deepspeed
    finally:
        sys.path.remove(FAKE_DEEPSPEED_DIR)
        _purge_deepspeed_modules()
        sys.modules.update(saved)

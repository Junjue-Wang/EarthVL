"""The code base imports, registers its modules and loads its configs with public packages only."""
import json
import os
import py_compile
import subprocess
import sys

import pytest

HEAVY_OPTIONAL = ('transformers', 'peft', 'bitsandbytes', 'deepspeed', 'spacy', 'nltk', 'timm', 'flash_attn')


def test_all_python_files_compile(repo_root, tmp_path):
    skip = {'EarthVLSet', 'features', 'pretrain', 'log', '__pycache__'}  # data / outputs; dot-folders below
    files = []
    for root, dirs, names in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.')]
        files += [os.path.join(root, n) for n in names if n.endswith('.py')]
    assert len(files) > 20
    for i, path in enumerate(files):
        py_compile.compile(path, cfile=str(tmp_path / f'{i}.pyc'), doraise=True)


def test_register_all_from_repo_root(repo_root):
    """``import ever as er; er.registry.register_all()`` registers every model / loader and stays light."""
    code = (
        'import json, sys\n'
        'import ever as er\n'
        'er.registry.register_all()\n'
        'from ever.core import registry\n'
        'print("RESULT" + json.dumps(dict(models=sorted(registry.MODEL), loaders=sorted(registry.DATALOADER),\n'
        f'    heavy=[m for m in {HEAVY_OPTIONAL!r} if m in sys.modules])))\n'
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.run([sys.executable, '-c', code], cwd=repo_root, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT')][-1]
    result = json.loads(line[len('RESULT'):])
    for name in ('EarthVLNet', 'EarthVLCount', 'SemanticFPN'):
        assert name in result['models']
    for name in ('EarthVLLoader', 'EarthVLSegLoader', 'EarthVLOpenQALoader'):
        assert name in result['loaders']
    assert not [name for name in result['loaders'] if 'EarthGPT' in name]  # no legacy aliases
    # optional / heavy packages are imported lazily, inside the models or the engine
    assert result['heavy'] == []


def test_train_generation_imports_with_ever_beta():
    """train_generation.py imports with the public ever-beta and takes the RMSE table rows from data.vocab."""
    pytest.importorskip('albumentations')
    import train_generation
    from data import vocab
    assert train_generation.COUNT_TYPES is vocab.COUNT_TYPES
    assert train_generation.COUNT_QUESTIONS is vocab.COUNT_QUESTIONS
    assert callable(train_generation.evaluate_cls_fn)


@pytest.mark.parametrize('name', ['EarthVLNet', 'EarthVLCount'])
def test_vqa_data_configs(name):
    """Rotterdam (HF EarthVL-GLOBAL Train ids 9118-9485, Val in the paper) is excluded from VQA training only; the
    test loaders keep QAs without answers (withheld EarthVQA Test labels)."""
    pytest.importorskip('albumentations')
    from engine.trainer import load_config
    cfg = load_config(name)
    train, test = cfg.data.train.params, cfg.data.test.params
    assert list(train.exclude_images) == [f'{i}.png' for i in range(9118, 9486)]
    assert len(set(train.exclude_images)) == 368
    assert not test.get('exclude_images')
    assert train.training is True and test.training is False
    assert train.select_type == test.select_type
    if name == 'EarthVLNet':  # data config of the released LLM-branch weights: no counting questions
        assert len(train.select_type) == 6
        assert not {'Basic Counting', 'Reasoning-based Counting'} & set(train.select_type)


@pytest.mark.parametrize('name, model_type, loader_type', [
    ('EarthVLNet', 'EarthVLNet', 'EarthVLLoader'),
    ('EarthVLCount', 'EarthVLCount', 'EarthVLLoader'),
    ('seg.sfpn_convnext_large', 'SemanticFPN', 'EarthVLSegLoader'),
])
def test_configs_load_and_sanitize(name, model_type, loader_type):
    pytest.importorskip('albumentations')
    from ever.core import registry
    from engine.trainer import _ensure_registered, load_config, sanitize_deepspeed_config

    cfg = load_config(name)
    assert cfg.model['type'] == model_type
    _ensure_registered(model_type, [registry.MODEL], 'model')
    for split in ('train', 'test'):
        assert cfg.data[split]['type'] == loader_type
        _ensure_registered(loader_type, [registry.DATALOADER], 'data loader')
    assert int(cfg.train['num_iters']) > 0

    micro = cfg.data.train.params['batch_size']
    ds = sanitize_deepspeed_config(cfg.deepspeed, micro_batch=micro, world_size=1)
    assert ds['train_micro_batch_size_per_gpu'] == micro
    assert 'auto' not in json.dumps(ds)
    for key in ('reduce_bucket_size', 'sub_group_size', 'allgather_bucket_size'):
        if key in ds.get('zero_optimization', {}):
            assert isinstance(ds['zero_optimization'][key], int)
    # an LR schedule is defined either by DeepSpeed or by the engine (learning_rate), not both
    assert ('scheduler' in ds) != bool(cfg.get('learning_rate', None))


def test_script_overrides_exist_in_configs(repo_root):
    """Every ``key value`` override in scripts/*.sh names an existing config entry (typos would be silently added)."""
    import re
    pytest.importorskip('albumentations')
    from engine.trainer import load_config
    filled_by_engine = {'deepspeed.train_micro_batch_size_per_gpu'}  # set from data.train.params.batch_size
    checked = 0
    for name in sorted(os.listdir(os.path.join(repo_root, 'scripts'))):
        text = open(os.path.join(repo_root, 'scripts', name), encoding='utf-8').read()
        m = re.search(r"config_path=['\"]?([\w.]+)", text)
        if not m or not re.search(r'(train_generation|train_seg)\.py', text):
            continue
        cfg = load_config(m.group(1))
        for key in re.findall(r'^\s+([a-z_]+(?:\.\w+)+)\s+\S+\s*\\?\s*$', text, flags=re.M):
            if key in filled_by_engine:
                continue
            node = cfg
            for part in key.split('.'):
                assert isinstance(node, dict) and part in node, f'scripts/{name}: unknown config key {key}'
                node = node[part]
            checked += 1
    assert checked >= 20


def test_config_overrides():
    pytest.importorskip('albumentations')
    from engine.trainer import load_config, parse_args

    args = parse_args(['--config_path', 'EarthVLCount', '--model_dir', 'x', 'train.num_iters', '7',
                       'data.train.params.qa_path', "['a.json', 'b.json']", 'model.params.count_cfg.depth', '1'])
    cfg = load_config(args.config_path, args.opts)
    assert cfg.train.num_iters == 7
    assert cfg.data.train.params.qa_path == ['a.json', 'b.json']
    assert cfg.model.params.count_cfg.depth == 1
    with pytest.raises(SystemExit):
        parse_args(['--config_path', 'EarthVLCount', '--model_dir', 'x', '--eval_only'])  # needs --ckpt
    with pytest.raises(SystemExit):
        parse_args(['--config_path', 'EarthVLCount', '--model_dir', 'x', 'train.num_iters'])  # odd overrides

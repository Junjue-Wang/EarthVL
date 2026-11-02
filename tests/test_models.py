"""Forward / backward of the three models on random inputs (CPU), and their checkpoint-compatible parameter names."""
import copy
import hashlib
import os

import pytest
import torch

import _synthetic

# Sorted "name:shape" signature of the released counting-branch weights (earthvl_count_40000.pth, the authors'
# 40k-step checkpoint without optimizer state and 'module.' prefix): 53 tensors, 21,755,673 parameters.
COUNT_CKPT_SIGNATURE = 'f860f0cc739ff5eab6ec57e666afd3c0881dc29e23f6675e96fa9ef71b36f3e6'
COUNT_CKPT_NUM_PARAMS = 21_755_673
# Same for Semantic-FPN + ConvNeXt-L (configs/seg/sfpn_convnext_large.py), whose 448 tensors have the names of the
# authors' segmentation code.
SEG_SIGNATURE = '26b8c4d27b58d8acac54985eb1485bea91253fa39f52b72098d9b75b670d3c4f'
SEG_NUM_PARAMS = 200_960_584


def signature(state_dict):
    text = '\n'.join(f'{k}:{tuple(v.shape)}' for k, v in sorted(state_dict.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def random_batch(questions, answers, channels=1536, mask_size=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(questions), channels, 32, 32, generator=g)
    seg_mask = torch.randint(0, 8, (len(questions), mask_size, mask_size), generator=g)
    return x, dict(question=list(questions), answer=list(answers), seg_mask=seg_mask)


# ---------------------------------------------------------------------------------------------------------------
# counting branch
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def count_model(t5_tokenizer_dir):
    from module.EarthVLNet import EarthVLCount
    torch.manual_seed(0)
    return EarthVLCount(dict(pretrained_model_dir=t5_tokenizer_dir))  # defaults = released checkpoint


def test_count_model_default_tokenizer_is_flan_t5_from_the_hub(monkeypatch, t5_tokenizer_dir):
    from transformers import AutoTokenizer
    from module.EarthVLNet import EarthVLCount
    requested = []
    load = AutoTokenizer.from_pretrained
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
                        lambda path, *args, **kwargs: requested.append(path) or load(t5_tokenizer_dir))
    EarthVLCount(dict())
    assert requested == ['google/flan-t5-xl']  # only the tokenizer files are downloaded


def test_count_model_matches_released_checkpoint(count_model):
    state = count_model.state_dict()
    assert len(state) == 53
    assert sum(v.numel() for v in state.values()) == COUNT_CKPT_NUM_PARAMS
    assert signature(state) == COUNT_CKPT_SIGNATURE


@pytest.mark.skipif(not os.environ.get('EARTHVL_COUNT_WEIGHTS'),
                    reason='set EARTHVL_COUNT_WEIGHTS to the released earthvl_count_40000.pth (or model-40000.pth)')
def test_released_count_weights_load_strictly(count_model):
    from engine.trainer import read_state_dict
    state, _ = read_state_dict(os.environ['EARTHVL_COUNT_WEIGHTS'])
    assert signature(state) == COUNT_CKPT_SIGNATURE
    model = copy.deepcopy(count_model)
    model.load_state_dict(state, strict=True)


def test_count_forward_backward(count_model):
    from module.metric import Reverse_Lookup
    questions = ['How many intersections are in this scene?', 'What is the area ratio of water?',
                 'What are the land use types in this scene?']
    answers = ['3.', '10%-20%.', 'There are residential areas.']  # the last one has no number: not trained
    x, y = random_batch(questions, answers)
    count_model.train()
    out = count_model(x, y)
    assert set(out) >= {'numerical_loss', 'scaler', 'cls_l', 'count_l'}
    loss = out['numerical_loss']
    assert loss.dim() == 1 and loss.numel() == 1 and torch.isfinite(loss).all()
    count_model.zero_grad()
    loss.sum().backward()
    grads = [p.grad for p in count_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert count_model.mm_projector.linear_1.weight.grad.abs().sum() > 0
    count_model.eval()
    with torch.no_grad():
        pred = count_model(x, y)
    assert len(pred) == 3 and all(p in Reverse_Lookup for p in pred)


def test_object_guided_attention_formula():
    """OGA: x + x * sigmoid(MLP(avg-pool(x)) + MLP(max-pool(x))), CBAM channel attention with a GELU bottleneck."""
    from module.base_modules import OGA
    torch.manual_seed(0)
    oga = OGA(32, cam_ratio=4)
    x = torch.randn(2, 32, 5, 6)
    fc1, fc2 = oga.pre_cam.fc1.weight[:, :, 0, 0], oga.pre_cam.fc2.weight[:, :, 0, 0]

    def mlp(v):
        return torch.nn.functional.gelu(v @ fc1.T) @ fc2.T

    weights = torch.sigmoid(mlp(x.mean(dim=(2, 3))) + mlp(x.amax(dim=(2, 3))))
    with torch.no_grad():
        torch.testing.assert_close(oga(x), x + x * weights[:, :, None, None])


def test_numerical_loss_statistics_are_finite_for_empty_batches():
    from module.loss import NumericalLoss
    loss_fn = NumericalLoss(uper_numer_idx=0, alpha=1., keep_ratio=0.8, gamma=0.5, min_keep=1)
    out = loss_fn(torch.zeros(0, 24), torch.zeros(0, dtype=torch.long))
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in out.values())
    out = loss_fn(torch.randn(5, 24), torch.tensor([0, 3, 11, 23, 5]))
    assert torch.isfinite(out['numerical_loss']).all() and out['numerical_loss'].item() > 0


# ---------------------------------------------------------------------------------------------------------------
# segmentation (stage 1)
# ---------------------------------------------------------------------------------------------------------------
def test_seg_model_real_config(monkeypatch):
    pytest.importorskip('albumentations')
    from engine.trainer import build_model, load_config
    from module.seg.convnext import SKIP_PRETRAINED_ENV
    monkeypatch.setenv(SKIP_PRETRAINED_ENV, '1')  # no ImageNet download
    torch.manual_seed(0)
    model = build_model(load_config('seg.sfpn_convnext_large')).eval()
    state = model.state_dict()
    assert len(state) == 448
    assert sum(p.numel() for p in model.parameters()) == SEG_NUM_PARAMS
    assert signature(state) == SEG_SIGNATURE
    with torch.no_grad():
        probs, feature = model(torch.randn(1, 3, 64, 64))
    assert tuple(probs.shape) == (1, 8, 64, 64)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(1, 64, 64))
    assert tuple(feature.shape) == (1, 1536, 2, 2) and feature.dtype == torch.float32
    # stage-4 feature is LayerNorm'd over the channels
    assert feature.mean(dim=1).abs().max() < 1e-4
    assert (feature.std(dim=1, unbiased=False) - 1).abs().max() < 1e-2


def test_seg_model_training_step():
    pytest.importorskip('albumentations')
    from engine.trainer import build_model, load_config
    cfg = load_config('seg.sfpn_convnext_large', [
        'model.params.encoder.convnext.arch', "'tiny'", 'model.params.encoder.convnext.init_cfg', 'None',
        'model.params.fpn.in_channels_list', '(96, 192, 384, 768)'])
    model = build_model(cfg).train()
    out = model(torch.randn(2, 3, 64, 64), dict(mask=torch.randint(-1, 8, (2, 64, 64))))
    assert set(out) == {'seg_loss'} and torch.isfinite(out['seg_loss'])
    out['seg_loss'].backward()
    assert model.en.downsample_layers[0][0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------------------------------------------
# LLM branch with a tiny random Llama
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tiny_llava_dir(tmp_path_factory):
    pytest.importorskip('transformers')
    pytest.importorskip('peft')
    pytest.importorskip('tokenizers')
    from module.EarthVLNet import SYSTEM_PROMPT
    texts = [SYSTEM_PROMPT, 'USER: ASSISTANT:'] + [q + ' ' + a for _, q, a in _synthetic.QA_TEMPLATES]
    path = str(tmp_path_factory.mktemp('tiny-vicuna'))
    image_id, pad_id, hidden = _synthetic.make_tiny_llava_dir(path, texts)
    return dict(pretrained_model_dir=path, image_token_index=image_id, pad_token_id=pad_id, hidden_size=hidden)


def test_earthvlnet_tiny_llm(tiny_llava_dir):
    from ever.core.builder import make_model
    from module.EarthVLNet import EarthVLNet  # noqa: F401  (registers the model)
    info = tiny_llava_dir
    params = dict(pretrained_model_dir=info['pretrained_model_dir'], vision_channel=64 + 16, mask_emb_dim=16,
                  attn_implementation='eager', max_new_tokens=3,
                  language_model=dict(pad_token_id=info['pad_token_id'], hidden_size=info['hidden_size'],
                                      image_token_index=info['image_token_index'], ignore_index=-100))
    torch.manual_seed(0)
    model = make_model(dict(type='EarthVLNet', params=params))
    assert model.config.lora.r == 64 and model.config.max_new_tokens == 3  # defaults kept + overrides applied
    assert model.config.load_in_4bit is False  # default: LoRA on the bf16 base, as in the released run
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # LoRA A and B on the 7 linear layers of every decoder layer (not lm_head), OGA (2), MMP (4) and the mask
    # embedding (8): the 462 tensors of the released weights for the 32 layers of Vicuna-7B
    num_layers = model.language_model.config.num_hidden_layers
    assert sum('.lora_' in n for n in trainable) == 2 * 7 * num_layers
    assert len(trainable) == 2 * 7 * num_layers + 2 + 4 + 8
    assert any(n.startswith('mm_projector.') for n in trainable) and any(n.startswith('oga.') for n in trainable)
    assert not any(n.endswith('base_layer.weight') or n.endswith('embed_tokens.weight') for n in trainable)
    assert 'image_newline' in model.state_dict() and 'image_newline' not in trainable  # checkpoint key, frozen
    # gradient checkpointing without reentrant autograd, also for the bf16 base
    decoder = model.language_model.get_base_model().model
    assert decoder.gradient_checkpointing
    assert decoder._gradient_checkpointing_func.keywords == dict(use_reentrant=False)

    # a weight file with the trainable parameters only (like the released one) leaves only frozen weights missing
    state = model.state_dict()
    result = model.load_state_dict({n: state[n] for n in trainable}, strict=False)
    assert not result.unexpected_keys and not set(result.missing_keys) & trainable
    assert 'image_newline' in result.missing_keys

    questions = [q for _, q, _ in _synthetic.QA_TEMPLATES[:2]]
    answers = ['Yes.', 'No. There are no bridges.']
    x, y = random_batch(questions, answers, channels=64, mask_size=128)
    model.train()
    # the model runs in bf16; oneDNN has no bf16 convolution backward on some CPUs (GPUs are not affected)
    with torch.backends.mkldnn.flags(enabled=False):
        out = model(x, y)
        loss = out['generation_loss']
        assert torch.isfinite(loss.float())
        loss.backward()
    for module in (model.mm_projector.linear_1, model.oga.pre_cam.fc1, model.mask_emb[0]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    named = dict(model.named_parameters())
    assert all(named[n].grad is not None for n in trainable)  # through the checkpointed, frozen decoder layers
    # lora_B starts at zero, so in the first step only its gradients (not those of lora_A) are non-zero
    assert any(named[n].grad.abs().sum() > 0 for n in trainable if 'lora_B' in n)
    assert model.language_model.get_input_embeddings().weight.grad is None and model.image_newline.grad is None

    # labels: only the answer tokens and </s> are trained
    tok = model.tokenizer
    prompts = [model.build_prompt(q) for q in questions]
    texts = ['%s %s </s>' % (p, a) for p, a in zip(prompts, answers)]
    inputs = model.language_processer(text=texts, return_tensors='pt', padding='longest')
    labels = model.build_labels(inputs.input_ids, inputs.attention_mask, prompts)
    for i, a in enumerate(answers):
        kept = labels[i][labels[i] != -100].tolist()
        assert kept == tok(' ' + a + ' </s>', add_special_tokens=False)['input_ids']

    padding_side = tok.padding_side
    model.eval()
    with torch.no_grad():
        pred = model(x, y)
    assert isinstance(pred, list) and len(pred) == 2 and all(isinstance(p, str) for p in pred)
    assert tok.padding_side == padding_side  # generation pads on the left and restores the setting

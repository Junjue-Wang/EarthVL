"""tools/prepare_vicuna.py turns the language model of a LLaVA checkpoint into a complete LlamaForCausalLM, lm_head
included, for both state-dict layouts of transformers: ``language_model.model.*`` / ``language_model.lm_head.*`` up
to 4.51 (the installed 4.44.2) and ``model.language_model.*`` / ``lm_head.*`` from 4.52 on (emulated by renaming)."""
import pytest
import torch

pytest.importorskip('transformers')
pytest.importorskip('accelerate')

# transformers >= 4.52 names, as its checkpoint conversion mapping of LlavaForConditionalGeneration
NEW_LAYOUT = (
    ('language_model.model.', 'model.language_model.'),
    ('language_model.lm_head.', 'lm_head.'),
    ('vision_tower.', 'model.vision_tower.'),
    ('multi_modal_projector.', 'model.multi_modal_projector.'),
)
LLAVA15_IMAGE_ID, LLAVA15_PAD_ID = 32000, 32001


def to_new_layout(state_dict):
    out = {}
    for key, value in state_dict.items():
        for old, new in NEW_LAYOUT:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    return out


def tiny_llava(vocab_size=96, seed=0):
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                              image_size=28, patch_size=14, projection_dim=16)
    text = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                       tie_word_embeddings=False)
    config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=vocab_size - 2,
                         pad_token_id=vocab_size - 1)
    torch.manual_seed(seed)
    return LlavaForConditionalGeneration(config).eval()


def assert_same_state(a, b):
    a, b = a.state_dict(), b.state_dict()
    assert sorted(a) == sorted(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key


def test_extract_language_model_keeps_lm_head():
    from transformers import LlamaForCausalLM
    from tools.prepare_vicuna import extract_language_model
    llava = tiny_llava()
    keys = list(llava.state_dict())
    assert 'language_model.lm_head.weight' in keys and 'language_model.model.norm.weight' in keys  # old layout
    vision_keys = [k for k in keys if not k.startswith('language_model.')]
    assert vision_keys

    lm = extract_language_model(llava)
    assert isinstance(lm, LlamaForCausalLM)
    assert len(lm.state_dict()) == len(keys) - len(vision_keys)
    assert torch.equal(lm.lm_head.weight, llava.state_dict()['language_model.lm_head.weight'])
    assert_same_state(lm, llava.language_model)  # transformers < 4.52: language_model is a LlamaForCausalLM
    assert not any(t.is_meta for t in list(lm.parameters()) + list(lm.buffers()))
    ids = torch.randint(0, 94, (2, 9), generator=torch.Generator().manual_seed(0))  # below <image> / <pad>
    with torch.no_grad():
        torch.testing.assert_close(lm(input_ids=ids).logits, llava(input_ids=ids).logits)


def test_both_state_dict_layouts_give_the_same_model():
    from tools.prepare_vicuna import build_language_model, language_model_state_dict
    llava = tiny_llava()
    old_state = llava.state_dict()
    new_state = to_new_layout(old_state)
    assert 'lm_head.weight' in new_state and not any(k.startswith('language_model.') for k in new_state)
    assert sorted(language_model_state_dict(new_state)) == sorted(language_model_state_dict(old_state))

    text_config = llava.config.text_config
    assert_same_state(build_language_model(text_config, new_state), build_language_model(text_config, old_state))


@pytest.mark.parametrize('layout', ['old', 'new'])
def test_missing_lm_head_fails(layout):
    from tools.prepare_vicuna import build_language_model
    llava = tiny_llava()
    state = llava.state_dict() if layout == 'old' else to_new_layout(llava.state_dict())
    state = {k: v for k, v in state.items() if 'lm_head' not in k}  # e.g. a bare decoder saved by transformers 4.52+
    with pytest.raises(RuntimeError, match='lm_head'):
        build_language_model(llava.config.text_config, state)


def save_tiny_llava_with_processor(path, pad_token='<pad>'):
    """A tiny LLaVA checkpoint and processor; the tokenizer has <image> = 32000 and <pad> = 32001 like that of
    llava-1.5 and pads with ``pad_token``."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(['<unk>', '<s>', '</s>'] + [f'w{i}' for i in range(3, LLAVA15_IMAGE_ID)])}
    vocab.update({'<image>': LLAVA15_IMAGE_ID, '<pad>': LLAVA15_PAD_ID})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                        pad_token=pad_token, additional_special_tokens=['<image>'],
                                        clean_up_tokenization_spaces=False)
    image_processor = CLIPImageProcessor(size={'shortest_edge': 28}, crop_size={'height': 28, 'width': 28})
    LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer).save_pretrained(path)
    llava = tiny_llava(vocab_size=32064)
    llava.save_pretrained(path)
    return llava


def test_cli_writes_a_complete_language_model(tmp_path):
    from transformers import AutoProcessor, LlamaForCausalLM
    from tools import prepare_vicuna
    src, dst = tmp_path / 'llava', tmp_path / 'vicuna'
    llava = save_tiny_llava_with_processor(str(src))
    prepare_vicuna.main(['--src', str(src), '--dst', str(dst)])

    lm, info = LlamaForCausalLM.from_pretrained(str(dst), torch_dtype='auto', output_loading_info=True)
    assert all(not v for v in info.values()), info  # no missing (lm_head!), unexpected or mismatched keys
    assert lm.dtype == torch.bfloat16
    reference = llava.to(torch.bfloat16).language_model
    assert_same_state(lm, reference)
    tokenizer = AutoProcessor.from_pretrained(str(dst)).tokenizer
    assert (tokenizer.convert_tokens_to_ids('<image>'), tokenizer.pad_token_id) == (LLAVA15_IMAGE_ID, LLAVA15_PAD_ID)


def test_cli_checks_the_tokenizer_before_loading_weights(tmp_path):
    from tools import prepare_vicuna
    src = tmp_path / 'llava'
    save_tiny_llava_with_processor(str(src), pad_token='<unk>')  # pads like the llava-v1.6-vicuna tokenizers
    for weights in src.glob('*.safetensors'):
        weights.unlink()  # loading the model would fail with OSError instead of the token check
    with pytest.raises(SystemExit, match='unexpected token ids'):
        prepare_vicuna.main(['--src', str(src), '--dst', str(tmp_path / 'vicuna')])
    assert not (tmp_path / 'vicuna').exists()

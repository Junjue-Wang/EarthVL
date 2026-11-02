"""The trainable parameters of EarthVLNet (default config) are exactly the tensors of the released LLM-branch weights,
earthvlnet_llm_lora_40000.safetensors: 448 LoRA matrices, OGA, MMP and the mask embedding, all bf16 (image_newline is
frozen and not stored). The 7B language model is built on the meta device, so no LLM weights are needed; the file is
only read when EARTHVL_LLM_WEIGHTS points to it."""
import hashlib
import json
import os
import struct
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformers')
pytest.importorskip('peft')

# text config of llava-hf/llava-1.5-7b-hf (Vicuna-7B v1.5, a Llama-2-7B)
LLAVA15_TEXT_CONFIG = dict(vocab_size=32064, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                           num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=4096,
                           rms_norm_eps=1e-5, tie_word_embeddings=False)
# sorted "name:shape" signature of the 462 tensors of earthvlnet_llm_lora_40000.safetensors (184,012,544 parameters)
RELEASE_SIGNATURE = 'd7ded82edf5decfb17ea1f68cda2c6ff0b48a253b425d10e4da20c0a9a0af689'
RELEASE_NUM_TENSORS = 462
RELEASE_NUM_PARAMS = 184_012_544


def signature(shapes):
    text = '\n'.join(f'{k}:{tuple(v)}' for k, v in sorted(shapes.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def read_safetensors_header(path):
    """``{name: {'dtype', 'shape', 'data_offsets'}}`` from the header of a safetensors file (8-byte little-endian
    length, then JSON); no tensor is read."""
    with open(path, 'rb') as f:
        (length,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
    header.pop('__metadata__', None)
    return header


@pytest.fixture(scope='module')
def meta_earthvlnet():
    """EarthVLNet with the default config, whose language model is a 7B Llama on the meta device (no weights)."""
    from transformers import AutoProcessor, LlamaConfig, LlamaForCausalLM
    from module.EarthVLNet import EarthVLNet

    def from_pretrained(cls, path, **kwargs):
        assert 'quantization_config' not in kwargs  # default: bf16 base, as in the released run
        with torch.device('meta'):
            model = cls(LlamaConfig(**LLAVA15_TEXT_CONFIG))
        return model.to(kwargs['torch_dtype'])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LlamaForCausalLM, 'from_pretrained', classmethod(from_pretrained))
        mp.setattr(AutoProcessor, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace(tokenizer=None))
        torch.manual_seed(0)
        model = EarthVLNet(dict())
    return model


def trainable_parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def test_trainable_parameters_match_the_release_signature(meta_earthvlnet):
    trainable = trainable_parameters(meta_earthvlnet)
    assert len(trainable) == RELEASE_NUM_TENSORS
    assert sum(p.numel() for p in trainable.values()) == RELEASE_NUM_PARAMS
    assert signature({name: p.shape for name, p in trainable.items()}) == RELEASE_SIGNATURE
    assert all(p.dtype == torch.bfloat16 for p in trainable.values())
    assert sum('.lora_' in name for name in trainable) == 448
    # kept for the keys of the original DeepSpeed checkpoints, but frozen and not in the released file
    assert 'image_newline' in meta_earthvlnet.state_dict() and 'image_newline' not in trainable


@pytest.mark.skipif(not os.environ.get('EARTHVL_LLM_WEIGHTS'),
                    reason='set EARTHVL_LLM_WEIGHTS to the released earthvlnet_llm_lora_40000.safetensors')
def test_trainable_parameters_match_the_released_weights(meta_earthvlnet):
    path = os.environ['EARTHVL_LLM_WEIGHTS']
    header = read_safetensors_header(path)
    trainable = trainable_parameters(meta_earthvlnet)
    assert len(header) == RELEASE_NUM_TENSORS
    assert sorted(header) == sorted(trainable)
    for name, p in trainable.items():
        assert header[name]['shape'] == list(p.shape), name
        assert header[name]['dtype'] == 'BF16' and p.dtype == torch.bfloat16, name
    assert signature({name: info['shape'] for name, info in header.items()}) == RELEASE_SIGNATURE

    # the non-LoRA tensors (about 48 MB) load strictly into the module's own OGA, MMP and mask embedding
    from safetensors import safe_open
    with safe_open(path, framework='pt') as f:
        for prefix in ('oga.', 'mm_projector.', 'mask_emb.'):
            module = getattr(meta_earthvlnet, prefix[:-1])
            state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
            assert all(torch.isfinite(v.float()).all() for v in state.values()), prefix
            module.load_state_dict(state, strict=True)

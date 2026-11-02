"""Build ./pretrain/vicuna-7b for EarthVLNet.

EarthVLNet fine-tunes the Vicuna-7B language model of LLaVA-1.5 with LoRA on a frozen bf16 base and reuses the LLaVA
processor, whose tokenizer already contains the <image> (id 32000) and <pad> (id 32001) tokens expected by
configs/EarthVLNet.py. The base is llava-hf/llava-1.5-7b-hf: the frozen language-model weights stored in the authors'
training checkpoint equal its weights after bf16 rounding (and differ from llava-hf/llava-v1.6-vicuna-7b-hf).
This script saves the language model of a LLaVA checkpoint as a LlamaForCausalLM, lm_head included, together with
the processor.

Usage:
    python tools/prepare_vicuna.py --src llava-hf/llava-1.5-7b-hf --dst ./pretrain/vicuna-7b

The download and the output take about 14 GB of disk each, and loading takes about 14 GB of memory. The
llava-v1.6-vicuna tokenizers pad with <unk> = 0 and fail the token check, which runs before the weights are downloaded.
"""
import argparse

import torch

# LLaVA state-dict prefix -> LlamaForCausalLM prefix. Up to transformers 4.51 ``language_model`` is a LlamaForCausalLM;
# from 4.52 on, the bare decoder is ``model.language_model`` and ``lm_head`` belongs to the top-level model.
KEY_PREFIXES = (
    ('language_model.model.', 'model.'),
    ('language_model.lm_head.', 'lm_head.'),
    ('model.language_model.', 'model.'),
    ('lm_head.', 'lm_head.'),
)


def language_model_state_dict(llava_state_dict):
    """The language-model entries of a LLaVA state dict under LlamaForCausalLM names; the vision tower, the projector
    and any other entry are left out."""
    state_dict = {}
    for key, value in llava_state_dict.items():
        for src, dst in KEY_PREFIXES:
            if key.startswith(src):
                state_dict[dst + key[len(src):]] = value
                break
    return state_dict


def build_language_model(text_config, llava_state_dict):
    """A LlamaForCausalLM (``text_config``) holding the decoder and lm_head weights of ``llava_state_dict``.

    Every parameter must be found (strict loading). The tensors are taken over, not copied.
    """
    from accelerate import init_empty_weights
    from transformers import LlamaForCausalLM

    with init_empty_weights(include_buffers=False):  # parameters on the meta device until they are assigned
        model = LlamaForCausalLM(text_config)
    model.load_state_dict(language_model_state_dict(llava_state_dict), strict=True, assign=True)
    return model


def extract_language_model(llava_model):
    """The language model of a LlavaForConditionalGeneration as a LlamaForCausalLM, for any transformers version."""
    return build_language_model(llava_model.config.text_config, llava_model.state_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--src', default='llava-hf/llava-1.5-7b-hf',
                        help='LLaVA checkpoint on the Hugging Face Hub or a local directory')
    parser.add_argument('--revision', default=None, help='branch, tag or commit of --src on the Hub')
    parser.add_argument('--dst', default='./pretrain/vicuna-7b', help='output directory')
    args = parser.parse_args(argv)

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.src, revision=args.revision)
    tokenizer = processor.tokenizer
    image_id = tokenizer.convert_tokens_to_ids('<image>')
    print(f'<image> token id = {image_id}, pad token id = {tokenizer.pad_token_id} (expected 32000 and 32001)')
    if (image_id, tokenizer.pad_token_id) != (32000, 32001):
        raise SystemExit(f'{args.src}: unexpected token ids; configs/EarthVLNet.py expects the tokenizer of '
                         'llava-hf/llava-1.5-7b-hf')

    llava = LlavaForConditionalGeneration.from_pretrained(args.src, revision=args.revision, torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True)
    language_model = extract_language_model(llava)
    language_model.save_pretrained(args.dst)
    processor.save_pretrained(args.dst)
    print(f'saved the language model (LlamaForCausalLM with lm_head) and the processor of {args.src} to {args.dst}')


if __name__ == '__main__':
    main()

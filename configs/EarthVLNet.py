"""LLM branch of EarthVLNet: LoRA fine-tuning of the Vicuna-7B language model of LLaVA-1.5 with semantic-guided
visual tokens.

As in the released run, the language model is loaded in bf16 and frozen, and LoRA (r=64, alpha=16, dropout 0.05)
adapts every linear layer except lm_head. ``load_in_4bit=True`` (QLoRA, needs bitsandbytes) loads it in 4-bit NF4
instead to fit smaller GPUs; this deviates from the released run.
Run (4 GPUs x micro batch 6 = batch 24 as in the released run, 40k steps): scripts/train_generation.sh; evaluation:
scripts/eval_cls.sh.
The DeepSpeed block is the one of the original experiments (AdamW betas (0.8, 0.999), WarmupCosineLR, bf16, ZeRO-2).
"""
from configs.EarthVL_cls import data as base_data
from configs.EarthVL_cls import test, train


def _with_params(loader, **params):
    return dict(loader, params=dict(loader['params'], **params))


config = dict(
    model=dict(
        type='EarthVLNet',
        params=dict(
            pretrained_model_dir='./pretrain/vicuna-7b',  # from llava-hf/llava-1.5-7b-hf by tools/prepare_vicuna.py
            vision_channel=1536 + 128,
            mask_emb_dim=128,
            load_in_4bit=False,  # True: QLoRA for smaller GPUs, not used for the released weights
            attn_implementation='flash_attention_2',  # as in the original run; 'sdpa' or 'eager' without flash-attn
            lora=dict(r=64, lora_alpha=16, lora_dropout=0.05),
            language_model=dict(
                pad_token_id=32001,
                hidden_size=4096,
                image_token_index=32000,
                ignore_index=-100,
            ),
        )),
    data=dict(
        train=_with_params(base_data['train'], batch_size=6, num_workers=4),  # per GPU; 4 GPUs: batch 24
        test=_with_params(base_data['test'], batch_size=4),
    ),
    train=dict(train, num_iters=40000),
    test=test,
    deepspeed={
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-4,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7,
                "adam_w_mode": True
            }
        },
        "fp16": {
            "enabled": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": True,
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0,
                "warmup_num_steps": 0,
                "total_num_steps": 40000,  # keep equal to train.num_iters
            }
        },
        # train_micro_batch_size_per_gpu follows data.train.params.batch_size (engine/trainer.py)
        "steps_per_print": 50,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8
        }
    }
)

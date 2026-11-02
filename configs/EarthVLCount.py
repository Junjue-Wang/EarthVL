"""Counting branch of EarthVLNet (EarthVLCount), configured like the released counting checkpoint.

The model settings are those of the released checkpoint. Its optimizer state shows fp32 torch AdamW (betas (0.9,
0.999), eps 1e-8, weight decay 0.2) and a final learning rate of 1.54e-12, i.e. lr 1e-3 under a 40k-step cosine
schedule; it was trained with batch 32 on one GPU, saving every 10240 steps (5 epochs).
Run: scripts/train_count.sh; evaluation: scripts/eval_count.sh.
"""
from configs.EarthVL_count import data as base_data
from configs.EarthVL_count import test, train


def _with_params(loader, **params):
    return dict(loader, params=dict(loader['params'], **params))


dim = 384
config = dict(
    model=dict(
        type='EarthVLCount',
        params=dict(
            pretrained_model_dir='google/flan-t5-xl',  # tokenizer only: Hub id (fetched on first use) or a local copy
            vision_channel=128 + 1536,
            text_hidden_size=dim,
            counting_candidate=1,
            mask_emb_dim=128,
            count_cfg=dict(
                dim=dim,
                depth=3,
                heads=4,
                mlp_dim=dim,
                dim_head=dim,
                dropout=0.,
            ),
            loss=dict(
                uper_numer_idx=0,
                alpha=1.,
                keep_ratio=0.8,
                gamma=0.5,
                min_keep=1
            )
        )),
    data=dict(
        train=_with_params(base_data['train'], batch_size=32, num_workers=2),  # one GPU: batch 32
        test=_with_params(base_data['test'], batch_size=16, num_workers=4),
    ),
    train=dict(train, num_iters=40000, save_ckpt_interval_step=10240, eval_interval_step=0, eval_after_train=True),
    test=test,
    # cosine schedule with a linear warm-up ('cosine_warm' of the original code), applied by engine/trainer.py
    learning_rate=dict(
        type='cosine',
        params=dict(
            base_lr=1e-3,
            max_iters=40000,
            eta_min=0.,
            warmup_step=1000,
            warmup_init_lr=1e-8,
        )),
    deepspeed={
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.2,
                "torch_adam": True,  # torch.optim.AdamW, as in the original run
            }
        },
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "zero_optimization": {"stage": 0},
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 0.0,
        "steps_per_print": 50,
    }
)

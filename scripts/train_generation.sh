#!/usr/bin/env bash
# Train the LLM branch of EarthVLNet (configs/EarthVLNet.py) as the released run: LoRA on the frozen bf16 Vicuna-7B of
# LLaVA-1.5 (./pretrain/vicuna-7b from tools/prepare_vicuna.py), 4 GPUs x micro batch 6 = batch 24, 40k steps,
# DeepSpeed ZeRO-2 + bf16. Writes checkpoints and log/EarthVLNet/prediction-40000.json (test-set evaluation).
# Extra `key value` overrides are passed on, e.g. `model.params.attn_implementation sdpa` without flash-attn, or
# `model.params.load_in_4bit True` for QLoRA (4-bit base, needs bitsandbytes) on smaller GPUs, which deviates from
# the released run. Run from the repository root.
set -e

export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='EarthVLNet'
model_dir='./log/EarthVLNet/'
train_micro_batch_size_per_gpu=6
step=40000

deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0,1,2,3 train_generation.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.num_iters ${step} \
    train.eval_after_train True \
    train.eval_interval_step 0 \
    train.save_ckpt_interval_step ${step} \
    data.test.params.batch_size 4 \
    data.train.params.batch_size ${train_micro_batch_size_per_gpu} \
    deepspeed.train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
    deepspeed.scheduler.params.total_num_steps ${step} \
    data.train.params.num_workers 4 \
    train.log_interval_step 50 \
    "$@"

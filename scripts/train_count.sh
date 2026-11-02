#!/usr/bin/env bash
# Train the counting branch of EarthVLNet (configs/EarthVLCount.py) as the released checkpoint: one GPU, batch 32,
# 40k steps, fp32 AdamW (lr 1e-3, cosine), checkpoints every 10240 steps.
# Writes log/EarthVLCount/prediction-40000.json (test-set evaluation). The flan-t5-xl tokenizer is fetched from the
# Hugging Face Hub; for a local copy, add `model.params.pretrained_model_dir ./pretrain/flan-t5-xl` (extra
# `key value` overrides are passed on). Run from the repository root.
set -e

export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='EarthVLCount'
model_dir='./log/EarthVLCount/'
train_micro_batch_size_per_gpu=32
step=40000

deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0 train_generation.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.num_iters ${step} \
    train.eval_after_train True \
    train.eval_interval_step 0 \
    train.save_ckpt_interval_step 10240 \
    data.test.params.batch_size 16 \
    data.test.params.num_workers 4 \
    data.train.params.batch_size ${train_micro_batch_size_per_gpu} \
    deepspeed.train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
    learning_rate.params.max_iters ${step} \
    data.train.params.num_workers 2 \
    train.log_interval_step 50 \
    "$@"

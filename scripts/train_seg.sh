#!/usr/bin/env bash
# Stage 1 of EarthVLNet: train Semantic-FPN (ConvNeXt-L) for land-cover segmentation.
# Paper setting: 2 GPUs x 8 images (global batch 16), 30k iterations, AdamW 1e-4, poly 0.9.
# Data: ./EarthVLSet (Hugging Face layout), trained on EarthVQA/Train + EarthVL-GLOBAL/Train,
# mIoU reported on the Val split. The ImageNet-22k ConvNeXt-L weights are downloaded on first use.
# Checkpoints: ${model_dir}/model-<step>.pth/ (DeepSpeed layout).
# Extra `key value` overrides are passed on, e.g. `model.params.amp.enabled False` to train in fp32.
set -e
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='seg.sfpn_convnext_large'
model_dir='./log/seg/sfpn_convnext_large'
NUM_GPUS=${NUM_GPUS:-2}
# Keep the global batch at 16: 8 per GPU on 2 GPUs, 4 on 4 GPUs, ...
MICRO_BATCH=${MICRO_BATCH:-$((16 / NUM_GPUS))}

deepspeed --num_gpus ${NUM_GPUS} --master_port $((RANDOM + 10000)) train_seg.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    data.train.params.batch_size ${MICRO_BATCH} \
    deepspeed.train_micro_batch_size_per_gpu ${MICRO_BATCH} \
    "$@"

# Evaluate a checkpoint (mIoU on Val):
# python train_seg.py --config_path=${config_path} --model_dir=${model_dir} \
#     --eval_only --ckpt ${model_dir}/model-30000.pth

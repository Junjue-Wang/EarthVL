#!/usr/bin/env bash
# Evaluate an LLM-branch checkpoint (configs/EarthVLNet.py) on the test set and write
# <model_dir>/prediction-<step>.json; merge it with the counting branch via evaluate.py.
#   bash scripts/eval_cls.sh [CKPT] [MODEL_DIR] [key value ...]
# CKPT: a DeepSpeed checkpoint directory (log/EarthVLNet/model-40000.pth), a .pth state dict, or the released weights
# (trainable parameters only: LoRA, OGA, MMP and mask embedding, on top of ./pretrain/vicuna-7b), e.g.
#   bash scripts/eval_cls.sh ./pretrain/EarthVL-pretrained/earthvlnet_llm_lora_40000.safetensors
# The released weights need the stage-1 features of the authors' Semantic-FPN checkpoint (not available yet); they
# are not meant for the features of a re-trained stage 1.
# Extra `key value` overrides are passed on, e.g. `model.params.load_in_4bit True` for a model trained with QLoRA.
# NUM_GPUS=4 bash scripts/eval_cls.sh ...   evaluates on several GPUs. Run from the repository root.
set -e

export PYTHONPATH=$PYTHONPATH:`pwd`
ckpt=${1:-./log/EarthVLNet/model-40000.pth}
model_dir=${2:-./log/EarthVLNet/}
NUM_GPUS=${NUM_GPUS:-1}

if [ "${NUM_GPUS}" -gt 1 ]; then
    launcher="torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((RANDOM + 10000))"
else
    launcher="python"
fi

${launcher} train_generation.py \
    --config_path=EarthVLNet \
    --model_dir=${model_dir} \
    --eval_only \
    --ckpt=${ckpt} \
    data.test.params.batch_size 4 \
    "${@:3}"

#!/usr/bin/env bash
# Evaluate a counting-branch checkpoint (configs/EarthVLCount.py) on the test set and write
# <model_dir>/prediction-<step>.json; merge it with the LLM branch via evaluate.py.
#   bash scripts/eval_count.sh [CKPT] [MODEL_DIR] [key value ...]
# CKPT: a DeepSpeed checkpoint directory (log/EarthVLCount/model-40000.pth), a .pth state dict such as the released
# earthvl_count_40000.pth, or an ever checkpoint ({'model': state_dict, ...}); the step in the file name (model-<step>
# or a trailing _<step>) names the prediction file.
# The released weights need the stage-1 features of the authors' Semantic-FPN checkpoint (not available yet); they
# are not meant for the features of a re-trained stage 1.
# Extra `key value` overrides are passed on, e.g. `model.params.pretrained_model_dir ./pretrain/flan-t5-xl` for a
# local copy of the tokenizer. NUM_GPUS=2 bash scripts/eval_count.sh ...   evaluates on several GPUs.
# Run from the repository root.
set -e

export PYTHONPATH=$PYTHONPATH:`pwd`
ckpt=${1:-./log/EarthVLCount/model-40000.pth}
model_dir=${2:-./log/EarthVLCount/}
NUM_GPUS=${NUM_GPUS:-1}

if [ "${NUM_GPUS}" -gt 1 ]; then
    launcher="torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((RANDOM + 10000))"
else
    launcher="python"
fi

${launcher} train_generation.py \
    --config_path=EarthVLCount \
    --model_dir=${model_dir} \
    --eval_only \
    --ckpt=${ckpt} \
    data.test.params.batch_size 16 \
    data.test.params.num_workers 4 \
    "${@:3}"

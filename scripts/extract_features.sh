#!/usr/bin/env bash
# Stage 1 -> 2: write the visual features and pseudo masks used by EarthVLNet,
#   ./features/sfpn_convnext_large/<EarthVQA|EarthVL-GLOBAL>/<Train|Test>/<image_id>.hdf5
# ('feature': float32 [1536, 32, 32], 'pred_mask': uint8 [1024, 1024] in 1..8).
# Disk: about 7.34 MB per image (6 MiB feature + 1 MiB mask), i.e. about 63 GB for
# Train (5,260 images) + Test (3,349 images), the only splits the VQA configs read.
# Val (2,341 images, about 17 GB more) is optional: add it to --splits below.
# Existing files are skipped, so the script can be re-run to resume.
# The released VQA weights were trained on features of the authors' own stage-1 checkpoint. Features of a
# re-trained stage 1 differ, so use them with VQA weights trained on them (train_count.sh, train_generation.sh).
set -e
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

config_path='seg.sfpn_convnext_large'
ckpt_path=${CKPT:-'./log/seg/sfpn_convnext_large/model-30000.pth'}
data_root='./EarthVLSet'
out_dir='./features/sfpn_convnext_large'

python tools/extract_seg_features.py \
    --config ${config_path} \
    --ckpt ${ckpt_path} \
    --data_root ${data_root} \
    --out_dir ${out_dir} \
    --subsets EarthVQA EarthVL-GLOBAL \
    --splits Train Test \
    --batch_size 4

# Several GPUs: run one process per GPU with a different shard, e.g.
# for i in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$i python tools/extract_seg_features.py --config ${config_path} --ckpt ${ckpt_path} \
#         --data_root ${data_root} --out_dir ${out_dir} --num_shards 4 --shard_id $i &
# done; wait

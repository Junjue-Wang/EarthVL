#!/usr/bin/env bash
# Merge the predictions of the LLM branch and the counting branch and print the accuracy / RMSE tables.
# Run from the repository root. Defaults:
#   --cls_pred   ./log/EarthVLNet/prediction-40000.json    (scripts/eval_cls.sh)
#   --count_pred ./log/EarthVLCount/prediction-40000.json  (scripts/eval_count.sh)
# Extra arguments are passed on, e.g. `bash scripts/test.sh --save_merged ./log/merged-40000.json`.

python ./evaluate.py "$@"

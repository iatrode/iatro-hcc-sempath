#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

CONFIG="experiments/00_local_eval/configs/local_sampled_eval.yaml"
CHECKPOINT="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
LOG_DIR="experiments/00_local_eval/logs"

mkdir -p "$LOG_DIR" experiments/00_local_eval/results experiments/00_local_eval/reports

python -m hcc_sempath.training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split val | tee "$LOG_DIR/eval_val.log"

python -m hcc_sempath.training.evaluate \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split exval | tee "$LOG_DIR/eval_exval.log"

python experiments/shared/scripts/summarize_eval.py \
  --inputs \
    val=experiments/00_local_eval/results/eval_val.json \
    exval=experiments/00_local_eval/results/eval_exval.json \
  --output experiments/00_local_eval/reports/local_eval_summary.md

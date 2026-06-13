#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

CONFIG="experiments/01_checkpoint_comparison/configs/local_sampled_eval.yaml"
RESULT_DIR="experiments/01_checkpoint_comparison/results"
LOG_DIR="experiments/01_checkpoint_comparison/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR" experiments/01_checkpoint_comparison/reports

for label in epoch61 epoch100; do
  if [[ "$label" == "epoch61" ]]; then
    checkpoint="artifacts/models/hcc-sempath-full/checkpoints-61/best_scientific_score.pt"
  else
    checkpoint="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
  fi
  for split in val exval; do
    python -m hcc_sempath.training.evaluate \
      --config "$CONFIG" \
      --checkpoint "$checkpoint" \
      --split "$split" | tee "$LOG_DIR/${label}_${split}.log"
    mv "$RESULT_DIR/eval_${split}.json" "$RESULT_DIR/${label}_${split}.json"
  done
done

python experiments/01_checkpoint_comparison/scripts/summarize_checkpoint_comparison.py \
  --result-dir "$RESULT_DIR" \
  --csv "$RESULT_DIR/checkpoint_metrics.csv" \
  --report experiments/01_checkpoint_comparison/reports/checkpoint_comparison.md

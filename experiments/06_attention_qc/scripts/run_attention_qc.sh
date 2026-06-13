#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p experiments/06_attention_qc/results experiments/06_attention_qc/logs experiments/06_attention_qc/reports

python scripts/tile_attention_review.py \
  --config artifacts/models/hcc-sempath-full/resolved_config.json \
  --checkpoint artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt \
  --manifest configs/local/mac/manifest.yaml \
  --prototype-dir artifacts/prototypes \
  --split exval \
  --output-dir experiments/06_attention_qc/results \
  --device mps \
  --candidates 6 \
  --samples 2 \
  --zhcc-bank-max 64 \
  --zhcc-bank-batch-size 16 \
  | tee experiments/06_attention_qc/logs/attention_qc.log

cp experiments/06_attention_qc/results/tile_attention_scores.csv \
  experiments/06_attention_qc/results/tile_attention_rows.csv

python experiments/06_attention_qc/scripts/summarize_attention_qc.py \
  --result-dir experiments/06_attention_qc/results \
  --report experiments/06_attention_qc/reports/attention_qc_summary.md

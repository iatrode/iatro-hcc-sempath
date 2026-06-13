#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p experiments/02_embedding_export/results experiments/02_embedding_export/logs experiments/02_embedding_export/reports

python experiments/02_embedding_export/scripts/export_embeddings.py \
  --config experiments/02_embedding_export/configs/local_sampled_export.yaml \
  --checkpoint artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt \
  --split val \
  --split exval \
  --output-dir experiments/02_embedding_export/results \
  | tee experiments/02_embedding_export/logs/embedding_export.log

python experiments/02_embedding_export/scripts/summarize_embedding_export.py \
  --result-dir experiments/02_embedding_export/results \
  --report experiments/02_embedding_export/reports/embedding_export_summary.md

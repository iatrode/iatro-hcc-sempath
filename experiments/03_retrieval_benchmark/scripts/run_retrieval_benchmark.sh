#!/usr/bin/env bash
set -euo pipefail

mkdir -p experiments/03_retrieval_benchmark/results experiments/03_retrieval_benchmark/logs experiments/03_retrieval_benchmark/reports

python experiments/03_retrieval_benchmark/scripts/run_retrieval_benchmark.py \
  --embedding-dir experiments/02_embedding_export/results \
  --output-dir experiments/03_retrieval_benchmark/results \
  --split exval \
  --queries 16 \
  --gallery 128 \
  --topk 10 \
  | tee experiments/03_retrieval_benchmark/logs/retrieval_benchmark.log

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p experiments/08_pre_review_gate/results experiments/08_pre_review_gate/logs experiments/08_pre_review_gate/reports

python experiments/08_pre_review_gate/scripts/run_pre_review_gate.py \
  --metadata experiments/07_full_exval_cache/results/manifests/exval_sampled_z_hcc_metadata.csv \
  --config experiments/shared/configs/local_sampled_eval.yaml \
  --split exval \
  --queries 200 \
  --gallery 50000 \
  --topk 10 \
  --output-dir experiments/08_pre_review_gate/results \
  --report experiments/08_pre_review_gate/reports/pre_review_gate.md \
  2>&1 | tee experiments/08_pre_review_gate/logs/pre_review_gate.log

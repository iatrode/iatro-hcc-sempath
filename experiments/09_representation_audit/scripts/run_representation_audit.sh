#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p experiments/09_representation_audit/results experiments/09_representation_audit/logs experiments/09_representation_audit/reports

python experiments/09_representation_audit/scripts/run_representation_audit.py \
  --metadata experiments/07_full_exval_cache/results/manifests/exval_sampled_z_hcc_metadata.csv \
  --retrieval experiments/08_pre_review_gate/results/merged_query_results_for_review.csv \
  --config experiments/shared/configs/local_sampled_eval.yaml \
  --prototype-dir artifacts/prototypes \
  --split exval \
  --output-dir experiments/09_representation_audit/results \
  --report experiments/09_representation_audit/reports/representation_audit.md \
  2>&1 | tee experiments/09_representation_audit/logs/representation_audit.log

#!/usr/bin/env bash
set -euo pipefail

mkdir -p experiments/04_blinded_review_package/results experiments/04_blinded_review_package/logs experiments/04_blinded_review_package/reports

python experiments/04_blinded_review_package/scripts/build_blinded_review_package.py \
  --retrieval experiments/03_retrieval_benchmark/results/merged_query_results_for_review.csv \
  --output-dir experiments/04_blinded_review_package/results \
  | tee experiments/04_blinded_review_package/logs/blinded_review_package.log

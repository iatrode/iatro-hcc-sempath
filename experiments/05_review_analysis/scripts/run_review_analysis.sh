#!/usr/bin/env bash
set -euo pipefail

mkdir -p experiments/05_review_analysis/results experiments/05_review_analysis/logs experiments/05_review_analysis/reports

python experiments/05_review_analysis/scripts/analyze_review_scores.py \
  --review-items experiments/04_blinded_review_package/results/blinded_review_items.csv \
  --key experiments/04_blinded_review_package/results/blinded_review_key.csv \
  --output-dir experiments/05_review_analysis/results \
  | tee experiments/05_review_analysis/logs/review_analysis.log

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

bash experiments/00_local_eval/scripts/run_local_eval.sh
bash experiments/01_checkpoint_comparison/scripts/run_checkpoint_comparison.sh
bash experiments/02_embedding_export/scripts/run_embedding_export.sh
bash experiments/03_retrieval_benchmark/scripts/run_retrieval_benchmark.sh
bash experiments/04_blinded_review_package/scripts/run_blinded_review_package.sh
bash experiments/05_review_analysis/scripts/run_review_analysis.sh
bash experiments/06_attention_qc/scripts/run_attention_qc.sh

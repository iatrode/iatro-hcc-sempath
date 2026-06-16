#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON:-python}"

declare -A CONFIGS=(
  [a0]="experiments/ablation/configs/a0_full_pamtd.yaml"
  [a1]="experiments/ablation/configs/a1_no_prototype.yaml"
  [a2]="experiments/ablation/configs/a2_no_adjudication.yaml"
  [a3]="experiments/ablation/configs/a3_single_teacher.yaml"
  [a4]="experiments/ablation/configs/a4_single_teacher_prototype.yaml"
  [a5]="experiments/ablation/configs/a5_static_prototypes.yaml"
  [a0p]="experiments/ablation/configs/a6_full_filter.yaml"
)

conditions=("$@")
if [ "${#conditions[@]}" -eq 0 ]; then
  conditions=(a0 a1 a2 a3 a4)
fi

for condition in "${conditions[@]}"; do
  config="${CONFIGS[$condition]:-}"
  if [ -z "$config" ]; then
    echo "unknown ablation condition: $condition" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m hcc_sempath.cli.main train --config "$config"
done

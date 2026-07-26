#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
  echo "hcc-camoe conda launcher not found: $CONDA_BIN" >&2
  exit 2
fi
PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n hcc-camoe python)
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

temp_root=""
cleanup() {
  if [ -n "$temp_root" ] && [ -d "$temp_root" ]; then
    rm -rf -- "$temp_root"
  fi
}
trap cleanup EXIT

declare -A CONFIGS=(
  [a0]="experiments/ablation/configs/a0_full_pamtd.yaml"
  [a1]="experiments/ablation/configs/a1_no_prototype.yaml"
  [a2]="experiments/ablation/configs/a2_no_adjudication.yaml"
  [a3]="experiments/ablation/configs/a3_single_teacher.yaml"
  [a4]="experiments/ablation/configs/a4_single_teacher_prototype.yaml"
  [a5]="experiments/ablation/configs/a5_static_global_prototypes.yaml"
  [a6]="experiments/ablation/configs/a6_static_spatial_prototypes.yaml"
  [a7]="experiments/ablation/configs/a7_full_filter_sensitivity.yaml"
  [a8]="experiments/ablation/configs/a8_detached_spatial_backbone.yaml"
)

conditions=("$@")
if [ "${#conditions[@]}" -eq 0 ]; then
  conditions=(a0 a1 a2 a3 a4 a5 a6 a7 a8)
fi
read -r -a seeds <<< "${HCC_SEMPATH_ABLATION_SEEDS:-13 37 71}"
if [ "${#seeds[@]}" -eq 0 ]; then
  echo "HCC_SEMPATH_ABLATION_SEEDS must contain at least one integer seed" >&2
  exit 2
fi
if [ -z "${HCC_SEMPATH_ABLATION_BASE_CONFIG:-}" ]; then
  echo "HCC_SEMPATH_ABLATION_BASE_CONFIG must point to the local full-run config" >&2
  exit 2
fi

for condition in "${conditions[@]}"; do
  config="${CONFIGS[$condition]:-}"
  if [ -z "$config" ]; then
    echo "unknown ablation condition: $condition" >&2
    exit 2
  fi
  if [ -z "$temp_root" ]; then
    temp_root="$(mktemp -d "${TMPDIR:-/tmp}/hcc-sempath-ablation.XXXXXX")"
  fi
  for seed in "${seeds[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
      echo "invalid ablation seed: $seed" >&2
      exit 2
    fi
    run_config="$temp_root/${condition}_seed_${seed}.yaml"
    "${PYTHON_CMD[@]}" experiments/ablation/scripts/resolve_ablation_config.py \
      --base "$HCC_SEMPATH_ABLATION_BASE_CONFIG" \
      --condition "$config" \
      --seed "$seed" \
      --output "$run_config"
    "${PYTHON_CMD[@]}" -m hcc_sempath.cli.main train --config "$run_config"
  done
done

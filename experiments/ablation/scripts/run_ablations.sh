#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${HCC_SEMPATH_PYTHON:-}" ]; then
  if [ ! -x "$HCC_SEMPATH_PYTHON" ]; then
    echo "HCC_SEMPATH_PYTHON is not executable: $HCC_SEMPATH_PYTHON" >&2
    exit 2
  fi
  PYTHON_CMD=("$HCC_SEMPATH_PYTHON")
else
  CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
  if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
    echo "hcc-camoe conda launcher not found: $CONDA_BIN" >&2
    exit 2
  fi
  PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n hcc-camoe python)
fi
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

temp_root=""
cleanup() {
  if [ -n "$temp_root" ] && [ -d "$temp_root" ]; then
    rm -rf -- "$temp_root"
  fi
}
trap cleanup EXIT

declare -A CONFIGS=(
  [a1]="experiments/ablation/configs/a1_no_prototype.yaml"
  [a2]="experiments/ablation/configs/a2_no_adjudication.yaml"
  [a3]="experiments/ablation/configs/a3_single_teacher.yaml"
  [a4]="experiments/ablation/configs/a4_single_teacher_prototype.yaml"
  [a5]="experiments/ablation/configs/a5_static_global_prototypes.yaml"
  [a6]="experiments/ablation/configs/a6_static_spatial_prototypes.yaml"
  [a7]="experiments/ablation/configs/a7_full_filter_sensitivity.yaml"
  [a8]="experiments/ablation/configs/a8_detached_spatial_backbone.yaml"
  [a9]="experiments/ablation/configs/a9_semantic_only_spatial.yaml"
  [a10]="experiments/ablation/configs/a10_local_only_spatial.yaml"
  [a11]="experiments/ablation/configs/a11_no_spatial_context.yaml"
  [a12]="experiments/ablation/configs/a12_dense_brush_target.yaml"
)

conditions=("$@")
if [ "${#conditions[@]}" -eq 0 ]; then
  conditions=(a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12)
fi
if [ -z "${HCC_SEMPATH_ABLATION_BASE_CONFIG:-}" ]; then
  echo "HCC_SEMPATH_ABLATION_BASE_CONFIG must point to the selected Optuna A0 trial config" >&2
  exit 2
fi

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/hcc-sempath-ablation.XXXXXX")"

# Resolve every requested condition before starting the first costly run.
for condition in "${conditions[@]}"; do
  config="${CONFIGS[$condition]:-}"
  if [ -z "$config" ]; then
    echo "unknown ablation condition: $condition" >&2
    exit 2
  fi
  run_config="$temp_root/${condition}.yaml"
  resolver_args=(
    --base "$HCC_SEMPATH_ABLATION_BASE_CONFIG"
    --condition "$config"
    --output "$run_config"
  )
  if [ -n "${HCC_SEMPATH_ABLATION_OUTPUT_ROOT:-}" ]; then
    resolver_args+=(--output-root "$HCC_SEMPATH_ABLATION_OUTPUT_ROOT")
  fi
  "${PYTHON_CMD[@]}" experiments/ablation/scripts/resolve_ablation_config.py \
    "${resolver_args[@]}"
done

for condition in "${conditions[@]}"; do
  run_config="$temp_root/${condition}.yaml"
  "${PYTHON_CMD[@]}" -m hcc_sempath.cli.main train --config "$run_config"
done

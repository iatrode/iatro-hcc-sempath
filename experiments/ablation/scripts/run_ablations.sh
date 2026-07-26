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
  [a5]="experiments/ablation/configs/a5_static_prototypes.yaml"
  [a6]="experiments/ablation/configs/a6_full_filter.yaml"
)

conditions=("$@")
if [ "${#conditions[@]}" -eq 0 ]; then
  conditions=(a0 a1 a2 a3 a4 a5 a6)
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
  run_config="$temp_root/${condition}.yaml"
  "${PYTHON_CMD[@]}" experiments/ablation/scripts/resolve_ablation_config.py \
    --base "$HCC_SEMPATH_ABLATION_BASE_CONFIG" \
    --condition "$config" \
    --output "$run_config"
  "${PYTHON_CMD[@]}" -m hcc_sempath.cli.main train --config "$run_config"
done

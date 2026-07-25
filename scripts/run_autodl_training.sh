#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/run_autodl_training.sh CONFIG_PATH [LOG_PATH]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="$1"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

PYTHON_BIN="${HCC_SEMPATH_PYTHON:-/root/miniconda3/envs/hcc-sempath/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "AutoDL training Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ $# -eq 2 ]]; then
  LOG_PATH="$2"
else
  CONFIG_NAME="$(basename -- "${CONFIG_PATH}")"
  LOG_PATH="/root/autodl-tmp/hcc-sempath-runtime/logs/${CONFIG_NAME%.yaml}.log"
fi
mkdir -p -- "$(dirname -- "${LOG_PATH}")"

export PYTHONPATH="${REPO_DIR}/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "training_start config=${CONFIG_PATH} log=${LOG_PATH}"
echo "attach_to_tmux_for_live_output=true"

cd "${REPO_DIR}"
set +e
"${PYTHON_BIN}" -m hcc_sempath.training.train \
  --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_PATH}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

echo "training_exit status=${TRAIN_STATUS}"
exit "${TRAIN_STATUS}"

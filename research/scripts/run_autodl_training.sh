#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: research/scripts/run_autodl_training.sh CONFIG_PATH [LOG_PATH]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="$1"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

PYTHON_BIN="${HCC_SEMPATH_PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "AutoDL training Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_DIR}/src"
export PYTHONUNBUFFERED=1
positive_thread_count() {
  local value="$1"
  local fallback="$2"
  if [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${fallback}"
  fi
}
export OMP_NUM_THREADS="$(positive_thread_count "${OMP_NUM_THREADS:-}" 2)"
export MKL_NUM_THREADS="$(positive_thread_count "${MKL_NUM_THREADS:-}" 2)"
export OPENBLAS_NUM_THREADS="$(positive_thread_count "${OPENBLAS_NUM_THREADS:-}" 1)"
export NUMEXPR_NUM_THREADS="$(positive_thread_count "${NUMEXPR_NUM_THREADS:-}" 1)"

cd "${REPO_DIR}"
if [[ $# -eq 1 ]]; then
  echo "training_start config=${CONFIG_PATH} log=none"
  exec "${PYTHON_BIN}" -m hcc_sempath.training.train \
    --config "${CONFIG_PATH}"
fi

LOG_PATH="$2"
mkdir -p -- "$(dirname -- "${LOG_PATH}")"
echo "training_start config=${CONFIG_PATH} log=${LOG_PATH}"
set +e
"${PYTHON_BIN}" -m hcc_sempath.training.train \
  --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_PATH}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
echo "training_exit status=${TRAIN_STATUS}"
exit "${TRAIN_STATUS}"

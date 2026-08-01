#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/launch_nvme_training.sh SESSION CONFIG_PATH" >&2
  exit 2
fi

SESSION="$1"
CONFIG_PATH="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${HCC_SEMPATH_PYTHON:-/root/miniconda3/bin/python}"
VERIFIED_ASSET_RECEIPT="${HCC_SEMPATH_VERIFIED_ASSET_RECEIPT:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ -z "${VERIFIED_ASSET_RECEIPT}" || ! -f "${VERIFIED_ASSET_RECEIPT}" ]]; then
  echo "verified asset receipt is required: ${VERIFIED_ASSET_RECEIPT:-unset}" >&2
  exit 2
fi
if grep -qE '/(root/)?autodl-fs' "${CONFIG_PATH}"; then
  echo "refusing network-volume config: ${CONFIG_PATH}" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "no NVIDIA GPU is visible" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'; then
  echo "PyTorch cannot use CUDA" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

OUTPUT_DIR="$(
  "${PYTHON_BIN}" -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["runtime"]["output_dir"])' \
    "${CONFIG_PATH}"
)"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/last.pt"
TRAIN_ARGS=(-m hcc_sempath.cli.main train --config "${CONFIG_PATH}")
if [[ -f "${CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume "${CHECKPOINT}")
fi

printf -v COMMAND '%q ' env \
  "PYTHONPATH=${REPO_DIR}/src" \
  "PYTHONUNBUFFERED=1" \
  "HCC_SEMPATH_SOURCE_COMMIT=${HCC_SEMPATH_SOURCE_COMMIT:?set HCC_SEMPATH_SOURCE_COMMIT}" \
  "HCC_SEMPATH_VERIFIED_ASSET_RECEIPT=${VERIFIED_ASSET_RECEIPT}" \
  "HCC_SEMPATH_ASSET_VERIFY_WORKERS=${HCC_SEMPATH_ASSET_VERIFY_WORKERS:-16}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}" \
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}" \
  "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
  "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}" \
  "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

tmux new-session -d -s "${SESSION}" -c "${REPO_DIR}" "exec ${COMMAND}"
echo "started tmux=${SESSION} config=${CONFIG_PATH} output=${OUTPUT_DIR}"
echo "attach: tmux attach -t ${SESSION}"

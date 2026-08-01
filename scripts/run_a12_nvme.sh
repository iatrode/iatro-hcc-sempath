#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${HCC_SEMPATH_DATA_ROOT:-/root/data}"
CONFIG_ROOT="${HCC_SEMPATH_CONFIG_ROOT:-${DATA_ROOT}/configs}"
SOURCE_COMMIT="${HCC_SEMPATH_SOURCE_COMMIT:?set HCC_SEMPATH_SOURCE_COMMIT}"
PYTHON_BIN="${HCC_SEMPATH_PYTHON:-/root/miniconda3/bin/python}"

cd "${REPO_DIR}"
PYTHONPATH="${REPO_DIR}/src" "${PYTHON_BIN}" -m scripts.prepare_nvme_run \
  --mode a12 \
  --best-config "${CONFIG_ROOT}/a0_best_config.yaml" \
  --manifest-template "${CONFIG_ROOT}/manifest.template.yaml" \
  --manifest-output "${CONFIG_ROOT}/manifest.local_nvme.yaml" \
  --data-root "${DATA_ROOT}" \
  --output-root "${DATA_ROOT}/outputs" \
  --output-config "${CONFIG_ROOT}/a12.local_nvme.yaml" \
  --source-commit "${SOURCE_COMMIT}"

exec "${SCRIPT_DIR}/launch_nvme_training.sh" \
  sempath-a12 "${CONFIG_ROOT}/a12.local_nvme.yaml"

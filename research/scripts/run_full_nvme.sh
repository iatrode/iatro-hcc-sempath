#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${HCC_SEMPATH_DATA_ROOT:-/root/data}"
CONFIG_ROOT="${HCC_SEMPATH_CONFIG_ROOT:-${DATA_ROOT}/configs}"
SOURCE_COMMIT="${HCC_SEMPATH_SOURCE_COMMIT:?set HCC_SEMPATH_SOURCE_COMMIT}"
PYTHON_BIN="${HCC_SEMPATH_PYTHON:-/root/miniconda3/bin/python}"
VERIFIED_ASSET_RECEIPT="${HCC_SEMPATH_VERIFIED_ASSET_RECEIPT:-${CONFIG_ROOT}/verified_asset_receipt.json}"

cd "${REPO_DIR}"
PYTHONPATH="${REPO_DIR}/src" "${PYTHON_BIN}" -m research.scripts.prepare_nvme_run \
  --mode full \
  --best-config "${CONFIG_ROOT}/a0_best_config.yaml" \
  --manifest-template "${CONFIG_ROOT}/manifest.template.yaml" \
  --manifest-output "${CONFIG_ROOT}/manifest.local_nvme.yaml" \
  --data-root "${DATA_ROOT}" \
  --output-root "${DATA_ROOT}/outputs" \
  --output-config "${CONFIG_ROOT}/a0_full.local_nvme.yaml" \
  --verified-asset-receipt "${VERIFIED_ASSET_RECEIPT}" \
  --source-commit "${SOURCE_COMMIT}"

export HCC_SEMPATH_VERIFIED_ASSET_RECEIPT="${VERIFIED_ASSET_RECEIPT}"
exec "${SCRIPT_DIR}/launch_nvme_training.sh" \
  sempath-full "${CONFIG_ROOT}/a0_full.local_nvme.yaml"

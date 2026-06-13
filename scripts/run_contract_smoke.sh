#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
PYTHON_BIN="${PYTHON:-python}"
CLI=("$PYTHON_BIN" -m hcc_sempath.cli.main)

rm -rf outputs/smoke
"$PYTHON_BIN" scripts/make_smoke_data.py
"${CLI[@]}" validate-package --input outputs/smoke/fixture/tiles.iac
"$PYTHON_BIN" scripts/render_tile_package_qc.py --package outputs/smoke/fixture/tiles.iac --output outputs/smoke/tile_package_qc.png --max-tiles 8
"${CLI[@]}" train --config configs/distill_smoke.yaml
"${CLI[@]}" evaluate --config configs/distill_smoke.yaml --checkpoint outputs/smoke/run/checkpoints/best.pt --split val
"${CLI[@]}" benchmark --config configs/distill_smoke.yaml --checkpoint outputs/smoke/run/checkpoints/best.pt --steps 2

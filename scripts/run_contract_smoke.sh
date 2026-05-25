#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python scripts/make_smoke_data.py
hcc-sempath validate-package --input smoke_data/tiles.iac
python scripts/render_tile_package_qc.py --package smoke_data/tiles.iac --output outputs/smoke/tile_package_qc.png --max-tiles 8
hcc-sempath train --config configs/distill_smoke.yaml
hcc-sempath evaluate --config configs/distill_smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt --split val
hcc-sempath benchmark --config configs/distill_smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt --steps 2

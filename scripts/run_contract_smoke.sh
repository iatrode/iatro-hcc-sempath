#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python scripts/make_smoke_data.py
python scripts/validate_tile_package.py --package smoke_data/tiles.iac
python scripts/render_tile_package_qc.py --package smoke_data/tiles.iac --output outputs/smoke/tile_package_qc.png --max-tiles 8
python -m hcc_sempath.train --config configs/distill_smoke.yaml
python -m hcc_sempath.evaluate --config configs/distill_smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt --split val
python -m hcc_sempath.benchmark --config configs/distill_smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt --steps 2

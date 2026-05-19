# Remote teacher-cache workflow

## Goal

Build teacher features once on a high-performance machine, then train the student locally without repeatedly running the teacher.

## Inputs copied to the remote machine

Preferred transfer artifact:

1. `tiles.hccspk`
2. Repository code

The package contains `metadata.json`, `manifest.csv`, and JXL-compressed `224 x 224`
tiles. It is a pre-inference data package, not the teacher feature cache.

Legacy transfer inputs are also supported:

1. `tile_manifest.csv`
2. Tile image directory referenced by the manifest
3. Repository code

## Remote commands

```bash
conda env create -f environment.yml
conda activate 2026-ct-wsi
python scripts/download_teacher.py
PYTHONPATH=src python scripts/build_teacher_cache.py \
  --tile-package data/packages/tiles.hccspk \
  --output-dir data/teacher_cache/h_optimus_1 \
  --model-name hf_hub:bioptimus/H-optimus-1 \
  --batch-size 256 \
  --device cuda
```

## Outputs copied back locally

```text
data/teacher_cache/h_optimus_1/<tile_id>.npy
```

The local training code expects exactly one cached feature per `tile_id` in the tile manifest.

## Local verification before training

```bash
PYTHONPATH=src python -m hcc_sempath.train --config configs/distill_train.example.yaml
```

If a cache file is missing or has the wrong dimensionality, training fails during startup before the first batch is launched.

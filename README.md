# HCC-SemPath

HCC-specific semantic pathology encoder repository.

## Environment

```bash
conda env create -f environment.yml
conda activate 2026-ct-wsi
```

## Repository flow

```text
slide/image
  -> 224 px tiling
  -> tile manifest
  -> teacher feature cache
  -> student training
  -> evaluation
  -> deployment benchmark
```

## Contract smoke test

```bash
cd hcc-sempath
conda activate 2026-ct-wsi
bash scripts/run_contract_smoke.sh
```

The smoke test validates the full local contract:

- tile manifest loading
- teacher cache loading
- anchor loading
- training
- checkpointing
- evaluation output
- throughput benchmark

## Core commands

Tile a rasterized slide/image:

```bash
PYTHONPATH=src python -m hcc_sempath.tiling \
  --image slide.png \
  --output-dir data/tiles \
  --manifest-out data/manifests/tile_manifest.csv \
  --patient-id P001 \
  --slide-id S001 \
  --split train
```

Tile a WSI:

```bash
PYTHONPATH=src python -m hcc_sempath.tiling \
  --wsi slide.svs \
  --output-dir data/tiles \
  --manifest-out data/manifests/tile_manifest.csv \
  --patient-id P001 \
  --slide-id S001 \
  --split train \
  --target-mpp 0.5
```

For a fast real-WSI smoke test, add `--max-tiles 64`. For a clean rerun of the same
`slide_id`, add `--overwrite-slide-dir`.

Build a JXL tile package for remote teacher inference:

```bash
PYTHONPATH=src python scripts/build_tile_package.py \
  --manifest data/manifests/tile_manifest.csv \
  --output data/packages/tiles.hccspk
```

Validate a package:

```bash
PYTHONPATH=src python scripts/validate_tile_package.py \
  --package data/packages/tiles.hccspk
```

Download teacher assets:

```bash
python scripts/download_teacher.py
```

Cache teacher features:

```bash
PYTHONPATH=src python -m hcc_sempath.teachers \
  --manifest data/manifests/tile_manifest.csv \
  --output-dir data/teacher_cache/h_optimus_1 \
  --model-name hf_hub:bioptimus/H-optimus-1
```

For remote/high-performance cache building:

```bash
PYTHONPATH=src python scripts/build_teacher_cache.py \
  --tile-package data/packages/tiles.hccspk \
  --output-dir data/teacher_cache/h_optimus_1 \
  --model-name hf_hub:bioptimus/H-optimus-1 \
  --batch-size 256 \
  --device cuda
```

Build semantic anchors:

```bash
PYTHONPATH=src python -m hcc_sempath.build_anchors \
  --concept-dir data/concept_features \
  --output data/anchors/hcc_semantic_anchors.pt
```

Train:

```bash
PYTHONPATH=src python -m hcc_sempath.train --config configs/distill_train.example.yaml
```

To train directly from an HCCSPK package, set `data.tile_package_path` in the
YAML config. The package manifest becomes the tile contract, while
`teacher_cache_dir/<tile_id>.npy` remains the distillation target.

Resume:

```bash
PYTHONPATH=src python -m hcc_sempath.train \
  --config configs/distill_train.example.yaml \
  --resume outputs/hcc_sempath_v1/checkpoints/last.pt
```

Evaluate:

```bash
PYTHONPATH=src python -m hcc_sempath.evaluate \
  --config configs/distill_train.example.yaml \
  --checkpoint outputs/hcc_sempath_v1/checkpoints/best.pt \
  --split val
```

## Data contracts

`tile_manifest.csv` must contain:

```text
tile_id,patient_id,slide_id,tile_path,x,y,split
```

`tiles.hccspk` is a first-class data contract and can replace loose tile files
during both remote teacher inference and local student training. It must contain:

```text
metadata.json
manifest.csv
tiles/<tile_id>.jxl
```

When `data.tile_package_path` is set, training and evaluation read images from
the package and use package `manifest.csv` as the tile index.

Teacher cache must contain one NumPy file per tile:

```text
<teacher_cache_dir>/<tile_id>.npy
```

Anchor payload must be a PyTorch tensor or a dict containing:

```python
{"anchors": Tensor[K, teacher_dim]}
```

Student-side image normalization is configured in YAML. Teacher-cache extraction uses
the teacher model's own `timm` data configuration.

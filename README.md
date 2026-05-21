# HCC-SemPath

HCC-specific semantic pathology encoder repository.

## Environment

```bash
conda env create -f environment.yml
conda activate hcc-sempath
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
conda activate hcc-sempath
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

Build an image-tile IatroCache package directly from an OpenSlide-readable WSI
such as `.svs` or `.mrxs`:

```bash
hcc-sempath-wsi2iac \
  --wsi slide.svs \
  --output data/packages/slide.tiles.iac \
  --target-mpp 0.5 \
  --tile-size 224 \
  --distance 1.0 \
  --workers 8 \
  --qc-out data/packages/slide.tiles.qc.png
```

Equivalent script form:

```bash
python scripts/build_wsi_package.py \
  --wsi slide.svs \
  --output data/packages/slide.tiles.iac
```

Batch-package a WSI directory with progress:

```bash
python scripts/build_wsi_iac_batch.py \
  --input-root /path/to/wsi-root \
  --output-root /path/to/output-iac-root \
  --target-mpp 0.5 \
  --tile-size 224 \
  --min-tissue-fraction 0.3 \
  --distance 1.0 \
  --workers 8
```

Build a JXL tile package for remote teacher inference:

```bash
PYTHONPATH=src python scripts/build_tile_package.py \
  --manifest data/manifests/tile_manifest.csv \
  --output data/packages/tiles.iac
```

Validate a package:

```bash
PYTHONPATH=src python scripts/validate_tile_package.py \
  --package data/packages/tiles.iac
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
  --tile-package data/packages/tiles.iac \
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

Training and evaluation use IatroCache packages as the data contract. Set
`data.image_tile_package_path` and `data.teacher_feature_package_path` in the
YAML config.

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

`tiles.iac` is the image-tile IatroCache package and the training image contract.
Format specification: `docs/IATROCACHE_FORMAT.md`.

Training and evaluation read images from `data.image_tile_package_path` and use
the package's record table as the tile index.

`teacher_features.iac` is the teacher-output IatroCache package and the
distillation target contract. Loose `.npy` teacher caches are intermediate build
artifacts, not the primary training input.

```text
data.teacher_feature_package_path
```

Anchor payload must be a PyTorch tensor or a dict containing:

```python
{"anchors": Tensor[K, teacher_dim]}
```

Student-side image normalization is configured in YAML. Teacher-cache extraction uses
the teacher model's own `timm` data configuration.

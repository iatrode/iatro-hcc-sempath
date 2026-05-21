# HCC-SemPath

HCC-SemPath is a hepatocellular carcinoma specific pathology representation
repository. The project target is a reusable HCC semantic embedding space, not a
single downstream diagnostic, prognosis, or report-generation model.

HCC-SemPath 是面向肝细胞癌（HCC）专病病理表征的仓库。项目目标是构建可复用的
HCC 语义 embedding 空间，而不是单一下游诊断、预后或报告生成模型。

## Current Direction

The main training design is a shared student encoder with HCC-centered
embeddings. During training, teacher-specific projection heads align the student
space to multiple pathology foundation teachers. At inference time, downstream
users consume the student embedding itself; teacher heads are training-time
adapters.

当前设计采用共享学生编码器生成 HCC 专病 embedding。训练时通过多个 teacher 专属
projection head 对接不同病理基座模型；推理和下游任务使用学生 embedding 本身，teacher
head 只是训练阶段的适配层。

HCC weak supervision is expected to further shape the embedding space around
domain-specific morphology and tissue semantics, such as tumor architecture,
cholangiocytic components, liver lobule context, necrosis, fibrosis, and immune
microenvironment patterns.

HCC 弱监督用于进一步把 embedding 空间塑造成专病语义空间，覆盖肿瘤结构、胆管样成分、
肝小叶背景、坏死、纤维化和免疫微环境等 HCC 相关形态语义。

## Repository Scope

This repository contains:

- data contracts and package readers/writers for offline tile and feature caches;
- student model, loss, training, and evaluation scaffolding;
- public-safe documentation of the HCC-SemPath model plan;
- smoke-test utilities based on synthetic data.

本仓库包含：

- 离线 tile 与 feature cache 的数据合同和读写实现；
- 学生模型、loss、训练和评估脚手架；
- 面向公开发布的 HCC-SemPath 模型规划文档；
- 基于合成数据的 smoke-test 工具。

This repository should not contain private WSIs, production tile packages,
teacher feature packages, checkpoints, patient-identifiable manifests, or local
machine paths.

本仓库不应包含私有 WSI、生产级 tile package、teacher feature package、checkpoint、
可识别患者身份的 manifest 或本机路径。

## Environment

```bash
conda env create -f environment.yml
conda activate hcc-sempath
```

## Data Flow

```text
WSI or raster image
  -> 224 px tiling
  -> image-tile package
  -> teacher feature package
  -> multi-teacher student training
  -> HCC semantic embedding
  -> downstream evaluation
```

IatroCache (`.iac`) is the current implementation-level data contract for tile
and feature packages. It is an engineering format for this repository, not the
scientific contribution itself.

IatroCache（`.iac`）是当前 tile 与 feature package 的工程数据合同。它属于仓库实现层，
不是论文或模型工作的科学贡献本身。

## Smoke Test

The contract smoke test uses synthetic data and validates package loading,
anchor loading, training, checkpointing, evaluation output, and throughput
benchmarking.

```bash
cd hcc-sempath
conda activate hcc-sempath
bash scripts/run_contract_smoke.sh
```

## Core Commands

Tile a rasterized image:

```bash
PYTHONPATH=src python -m hcc_sempath.tiling \
  --image slide.png \
  --output-dir data/tiles \
  --manifest-out data/manifests/tile_manifest.csv \
  --patient-id P001 \
  --slide-id S001 \
  --split train
```

Package an OpenSlide-readable WSI directly into an image-tile package:

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

Batch-package a WSI directory:

```bash
conda run --no-capture-output -n hcc-sempath python scripts/build_wsi_iac_batch.py \
  --input-root /path/to/wsi-root \
  --output-root /path/to/output-iac-root \
  --target-mpp 0.5 \
  --tile-size 224 \
  --min-tissue-fraction 0.3 \
  --distance 1.0 \
  --workers 8
```

Validate a package:

```bash
PYTHONPATH=src python scripts/validate_tile_package.py \
  --package data/packages/tiles.iac
```

Build teacher features:

```bash
PYTHONPATH=src python scripts/build_teacher_cache.py \
  --tile-package data/packages/tiles.iac \
  --output-dir data/teacher_cache/h_optimus_1 \
  --model-name hf_hub:bioptimus/H-optimus-1 \
  --batch-size 256 \
  --device cuda
```

Convert teacher outputs into a feature package:

```bash
PYTHONPATH=src python scripts/build_feature_package.py \
  --tile-package data/packages/tiles.iac \
  --feature-dir data/teacher_cache/h_optimus_1 \
  --output data/packages/h_optimus_1_features.iac \
  --teacher-name h_optimus_1
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

## Documentation

- `docs/model_plan.md`: public bilingual model direction.
- `docs/CURRENT_STATUS.md`: current engineering and training status.
- `docs/TECHNICAL_FRAMEWORK.md`: public technical framework.
- `docs/IATROCACHE_FORMAT.md`: implementation-level package format.
- `docs/remote_cache_workflow.md`: teacher-cache workflow.

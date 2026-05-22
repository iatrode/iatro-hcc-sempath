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

## Source Layout

```text
src/hcc_sempath/
  io/        IatroCache, tile packages, feature packages, manifests, QC
  teacher/   teacher model loading and offline feature-cache construction
  modeling/  student models and semantic anchors
  training/  datasets, losses, metrics, engine, train/evaluate/benchmark CLIs
  cli/       installed command-line entry points
```

## Environment

```bash
conda env create -f environment.yml
conda activate hcc-sempath
python -m pip install --no-deps -e .
hcc-sempath --help
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate hcc-sempath
python -m pip install --no-deps -e .
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

Build an image-tile cache from one WSI:

```bash
hcc-sempath build-tile-cache \
  --input slide.svs \
  --output data/packages/slide.tiles.iac \
  --target-mpp 0.5 \
  --tile-size 224 \
  --distance 1.0 \
  --workers 8 \
  --qc
```

Build image-tile caches from a WSI directory:

```bash
hcc-sempath build-tile-cache \
  --input /path/to/wsi-root \
  --output /path/to/output-iac-root \
  --target-mpp 0.5 \
  --tile-size 224 \
  --min-tissue-fraction 0.3 \
  --distance 1.0 \
  --workers 8
```

`build-tile-cache` is the only public WSI ingestion command. It writes tile
IAC directly; PNG tile directories and standalone CSV tile manifests are kept
out of the training workflow. Directory input scans only the specified
directory's top-level WSI files because MRXS slides use a file plus companion
data directory layout. Tissue filtering excludes both white background and
near-black empty regions (`--black-threshold`, default `8`) so MRXS skipped
regions do not become retained tiles.

Validate a package:

```bash
hcc-sempath validate-package \
  --package data/packages/tiles.iac
```

Build teacher features:

```bash
hcc-sempath build-teacher-cache \
  --tile-package data/packages/tiles.iac \
  --output data/packages/h_optimus_1.features.iac \
  --model h_optimus_1 \
  --batch-size 256 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --device cuda
```

Teacher output packages use the suffix pattern `<teacher-name>.features.iac`.
The input image-tile package naming remains unchanged.
Tile size is read from the input `.iac` header. Directory inputs are allowed;
all discovered image-tile packages must have the same tile dimensions.
The planned supported presets are `h_optimus_1` and `gigapath`. Local model directories
and custom timm / `hf_hub:*` names remain available for controlled experiments,
but the documented path should use the supported presets.

Build semantic anchors:

```bash
hcc-sempath build-anchors \
  --concept-dir data/concept_features \
  --output data/anchors/hcc_semantic_anchors.pt
```

Train:

```bash
hcc-sempath train --config configs/distill_train.example.yaml
```

Resume:

```bash
hcc-sempath train \
  --config configs/distill_train.example.yaml \
  --resume outputs/hcc_sempath_v1/checkpoints/last.pt
```

Evaluate:

```bash
hcc-sempath evaluate \
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

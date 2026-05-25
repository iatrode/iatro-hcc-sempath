# HCC-SemPath

HCC-SemPath is a lightweight vertical pathology representation repository for
hepatocellular carcinoma. The project target is a reusable HCC semantic
embedding space, not a general-purpose pathology foundation model and not a
single downstream diagnostic, prognostic, or report-generation model.

HCC-SemPath 是面向肝细胞癌（HCC）的轻量级垂直病理表征仓库。项目目标是构建可复用的
HCC 语义 embedding 空间，而不是通用病理基础模型，也不是单一下游诊断、预后或报告生成模型。

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

The guiding public research direction is maintained in
[`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md).

公开研究方向以 [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md) 为准。

## Repository Scope

This repository contains:

- data contracts and package readers/writers for offline tile and feature caches;
- student model, loss, training, and evaluation scaffolding;
- public project direction for a lightweight vertical HCC representation model;
- public-safe documentation of the HCC-SemPath model plan;
- smoke-test utilities based on synthetic data.

本仓库包含：

- 离线 tile 与 feature cache 的数据合同和读写实现；
- 学生模型、loss、训练和评估脚手架；
- 轻量级 HCC 垂直表征模型的公开项目方向；
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
  io/        offline cache readers/writers, manifests, QC
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

IatroCache (`.iac`) is the repository-internal engineering cache contract used
for offline tile caches and teacher feature caches. It is an implementation
detail for training data preparation, not the scientific contribution and not a
general-purpose pathology format. The compact format note lives in
[`docs/IATROCACHE_FORMAT.md`](docs/IATROCACHE_FORMAT.md).

IatroCache（`.iac`）是本仓库内部用于离线 tile cache 与 teacher feature cache 的工程合同，
属于训练数据准备实现细节，不是论文或模型工作的科学贡献，也不是通用病理格式。
格式说明见 [`docs/IATROCACHE_FORMAT.md`](docs/IATROCACHE_FORMAT.md)。

## Training Data Baseline

The current local image-tile IAC baseline contains 928 effective tile packages
and 13,964,919 tiles. This count is the reference scale for subsequent
multi-teacher distillation training.

当前本地 image-tile IAC 基准包含 928 个有效 tile package，共 13,964,919 个 tile。
该规模作为后续多 teacher 蒸馏训练的基准。

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

`build-tile-cache` is the WSI ingestion command used to prepare image-tile
caches for training. It writes the internal cache package directly; PNG tile
directories and standalone CSV tile manifests are kept out of the training
workflow. Directory input scans only the specified directory's top-level WSI
files because MRXS slides use a file plus companion data directory layout.
Tissue filtering excludes both white background and near-black empty regions
(`--black-threshold`, default `8`) so MRXS skipped regions do not become
retained tiles.

Validate a package:

```bash
hcc-sempath validate-package \
  --input data/packages/tiles.iac
```

Validate all tile and teacher-feature IAC packages under a directory:

```bash
hcc-sempath validate-package \
  --input data/packages \
  --max-decode 8 \
  --max-crc 0
```

Directory validation recursively scans `*.iac` files, shows a package progress
bar, validates both image-tile packages and teacher-feature packages, and prints
a final ok/failed summary.

Inspect an internal cache package in a local browser:

```bash
hcc-sempath view-iac \
  --package data/packages/tiles.iac
```

Image-tile packages show a spatial coordinate map and a clickable 5x5 tile
window centered on the clicked coordinate.
Teacher feature packages show a coordinate heatmap without decoding feature
payloads.

Build teacher features:

```bash
hcc-sempath build-teacher-cache \
  --input data/packages \
  --output data/features/h_optimus_1 \
  --teacher h_optimus_1 \
  --batch-size 512 \
  --precision bf16 \
  --feature-dtype auto \
  --compile \
  --num-workers 8 \
  --prefetch-factor 2 \
  --continue-on-error \
  --device cuda
```

Teacher output packages use the suffix pattern `<teacher-name>.features.iac`.
The input image-tile package naming remains unchanged.
Tile size is read from the input `.iac` header. Feature payloads are stored as
a package-level losslessly compressed matrix, with tile order defined by the
record table. Directory inputs are allowed; all discovered image-tile packages
must have the same tile dimensions.
For directory inputs, existing valid outputs are skipped unless `--overwrite`
is passed. Each generated or skipped package is header-checked and recorded in
`teacher_cache_progress.csv` plus a JSON summary under the output directory.
Use `--continue-on-error` for large remote batches so one failed package is
recorded without stopping the whole teacher-cache run.
Use `--validate-output` only when a full output IAC matrix validation is needed;
it decompresses each feature matrix and is intentionally off by default for
large directory runs.
Directory runs prefetch the next input package while the current package is
running teacher inference; use `--no-prefetch-packages` only for debugging.
The planned supported presets are `h_optimus_1`, `gigapath`, `uni2_h`, and
`virchow2`. Local model directories and custom timm / `hf_hub:*` names remain
available for controlled experiments, but the documented path should use the
supported presets.
Teacher-cache defaults are tuned for the development workflow: batch size 512,
bf16 CUDA inference, `torch.compile`, and `--feature-dtype auto`, which writes
float16 feature matrices for fp16/bf16 inference to keep IAC caches compact.
Training casts teacher features back to float32 before computing losses.

`uni2_h` and `virchow2` are gated Hugging Face models. Request access with an
institutional account, accept the model terms, and run `hf auth login` in the
feature-cache environment before building caches. Local snapshots can also be
loaded directly with `--teacher weights/teachers/uni2_h` or
`--teacher weights/teachers/virchow2`; the `weights/` tree is git-ignored.

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
- `docs/IATROCACHE_FORMAT.md`: internal cache contract used by data preparation.
- `docs/remote_cache_workflow.md`: teacher-cache workflow.

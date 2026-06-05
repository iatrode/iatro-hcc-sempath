# HCC-SemPath

HCC-SemPath is an HCC-centered vertical pathology foundation representation
repository. The project target is a reusable HCC histomorphology embedding space,
not a general-purpose pathology foundation model and not a single downstream
diagnostic, prognostic, or report-generation model.

HCC-SemPath 是面向肝细胞癌（HCC）的垂直病理 foundation representation 仓库。项目目标
是构建可复用的 HCC 组织形态 embedding 空间，而不是通用病理 foundation model，也不是
单一下游诊断、预后或报告生成模型。

## Current Direction

The main training design is a shared student encoder with HCC-centered
embeddings. During training, teacher-specific projection heads align the student
space to multiple pathology foundation teachers. At inference time, downstream
users consume the normalized student embedding (`embedding_norm`) by default;
teacher heads are training-time adapters.

当前设计采用共享学生编码器生成 HCC 专病 embedding。训练时通过多个 teacher 专属
projection head 对接不同病理基座模型；推理和下游任务默认使用归一化学生 embedding
（`embedding_norm`），teacher head 只是训练阶段的适配层。

HCC prototype-mediated semantic response supervision further shapes the
embedding space around domain-specific morphology and tissue context, including
tumor, background liver, inflammatory/stromal, degenerative, hepatocellular,
necrotic, hemorrhagic, bile-pigment, fibrous, vascular, and ductular/portal
patterns.

HCC prototype-mediated semantic response supervision 用于进一步把 embedding 空间塑造成专病语义空间，覆盖肿瘤、
背景肝、炎症/间质、退变物、肝细胞性实质、坏死、出血、胆色素、纤维间质、血管结构和
胆管/汇管区等 HCC 相关形态语义。

The current method configuration is PAMT-D, Prototype-Adjudicated Multi-Teacher
Distillation. Two-level HCC prototypes compute per-tile, per-teacher reliability
from cross-teacher consensus, expert prototype-label agreement, and agreement
with the current shared `z_hcc` prototype response. The prototype tiles define
the fixed HCC semantic blueprint; they are not used as a hard-label image
classification training set.

当前方法配置为 PAMT-D，即原型裁决的多教师蒸馏。两级 HCC prototype 通过跨 teacher
共识、专家 prototype label 一致性，以及与当前共享 `z_hcc` prototype response 的一致性，计算
每个 tile / teacher 的软可靠性权重。Prototype tiles 定义固定 HCC semantic blueprint，
不作为 hard-label 图像分类训练集。

The public scientific design is maintained in
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md).

公开科学设计以 [`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md) 为准。

## Repository Scope

This repository contains:

- data contracts and package readers/writers for offline tile and feature caches;
- student model, loss, training, and evaluation scaffolding;
- public scientific design for an HCC-centered vertical foundation
  representation model;
- smoke-test utilities based on synthetic data.

本仓库包含：

- 离线 tile 与 feature cache 的数据合同和读写实现；
- 学生模型、loss、训练和评估脚手架；
- 面向 HCC 垂直 foundation representation model 的公开科学设计；
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
HCC WSIs
  -> tile-level morphology corpus
  -> multi-teacher morphology priors
  -> HCC prototype-shaped student embedding
  -> blinded result-level morphology retrieval evaluation
```

IatroCache (`.iac`) is an internal cache format for tile and teacher-feature
artifacts. It is an implementation detail, not the scientific contribution and
not a general-purpose pathology data format.

IatroCache（`.iac`）是本仓库内部用于离线 tile cache 与 teacher feature cache 的工程合同，
属于训练数据准备实现细节，不是论文或模型工作的科学贡献，也不是通用病理格式。

## Training Data Baseline

The current local image-tile IAC baseline contains 928 effective tile packages
and 13,964,919 tiles. This count is the reference scale for subsequent
multi-teacher distillation training.

当前本地 image-tile IAC 基准包含 928 个有效 tile package，共 13,964,919 个 tile。
该规模作为后续多 teacher 蒸馏训练的基准。

## Smoke Test

The contract smoke test uses synthetic data and validates package loading,
prototype loading, training, checkpointing, evaluation output, and throughput
benchmarking.

```bash
cd hcc-sempath
conda activate hcc-sempath
bash scripts/run_contract_smoke.sh
```

## Scientific Prototype Design

The current prototype design is part of the scientific method and is documented
in [`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md). It uses four
mutually exclusive Level-1 tissue states and ten non-exclusive Level-2 morphology
presence attributes. Level-1 and Level-2 are parallel prototype axes rather than
a parent-child taxonomy; the manuscript-grade primary evaluation is not
prototype reconstruction.

当前 prototype 设计属于科学方法本身，见
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md)。它包含 4 个互斥 Level-1
tissue states 和 10 个非互斥 Level-2 morphology presence attributes。Level-1 与
Level-2 是并行 prototype 语义轴，不是父子层级；论文级主评价不是 prototype reconstruction。

## Reproducibility Commands

Train:

```bash
hcc-sempath train --config configs/server/train_full.example.yaml
```

Large multi-package training uses `data.dynamic_package_sampling: true` in the
example config. All packages selected by the manifest or explicit train/val
package paths participate. The loader reads package-local row chunks for I/O
locality, then builds batches from a cross-package shuffle buffer so batches are
not dominated by a single WSI/package.

For large cohorts, validation and embedding metrics should be sampled with
`train.max_val_batches` and `train.max_eval_batches`. PAMT-D trains teacher
priors first; prototype intervention starts after the teacher prior loss
plateaus or reaches `loss.max_teacher_warmup_steps`, and prototype filtering
then follows after `loss.proto_to_filter_delay_steps`.

Resume:

```bash
hcc-sempath train \
  --config configs/server/train_full.example.yaml \
  --resume outputs/hcc_sempath_v1/checkpoints/last.pt
```

Evaluate:

```bash
hcc-sempath evaluate \
  --config configs/server/train_full.example.yaml \
  --checkpoint outputs/hcc_sempath_v1/checkpoints/best_scientific_score.pt \
  --split val
```

The training `evaluate` command writes `eval_<split>.json` and reports
teacher-imitation QC plus `z_hcc` prototype-response diagnostics from
`embedding_norm`, including Level-1 accuracy, Level-2 macro F1/AUC, prototype
top-k precision, and neighborhood purity. The manuscript-grade primary
evaluation is the blinded result-level morphology retrieval protocol described
in `docs/HCC_SEMPATH_DESIGN.md`; it is not dense exval tile annotation and not a
clinical endpoint benchmark.

## Documentation

- `docs/HCC_SEMPATH_DESIGN.md`: single public scientific design and evaluation protocol.

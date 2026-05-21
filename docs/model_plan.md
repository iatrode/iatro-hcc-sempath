# HCC-SemPath Model Plan / 模型方案

This document describes the current model design for HCC-SemPath. It is written as a public-facing technical plan for future open-source release.

本文档描述 HCC-SemPath 当前模型设计，采用面向未来开源发布的公开技术方案写法。

## 1. Objective / 目标

HCC-SemPath aims to learn a compact, reusable, HCC-specific pathology embedding model. The model is not designed as a single-task diagnostic classifier. Its primary output is a shared representation space for hepatocellular carcinoma histopathology.

HCC-SemPath 旨在学习一个轻量、可复用、面向 HCC 专病的病理 embedding 模型。模型不是单任务诊断分类器，其主要输出是面向肝细胞癌组织病理的共享表征空间。

The intended representation should support morphology retrieval, semantic organization, weakly supervised learning, downstream adaptation, and efficient large-scale WSI processing.

该表征应支持形态检索、语义组织、弱监督学习、下游适配，以及大规模 WSI 的高效处理。

## 2. Representation Strategy / 表征策略

The central design is a shared student encoder that learns an HCC-oriented embedding `z_hcc`.

核心设计是一个共享 student encoder，用于学习 HCC-oriented embedding `z_hcc`。

During training, multiple pathology foundation models can serve as teachers. Each teacher may have a different training objective, context scale, and feature geometry. HCC-SemPath therefore uses teacher-specific projection heads instead of forcing all teacher features into one averaged target.

训练阶段可以使用多个病理基础模型作为 teacher。不同 teacher 可能具有不同训练目标、上下文尺度和 feature geometry。因此，HCC-SemPath 使用 teacher-specific projection heads，而不是把多个 teacher feature 强行平均成一个目标。

```text
image tile
  -> shared student encoder
  -> z_hcc
       -> head_teacher_a -> teacher A feature space
       -> head_teacher_b -> teacher B feature space
       -> head_teacher_c -> teacher C feature space
```

The teacher-specific heads are alignment modules used during training. The reusable model output is `z_hcc`.

teacher-specific heads 是训练阶段的对齐模块。可复用的模型输出是 `z_hcc`。

## 3. Training Stages / 训练阶段

### 3.1 Stage 1: multi-teacher morphology distillation / 阶段一：多教师形态蒸馏

The first stage injects generic pathology morphology priors into the student encoder.

第一阶段将通用病理形态先验注入 student encoder。

Candidate objectives:

候选目标函数：

```text
L_stage1 =
  sum_t w_t * L_feature_t
  + lambda_rel * L_relation
  + lambda_rank * L_neighborhood
```

Where:

其中：

- `L_feature_t` aligns each teacher-specific head to the corresponding teacher feature space.
- `L_feature_t` 将每个 teacher-specific head 对齐到对应 teacher feature space。
- `L_relation` preserves pairwise similarity structure within a mini-batch.
- `L_relation` 保留 mini-batch 内样本间相似性结构。
- `L_neighborhood` preserves nearest-neighbor or ranking structure from teacher spaces when available.
- `L_neighborhood` 在可用时保留 teacher space 中的近邻或排序结构。

### 3.2 Stage 2: HCC-specific weak supervision / 阶段二：HCC 专病弱监督

The second stage reshapes the shared embedding space using HCC-specific weak supervision.

第二阶段使用 HCC 专病弱监督重塑共享 embedding space。

Potential supervision sources:

潜在监督来源：

- expert-defined morphology anchors;
- 专家定义的形态锚点；
- weak region labels;
- 弱区域标签；
- slide-level or region-level pathology descriptors;
- 切片级或区域级病理描述；
- HCC-specific structural patterns;
- HCC 专病结构模式；
- curated retrieval groups;
- 人工整理的检索组；
- other disease-domain signals that do not require dense pixel-level annotation.
- 其他不需要密集像素级标注的专病信号。

The role of this stage is not to imitate a generic teacher more closely. It is to organize HCC-relevant morphology into a representation geometry that is useful for HCC computational pathology.

该阶段的作用不是更接近地模仿通用 teacher，而是把 HCC 相关形态组织成对 HCC 计算病理有用的表征几何。

## 4. Student Backbone / Student 主干

The current compact student candidate is a ViT-S/14 style encoder.

当前轻量 student 候选为 ViT-S/14 风格编码器。

Current configuration:

当前配置：

```yaml
backbone_name: vit_small_patch14_dinov2.lvd142m
teacher_dim: 1536
pretrained: true
```

Estimated trainable parameter count:

估计可训练参数量：

- backbone: approximately 21M parameters;
- 主干：约 21M 参数；
- projection from 384 to 1536 dimensions: approximately 0.592M parameters;
- 384 到 1536 维投影：约 0.592M 参数；
- total: approximately 21.6M trainable parameters.
- 合计：约 21.6M 可训练参数。

The `lvd142m` suffix refers to DINOv2 pretraining data scale and is not a parameter count.

`lvd142m` 后缀表示 DINOv2 预训练数据规模，不是参数量。

## 5. Data Organization / 数据组织

The planned full-scale training corpus is approximately:

计划中的全量训练语料规模约为：

```text
900 WSIs x approximately 20,000 tiles per WSI ~= 18,000,000 tiles
```

Data organization should support:

数据组织需要支持：

- WSI-level or patient-level splits;
- WSI 级或患者级拆分；
- multiple teacher feature sources per tile;
- 每个 tile 对应多个 teacher feature 来源；
- multi-package or per-slide package reading;
- multi-package 或 per-slide package 读取；
- reproducible tile coordinates and preprocessing metadata;
- 可复现的 tile 坐标和预处理元数据；
- separation of public schemas from private institutional data.
- 公开 schema 与私有机构数据分离。

Large raw data, large feature packages, and patient-identifiable manifests should not be committed to the public repository.

大型原始数据、大规模 feature package 和可识别患者身份的 manifest 不应提交到公开仓库。

## 6. Evaluation Plan / 评估方案

Evaluation should distinguish teacher imitation from HCC-specific representation quality.

评估应区分 teacher imitation 与 HCC-specific representation quality。

### 6.1 Teacher-alignment metrics / Teacher 对齐指标

- feature cosine similarity;
- feature cosine similarity；
- normalized feature MSE;
- normalized feature MSE；
- pairwise relation preservation;
- pairwise relation preservation；
- nearest-neighbor overlap;
- nearest-neighbor overlap；
- teacher-specific retrieval consistency.
- teacher-specific retrieval consistency。

These metrics verify successful distillation, but they do not by themselves prove HCC-specific representation value.

这些指标用于验证蒸馏是否成功，但不能单独证明 HCC 专病表征价值。

### 6.2 HCC representation metrics / HCC 表征指标

- expert-reviewed morphology retrieval;
- 专家审阅的形态检索；
- HCC semantic anchor response consistency;
- HCC 语义锚点响应一致性；
- clustering or neighborhood purity for HCC-relevant morphology groups;
- HCC 相关形态组的聚类或邻域纯度；
- cross-cohort stability;
- 跨队列稳定性；
- downstream adaptation with lightweight heads or weakly supervised modules;
- 使用轻量 head 或弱监督模块进行下游适配；
- efficiency in parameter count, memory, throughput, and WSI-level processing time.
- 参数量、显存、吞吐和 WSI 级处理时间方面的效率。

## 7. Required Baselines / 必要基线

Required comparisons:

必要对照：

1. individual teacher embeddings;
1. 单个 teacher embedding；
2. single-teacher student distillation;
2. 单 teacher student 蒸馏；
3. multi-teacher distillation without HCC weak supervision;
3. 无 HCC 弱监督的多 teacher 蒸馏；
4. HCC weak supervision without multi-teacher distillation;
4. 无多 teacher 蒸馏的 HCC 弱监督；
5. full HCC-SemPath model;
5. 完整 HCC-SemPath 模型；
6. alternative student backbone or projection-head capacities when feasible.
6. 条件允许时比较不同 student backbone 或 projection-head capacity。

## 8. Engineering Requirements / 工程要求

Before full-scale training, the implementation should support:

全量训练前，实现应支持：

- CUDA mixed precision training;
- CUDA 混合精度训练；
- multiple teacher feature packages;
- 多 teacher feature package；
- multi-package dataset loading;
- multi-package dataset loading；
- WSI-level split management;
- WSI-level split 管理；
- sampled validation subsets;
- 抽样 validation subset；
- teacher metadata recording, including model name, version, preprocessing, feature dimension, and normalization convention.
- teacher metadata 记录，包括模型名称、版本、预处理、feature dimension 和 normalization convention。

## 9. Public Release Notes / 公开发布注意事项

The repository should expose code, schemas, configuration templates, small fixtures, model cards, and reproducibility scripts.

仓库应公开代码、schema、配置模板、小型测试数据、model card 和复现脚本。

The repository should not expose private WSIs, large feature artifacts, patient-identifiable metadata, or institutional file paths.

仓库不应公开私有 WSI、大规模 feature artifact、可识别患者身份的元数据或机构内部文件路径。

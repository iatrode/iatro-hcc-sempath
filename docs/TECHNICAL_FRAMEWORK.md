# HCC-SemPath Technical Framework / 技术框架

This document describes the public technical framework for HCC-SemPath.

本文档描述 HCC-SemPath 面向公开发布的技术框架。

## 1. Purpose / 目的

HCC-SemPath aims to build a lightweight vertical pathology representation model
for hepatocellular carcinoma. It is designed to produce compact reusable
embeddings for HCC histopathology rather than a general-purpose pathology
foundation model or task-specific diagnostic, prognostic, or
visual-question-answering outputs.

HCC-SemPath 旨在构建面向肝细胞癌的轻量级垂直病理表征模型。模型输出面向 HCC
组织病理的紧凑可复用 embedding，而不是通用病理基础模型，也不是单一诊断、预后或视觉问答结果。

## 2. Core Hypothesis / 核心假设

General pathology foundation models contain useful morphology priors, but their
embedding spaces are not optimized for HCC-specific histopathology semantics.
A compact student model can integrate heterogeneous teacher priors and then be
reshaped by HCC-specific weak supervision into a disease-oriented embedding space.

通用病理基础模型包含有价值的形态学先验，但其 embedding space 并非为
HCC 专病组织病理语义优化。轻量 student 可以整合异质 teacher 先验，并通过
HCC 专病弱监督进一步重塑为 disease-oriented embedding space。

The framework assumes that teacher agreement and disagreement may both contain
useful information. Agreement can provide relatively stable morphology priors,
while disagreement may highlight morphology that is difficult, context-sensitive,
or not consistently organized by a single generic teacher.

该框架认为 teacher 的共识与分歧都可能包含有价值的信息。共识可提供相对稳定的
形态学先验，而分歧可能反映复杂形态、上下文敏感性，或单一通用 teacher 未能稳定组织的区域。

## 3. System Modules / 系统模块

### 3.1 WSI preprocessing and manifest contract / WSI 预处理与 manifest 约定

Inputs are tiled at a defined magnification, resolution, and tissue-filtering
policy. Each retained tile is represented by a reproducible manifest row.

输入 WSI 按固定倍率、分辨率和组织筛选策略切块。每个保留 tile 都由可复现的
manifest 行表示。

Minimum manifest fields:

最小 manifest 字段：

```text
tile_id, patient_id, slide_id, tile_path, x, y, split
```

Public repositories should provide schemas and synthetic examples, not private
institutional manifests.

公开仓库应提供 schema 和 synthetic examples，而不是私有机构 manifest。

### 3.2 Multi-teacher feature extraction / 多 teacher 特征提取

Multiple pathology foundation models may be used as teachers. Each teacher source
must record model name, version, feature dimension, preprocessing, normalization
convention, and extraction environment.

可使用多个病理基础模型作为 teacher。每个 teacher 来源都需要记录模型名称、版本、
feature dimension、预处理、normalization convention 和特征提取环境。

Teacher features are training artifacts and should generally be stored outside git.

Teacher feature 属于训练 artifact，通常应存储在 git 之外。

Optional cached agreement or disagreement summaries may also be generated for
selective distillation experiments.

可选生成缓存 agreement / disagreement summary，用于选择性蒸馏实验。

### 3.3 Shared HCC embedding with teacher-specific heads / 共享 HCC embedding 与 teacher-specific heads

The student model learns a shared HCC embedding `z_hcc`. During training,
teacher-specific heads align this shared embedding to each teacher feature space.

Student 模型学习共享 HCC embedding `z_hcc`。训练阶段，teacher-specific heads
将该共享 embedding 对齐到各 teacher feature space。

```text
tile image -> student encoder -> z_hcc
                              -> head_teacher_a
                              -> head_teacher_b
                              -> head_teacher_c
```

The teacher-alignment heads are not the primary released representation. The
primary reusable output is `z_hcc`.

teacher-alignment heads 不是主要发布表征。主要可复用输出是 `z_hcc`。

This distinction is important because a downstream task head attached to a fixed
teacher embedding does not modify the shared embedding geometry itself. In
contrast, the student representation can combine morphology priors from multiple
teachers and be reorganized by HCC-specific weak supervision.

这一点很重要，因为固定 teacher embedding 后接 downstream head 并不会改变共享
embedding geometry 本身。相比之下，student representation 可以整合多个 teacher
的形态学先验，并通过 HCC 专病弱监督进一步重组。

### 3.4 Consensus-disagreement guided selective distillation / 基于共识与分歧的选择性蒸馏

Teacher agreement and disagreement can be treated as soft reliability signals.
Tiles with high teacher agreement may receive stronger morphology distillation,
while tiles with higher disagreement may gradually receive relatively stronger
HCC semantic shaping.

teacher agreement 与 disagreement 可作为软可靠性信号。高共识 tile 可以接受更强
形态蒸馏，而高分歧 tile 则可逐步接受更强的 HCC semantic shaping。

Potential signals include neighborhood overlap, pairwise similarity consistency,
rank consistency, or normalized disagreement scores computed from teacher feature
relations.

候选信号包括 neighborhood overlap、pairwise similarity consistency、rank consistency，
以及基于 teacher feature relation 计算的 normalized disagreement score。

These signals should be interpreted conservatively as optimization guidance and
representation diagnostics rather than direct biological labels.

这些信号应被保守解释为优化引导与表征诊断，而不是直接生物学标签。

### 3.5 HCC-specific weak supervision / HCC 专病弱监督

HCC-specific weak supervision should reshape the embedding space beyond teacher
imitation. Candidate signals include expert morphology anchors, weak region
labels, structured pathology descriptions, curated retrieval sets, and slide- or
region-level disease-domain labels.

HCC 专病弱监督应将 embedding space 从 teacher imitation 推向专病语义空间。候选信号
包括专家形态锚点、弱区域标签、结构化病理描述、人工整理检索集合，以及切片级或
区域级专病标签。

Whenever possible, HCC semantic objectives should act directly on the shared
`z_hcc` representation because `z_hcc` is the reusable model output.

在条件允许时，HCC semantic objective 应直接作用于共享 `z_hcc`，因为 `z_hcc`
是模型的可复用输出。

A practical strategy is to activate HCC semantic shaping gradually instead of
using strong prototype constraints from the beginning of training.

一个可行策略是渐进启用 HCC semantic shaping，而不是从训练开始阶段就施加强 prototype 约束。

## 4. Evaluation / 评估

Evaluation is divided into two groups.

评估分为两组。

### 4.1 Teacher-alignment metrics / Teacher 对齐指标

- cosine similarity and normalized feature MSE;
- cosine similarity 和 normalized feature MSE；
- pairwise relation preservation;
- pairwise relation preservation；
- nearest-neighbor overlap;
- nearest-neighbor overlap；
- teacher-specific retrieval consistency;
- teacher-specific retrieval consistency；
- teacher agreement/disagreement diagnostics.
- teacher agreement / disagreement 诊断。

These metrics verify distillation quality but do not by themselves prove HCC-specific
representation value.

这些指标验证蒸馏质量，但不能单独证明 HCC 专病表征价值。

### 4.2 HCC representation metrics / HCC 表征指标

- expert-reviewed morphology retrieval;
- 专家审阅的形态检索；
- HCC semantic anchor consistency;
- HCC 语义锚点一致性；
- morphology-group clustering or neighborhood purity;
- 形态组聚类或邻域纯度；
- cross-cohort stability;
- 跨队列稳定性；
- lightweight downstream adaptation;
- 轻量下游适配；
- stratified evaluation across low- and high-disagreement tile groups;
- 按低分歧与高分歧 tile group 分层评估；
- parameter count, memory, throughput, and WSI-level processing time.
- 参数量、显存、吞吐和 WSI 级处理时间。

## 5. Required Baselines / 必要对照

Public experiments should distinguish the proposed representation from simpler
alternatives.

公开实验应将本项目表征与更简单的替代方案区分开。

Required baselines:

必要基线：

1. individual teacher embeddings;
1. 单个 teacher embedding；
2. single-teacher distillation;
2. 单 teacher 蒸馏；
3. multi-teacher distillation without HCC weak supervision;
3. 无 HCC 弱监督的多 teacher 蒸馏；
4. multi-teacher distillation without selective weighting;
4. 无选择性权重的多 teacher 蒸馏；
5. fixed teacher embedding with the same HCC semantic head or prototype module;
5. 固定 teacher embedding 后接相同 HCC semantic head 或 prototype module；
6. full HCC-SemPath training.
6. 完整 HCC-SemPath 训练。

## 6. Public Reproducibility / 公开可复现

The public repository should contain code, schemas, configuration templates,
small synthetic fixtures, public-safe benchmark summaries, documentation, and
model cards.

公开仓库应包含代码、schema、配置模板、小型 synthetic fixtures、公开安全的
benchmark summaries、文档和 model cards。

The public repository should not contain raw WSIs, large feature artifacts,
large checkpoints, patient-identifiable metadata, institutional file paths, or
production-scale per-tile tables.

公开仓库不应包含原始 WSI、大型 feature artifact、大型 checkpoint、可识别患者身份的
元数据、机构内部路径或生产规模每 tile 表。

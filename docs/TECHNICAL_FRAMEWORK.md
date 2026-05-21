# HCC-SemPath Technical Framework / 技术框架

This document describes the public technical framework for HCC-SemPath.

本文档描述 HCC-SemPath 面向公开发布的技术框架。

## 1. Purpose / 目的

HCC-SemPath aims to build a reusable HCC-specific pathology representation model.
It is designed to produce compact embeddings for HCC histopathology rather than
task-specific diagnostic, prognostic, or visual-question-answering outputs.

HCC-SemPath 旨在构建可复用的 HCC 专病病理表征模型。模型输出面向 HCC
组织病理的轻量 embedding，而不是单一诊断、预后或视觉问答结果。

## 2. Core Hypothesis / 核心假设

General pathology foundation models contain useful morphology priors, but their
embedding spaces are not optimized for HCC-specific histopathology semantics.
A compact student model can integrate heterogeneous teacher priors and then be
reshaped by HCC-specific weak supervision into a disease-oriented embedding space.

通用病理基础模型包含有价值的形态学先验，但其 embedding space 并非为
HCC 专病组织病理语义优化。轻量 student 可以整合异质 teacher 先验，并通过
HCC 专病弱监督进一步重塑为 disease-oriented embedding space。

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

### 3.4 HCC-specific weak supervision / HCC 专病弱监督

HCC-specific weak supervision should reshape the embedding space beyond teacher
imitation. Candidate signals include expert morphology anchors, weak region
labels, structured pathology descriptions, curated retrieval sets, and slide- or
region-level disease-domain labels.

HCC 专病弱监督应将 embedding space 从 teacher imitation 推向专病语义空间。候选信号
包括专家形态锚点、弱区域标签、结构化病理描述、人工整理检索集合，以及切片级或
区域级专病标签。

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
- teacher-specific retrieval consistency.
- teacher-specific retrieval consistency。

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
4. HCC weak supervision without teacher distillation;
4. 无 teacher 蒸馏的 HCC 弱监督；
5. full HCC-SemPath training.
5. 完整 HCC-SemPath 训练。

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

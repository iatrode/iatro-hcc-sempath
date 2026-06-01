# HCC-SemPath Model Plan / 模型方案

This document describes the current model design for HCC-SemPath. It is written as a public-facing technical plan for future open-source release.

本文档描述 HCC-SemPath 当前模型设计，采用面向未来开源发布的公开技术方案写法。

## 1. Objective / 目标

HCC-SemPath aims to learn a lightweight vertical pathology representation model for hepatocellular carcinoma. The model is not designed as a general-purpose pathology foundation model or a single-task diagnostic classifier. Its primary output is a compact shared embedding space for HCC histopathology.

HCC-SemPath 旨在学习一个面向肝细胞癌的轻量级垂直病理表征模型。模型不是通用病理基础模型，也不是单任务诊断分类器；其主要输出是面向 HCC 组织病理的紧凑共享 embedding space。

The intended representation should support morphology retrieval, semantic organization, weakly supervised learning, downstream adaptation, and efficient large-scale WSI processing with a compact deployable model.

该表征应通过紧凑、可部署的模型支持形态检索、语义组织、弱监督学习、下游适配，以及大规模 WSI 的高效处理。

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

A practical distinction from a simple downstream head on top of an existing teacher is that HCC-SemPath aims to modify the shared representation geometry itself. A downstream head can read a fixed teacher space, while the student representation can combine teacher consensus, reduce over-dependence on any single teacher, and be further organized by HCC-specific weak supervision.

与直接在现有 teacher feature 后接下游 head 相比，HCC-SemPath 的目标是调整共享表征空间本身。下游 head 只能读取固定 teacher space，而 student representation 可以整合 teacher 共识、降低对单一 teacher 的依赖，并通过 HCC 专病弱监督进一步组织表征几何。

## 3. Training Strategy / 训练策略

### 3.1 Multi-teacher morphology distillation / 多教师形态蒸馏

The first component injects general pathology morphology priors into the student encoder.

第一部分将通用病理形态先验注入 student encoder。

Candidate objectives:

候选目标函数：

```text
L_distill =
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

### 3.2 Consensus-disagreement guided selective distillation / 基于共识与分歧的选择性蒸馏

Multi-teacher supervision should not treat all tiles as equally reliable distillation targets. When multiple teachers produce similar neighborhood or similarity structures for a tile, their agreement can be treated as a relatively stable morphology prior. When teachers disagree, the tile may reflect teacher-specific bias, difficult morphology, context sensitivity, or HCC-relevant patterns that are not consistently represented by a single generic teacher.

多 teacher 监督不应把所有 tile 都视为同等可靠的蒸馏目标。当多个 teacher 对某个 tile 给出相似的近邻或相似性结构时，这种共识可作为相对稳定的形态先验。当 teacher 之间存在明显分歧时，该 tile 可能反映 teacher-specific bias、复杂形态、上下文敏感性，或单一通用 teacher 未能稳定表达的 HCC 相关模式。

The planned strategy is to use teacher agreement as a soft reliability signal rather than a hard label:

计划中的策略是将 teacher agreement 作为软可靠性信号，而不是硬标签：

```text
high teacher agreement      -> stronger morphology distillation
low teacher agreement       -> weaker hard imitation and stronger HCC semantic shaping
```

This design is intended to avoid simple averaging across heterogeneous teachers. It also provides a reason to train a shared student representation instead of only attaching an HCC head to a fixed teacher embedding.

该设计旨在避免对异质 teacher 进行简单平均，同时也解释了为何需要训练共享 student representation，而不仅是在固定 teacher embedding 后接 HCC head。

Potential agreement signals include pairwise teacher similarity consistency, teacher-neighborhood overlap, teacher-rank consistency, or normalized disagreement scores computed within sampled batches or cached feature subsets. These scores should be used as weighting signals and diagnostics, not as direct biological labels.

潜在 agreement signal 包括 teacher 之间的相似性结构一致性、teacher-neighborhood overlap、teacher-rank consistency，或基于 sampled batch / cached feature subset 计算的 normalized disagreement score。这些分数应作为 loss weighting 和诊断指标，而不是直接生物学标签。

### 3.3 Prototype-guided HCC weak supervision on `z_hcc` / Prototype 引导的 HCC 专病弱监督

HCC-specific weak supervision should reshape the shared embedding space beyond teacher imitation. In the selective-distillation setting, this component should be activated gradually rather than applied as a strong constraint from the first training step.

HCC 专病弱监督应将共享 embedding space 从 teacher imitation 推向专病语义空间。在选择性蒸馏设定下，该部分应渐进启用，而不是从训练第一步开始施加强约束。

The weak-supervision mechanism is prototype-based. A prototype represents one reusable HCC morphology or tissue-context concept in the `z_hcc` space. Prototypes are not a fixed class list in model code. They are loaded at runtime from a prototype directory, a prebuilt prototype package, or a `.pt/.pth` payload with metadata, so new prototypes can be added, removed, merged, or revised without changing the student architecture.

弱监督机制采用 prototype。一个 prototype 表示 `z_hcc` 空间中的一个可复用 HCC 形态或组织上下文概念。Prototype 不是写死在模型代码中的类别表，而是在运行时从 prototype 目录、预构建 prototype package，或带 metadata 的 `.pt/.pth` 文件加载，因此新 prototype 的添加、删除、合并或修订不需要修改 student 架构。

Prototype sources:

潜在监督来源：

- expert-defined morphology prototypes;
- 专家定义的形态 prototype；
- semantic prototypes initialized from curated concept embeddings or reviewed examples;
- 基于 curated concept embeddings 或人工审阅样本初始化的语义原型；
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

The HCC semantic loss should act on the shared `z_hcc` representation, because `z_hcc` is the reusable model output. Teacher-specific heads remain useful for teacher alignment, but prototype constraints should not be applied only inside teacher-specific feature spaces.

HCC semantic loss 应直接作用于共享 `z_hcc`，因为 `z_hcc` 是可复用模型输出。Teacher-specific heads 仍可用于 teacher alignment，但 prototype 约束不应只作用在各 teacher-specific feature space 内。

### 3.4 Prototype module and dynamic prototype registry / Prototype 模块与动态注册

The prototype system should be implemented as a module with three public inputs:

Prototype 系统应实现为一个模块，并支持三类公开输入：

```text
prototype_dir/
  prototype_manifest.yaml
  prototypes.pt or one file per prototype group

prototype_package.pth
  {
    "version": 1,
    "prototypes": Tensor[num_prototypes, dim],
    "names": list[str],
    "groups": optional list[str],
    "levels": list[int],
    "exclusive": list[bool],
    "thresholds": optional Tensor[num_prototypes],
    "source": metadata
  }

config:
  data.prototype_path or data.prototype_dir
```

The model must treat the number and identity of prototypes as data, not code. Prototype names, group membership, semantic levels, exclusivity flags, confidence thresholds, source notes, and initialization counts belong in metadata. This keeps future open-source releases clean: public code exposes the mechanism, while institution-specific prototype curation can remain outside the repository.

模型必须把 prototype 的数量和身份视为数据，而不是代码。Prototype 名称、group 归属、语义层级、互斥标记、置信阈值、来源说明和初始化样本量应写入 metadata。这样开源仓库只公开机制，机构内 prototype curate 过程可以留在仓库外部。

### 3.5 Prototype-filtered distillation / Prototype 筛选蒸馏

Prototype is not only an auxiliary label signal. It should control how much each teacher is trusted for each tile. Different teachers may organize the same HCC morphology differently, so direct teacher averaging can force conflicting semantics into `z_hcc`. Prototype response in the shared HCC semantic space should provide a filtering signal:

Prototype 不只是辅助标签信号，还应控制每个 tile 对每个 teacher 的信任程度。不同 teacher 对同一 HCC 形态的语义组织不一定一致，直接平均会把冲突压进 `z_hcc`。共享 HCC 语义空间中的 prototype response 应提供筛选信号：

```text
z_hcc -> prototype responses r_i
teacher_t feature/head -> teacher-specific semantic response q_ti
agreement(z_hcc, teacher_t, prototypes) -> reliability alpha_t

L = sum_t alpha_t * L_teacher_t
  + lambda_proto * L_proto_multi_label
  + lambda_cons * L_prototype_consistency
  + lambda_rel * L_relation
```

`alpha_t` should be a soft weight, not a hard exclusion. A practical form is to compute teacher reliability from agreement between the teacher-specific response and the current `z_hcc` prototype response, optionally combined with cross-teacher disagreement. Tiles whose teachers agree with HCC prototype semantics receive stronger distillation; tiles with teacher conflict receive weaker hard imitation and stronger prototype shaping.

`alpha_t` 应是软权重，不是硬排除。可行做法是根据 teacher-specific response 与当前 `z_hcc` prototype response 的一致性计算 teacher reliability，并可结合跨 teacher disagreement。与 HCC prototype 语义一致的 tile 接受更强蒸馏；teacher 冲突明显的 tile 降低硬模仿权重，并增强 prototype shaping。

The first implementation should use a bounded reliability weight:

第一版实现采用有界 reliability weight：

```text
alpha_t = clamp(alpha_min + (1 - alpha_min) * agreement_t, alpha_min, 1)
```

This preserves gradient signal from all teachers while preventing one inconsistent teacher from dominating the shared HCC space.

这能保留所有 teacher 的梯度信号，同时避免单个语义不一致的 teacher 主导共享 HCC 空间。

### 3.6 Two-level prototype target / 两级 prototype 目标

Prototype supervision should be two-level. Level 1 encodes the primary mutually exclusive state, such as tumor versus non-tumor / background tissue. Level 2 encodes non-exclusive morphology and microenvironment attributes, such as lymphocytic infiltration, necrosis, fibrosis, steatosis, vascular context, and background liver changes.

Prototype supervision 应分两级。Level 1 表示互斥的主状态，例如 tumor versus non-tumor / background tissue。Level 2 表示非互斥的形态与微环境属性，例如淋巴细胞浸润、坏死、纤维化、脂肪变、血管上下文和背景肝改变。

The current taxonomy uses nine L1 primary prototypes and seven L2 attribute
prototypes. L1 covers HCC architectural patterns, background liver, stromal
regions, degenerative material, indeterminate tissue, and artifact / non-tissue.
L2 covers necrotic, hemorrhagic-blood-rich, bile-pigment-rich, inflammatory-rich,
fibrotic, steatotic-vacuolated, and interface-capsule attributes. The detailed
classification rules are maintained in `docs/PROTOTYPE_TAXONOMY.md`.

当前 taxonomy 包含 9 个 L1 主 prototype 和 7 个 L2 属性 prototype。L1 覆盖 HCC
结构型、背景肝、纤维间质、退变物、判读不确定组织以及 artifact / non-tissue。L2 覆盖
necrotic、hemorrhagic-blood-rich、bile-pigment-rich、inflammatory-rich、fibrotic、
steatotic-vacuolated 和 interface-capsule。详细分类规则见
`docs/PROTOTYPE_TAXONOMY.md`。

The default target is:

默认目标是：

```text
y_primary: one mutually exclusive distribution over level-1 prototypes
y_attr:    soft affinities in [0, 1] over level-2 prototypes
```

Level-1 supervision should use softmax cross-entropy or soft-label KL over primary states. Level-2 supervision should use binary cross entropy, focal BCE, similarity regression, or positive-unlabeled contrastive loss. When no explicit tile-level label is available, use prototype consistency as a self-training signal with confidence thresholds recorded in prototype metadata.

Level 1 使用 softmax cross-entropy 或 soft-label KL 约束主状态；Level 2 使用 BCE、focal BCE、similarity regression 或 positive-unlabeled contrastive loss。没有显式 tile-level 标签时，使用带置信阈值的 prototype consistency self-training，阈值写入 prototype metadata。

The implemented supervised path applies the prototype objective directly to
the normalized reusable embedding (`embedding_norm`): Level 1 uses cross
entropy over runtime-loaded primary prototypes, and Level 2 uses BCE over
runtime-loaded attribute prototypes. Prototype labels are resolved by prototype
name from `data.prototype_supervision_manifest_path`, so the prototype set can
evolve without changing model code.

当前已实现的监督路径直接作用于归一化的可复用 embedding（`embedding_norm`）：Level 1
对运行时加载的主 prototype 使用 cross entropy，Level 2 对运行时加载的属性 prototype
使用 BCE。Prototype label 通过 `data.prototype_supervision_manifest_path` 中的名称
与 prototype package 对齐，因此 prototype 集合演进不需要改模型代码。

### 3.7 Loss schedule / Loss 阶段调度

A practical schedule is:

一个可行训练调度为：

```text
stage 0: contract smoke and feature sanity checks only
stage 1: teacher distillation dominates; prototype loss is zero or very small
stage 2: prototype response loss warms up; teacher reliability remains weakly bounded
stage 3: prototype-filtered distillation is active; high-conflict teacher signals are down-weighted
stage 4: freeze or slow prototype updates; evaluate retrieval, clustering, and cross-cohort stability
```

This can be implemented either as explicit staged training or as a single training run with scheduled loss weights. In both cases, the scientific interpretation should remain conservative: the objective is to organize HCC-relevant morphology, not to claim that prototypes are exhaustive or definitive pathology categories.

该策略既可以实现为显式分阶段训练，也可以实现为单次训练中的 scheduled loss weights。无论采用哪种工程实现，科学表述均应保持克制：目标是组织 HCC 相关形态，而不是声称 prototypes 构成穷尽或确定性的病理类别。

At multi-million tile scale, validation and embedding metrics should use fixed
or bounded sampled subsets during training. Full validation should be reserved
for selected checkpoints, because collecting all validation embeddings every
epoch is not memory- or time-efficient.

在百万级 tile 训练规模下，训练过程中的 validation 与 embedding metric 应使用固定或有上限的抽样子集。全量 validation 应保留给关键 checkpoint，因为每个 epoch 收集全部 validation embedding 在内存和时间上都不合适。

## 4. Student Backbone / Student 主干

The current compact student candidate is a ViT-S/14 style encoder.

当前轻量 student 候选为 ViT-S/14 风格编码器。

Current configuration:

当前配置：

```yaml
backbone_name: vit_small_patch14_reg4_dinov2.lvd142m
embedding_dim: 1536
teacher_dims:
  gigapath: 1536
  h_optimus_1: 1536
  uni2_h: 1536
  virchow2: 2560
pretrained: true
```

The selected student uses the register-token DINOv2 ViT-S/14 variant (`reg4`).
It keeps the compact ViT-S/14 scale while using four register tokens for more
stable representation learning.

当前 student 采用带 register token 的 DINOv2 ViT-S/14 变体（`reg4`）。它保持
ViT-S/14 的轻量规模，同时使用 4 个 register token 以提高表征稳定性。

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

- a generated training manifest that lists per-WSI IAC stems and train/val/exval cohort membership;
- 生成式 training manifest，用于记录 per-WSI IAC stem 及 train/val/exval 队列归属；
- WSI-level or patient-level splits;
- WSI 级或患者级拆分；
- multiple teacher feature sources per tile;
- 每个 tile 对应多个 teacher feature 来源；
- convention-based teacher feature package resolution from WSI stems and configured teachers;
- 基于 WSI stem 与配置的 teacher 通过命名约定解析 teacher feature package；
- multi-package or per-slide package reading without changing the image-tile IAC format;
- 在不改变 image-tile IAC 格式的前提下进行 multi-package 或 per-slide package 读取；
- package-local chunk reads with cross-package shuffle-buffer batch construction;
- 按 package 聚合读取 chunk，并通过跨 package shuffle buffer 构建 batch；
- reproducible tile coordinates and preprocessing metadata;
- 可复现的 tile 坐标和预处理元数据；
- optional cached teacher-agreement or disagreement summaries for efficient selective distillation;
- 可选缓存 teacher agreement / disagreement summary，用于高效选择性蒸馏；
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
- teacher-specific retrieval consistency;
- teacher-specific retrieval consistency；
- teacher agreement and disagreement diagnostics.
- teacher agreement / disagreement 诊断指标。

These metrics verify successful distillation and characterize teacher consistency, but they do not by themselves prove HCC-specific representation value.

这些指标用于验证蒸馏是否成功，并刻画 teacher consistency，但不能单独证明 HCC 专病表征价值。

### 6.2 HCC representation metrics / HCC 表征指标

- expert-reviewed morphology retrieval;
- 专家审阅的形态检索；
- HCC semantic prototype response consistency;
- HCC 语义 prototype response 一致性；
- clustering or neighborhood purity for HCC-relevant morphology groups;
- HCC 相关形态组的聚类或邻域纯度；
- direct `z_hcc` Level-1 accuracy, Level-2 macro F1/AUC, prototype top-k precision, and neighborhood purity on supervised prototype tiles;
- 基于 prototype 监督 tile 的直接 `z_hcc` Level-1 accuracy、Level-2 macro F1/AUC、prototype top-k precision 与 neighborhood purity；
- cross-cohort stability;
- 跨队列稳定性；
- downstream adaptation with lightweight heads or weakly supervised modules;
- 使用轻量 head 或弱监督模块进行下游适配；
- stratified analysis across low- and high-disagreement tile groups;
- 按低分歧与高分歧 tile group 分层分析；
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
4. multi-teacher distillation without selective weighting;
4. 无选择性权重的多 teacher 蒸馏；
5. HCC weak supervision without multi-teacher distillation;
5. 无多 teacher 蒸馏的 HCC 弱监督；
6. fixed-teacher embedding with the same HCC semantic head or prototype module;
6. 固定 teacher embedding 后接相同 HCC semantic head 或 prototype module；
7. full HCC-SemPath model;
7. 完整 HCC-SemPath 模型；
8. alternative student backbone or projection-head capacities when feasible.
8. 条件允许时比较不同 student backbone 或 projection-head capacity。

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
- manifest-based public held-out external validation;
- 基于 manifest 的 public held-out external validation；
- feature-cache-aware training sampling for current whole-matrix feature packages;
- 面向当前 whole-matrix feature package 的 feature-cache-aware training sampling；
- sampled validation subsets;
- 抽样 validation subset；
- optional teacher agreement/disagreement caching for selective distillation;
- 可选的 teacher agreement / disagreement 缓存，用于选择性蒸馏；
- scheduled loss weighting for gradual HCC semantic activation;
- 用于渐进启用 HCC 语义约束的 scheduled loss weighting；
- teacher metadata recording, including model name, version, preprocessing, feature dimension, and normalization convention.
- teacher metadata 记录，包括模型名称、版本、预处理、feature dimension 和 normalization convention。

## 9. Public Release Notes / 公开发布注意事项

The repository should expose code, schemas, configuration templates, small fixtures, model cards, and reproducibility scripts.

仓库应公开代码、schema、配置模板、小型测试数据、model card 和复现脚本。

The repository should not expose private WSIs, large feature artifacts, patient-identifiable metadata, or institutional file paths.

仓库不应公开私有 WSI、大规模 feature artifact、可识别患者身份的元数据或机构内部路径。

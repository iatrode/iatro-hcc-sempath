# HCC-SemPath V1 Scientific Design (DEPRECATED) / HCC-SemPath V1 科学设计（已弃用）

> Historical V1 foundation retained only for traceability. It is not the implementation
> specification for the ROI-guided Level-2 V2 upgrade.
>
> 本文档仅作为 V1 历史基线保留，不再作为 ROI 显式引导的 Level-2 V2 实现依据。

This is the single scientific design document for HCC-SemPath. It is written to
support the paper and public release narrative, not to document software
implementation details.

本文档是 HCC-SemPath 的单一科学设计文档，用于支撑论文和公开发布叙事，不记录软件工程
实现细节。

## 1. Positioning / 定位

HCC-SemPath is an HCC-centered vertical pathology foundation representation
model. Its released object is a reusable HCC histomorphology embedding space,
not a general-purpose pathology foundation model and not a diagnostic,
prognostic, report-generation, or visual-question-answering model.

HCC-SemPath 是面向肝细胞癌的垂直病理 foundation representation model。其核心产出是可
复用的 HCC 组织形态 embedding space，而不是通用病理 foundation model，也不是诊断、预后、
报告生成或视觉问答模型。

The scientific question is:

核心科学问题是：

```text
Can a compact student representation organize HCC histomorphology better than
directly using general pathology foundation-model embeddings?
```

The central claim is representation quality in an HCC morphology space. Clinical
endpoint prediction is not the main evidence for this claim.

核心主张是 HCC morphology space 中的表征质量；临床终点预测不是该主张的主要证据。

## 2. Hypothesis / 假设

General pathology foundation models encode useful morphology priors, but their
embedding geometries are not optimized for HCC-specific tissue states and
histomorphology. A compact student can integrate heterogeneous teacher priors and
then be reshaped by HCC prototype supervision into a disease-centered embedding
space.

通用病理 foundation models 包含有价值的形态学先验，但其 embedding geometry 并不是围绕
HCC-specific tissue states 和 histomorphology 优化。轻量 student 可以整合异质 teacher
priors，并通过 HCC prototype supervision 重塑为专病 embedding space。

The intended contribution is:

预期贡献是：

- a compact HCC-centered embedding `z_hcc`;
- 一个紧凑的 HCC-centered embedding `z_hcc`；
- multi-teacher morphology distillation without collapsing heterogeneous
  teacher spaces into a single averaged target;
- 多 teacher morphology distillation，但不把异质 teacher spaces 压成单一平均目标；
- prototype-adjudicated teacher reliability for HCC morphology;
- 基于 prototype 裁决的 HCC morphology teacher reliability；
- a blinded result-level morphology retrieval benchmark for HCC representation
  quality.
- 用于评价 HCC representation quality 的结果级盲态 morphology retrieval benchmark。

## 3. Representation Strategy / 表征策略

HCC-SemPath learns one shared embedding. Teacher-specific heads are training-time
alignment adapters; they are not the representation being claimed or evaluated.
The scientific object is the shared HCC embedding.

HCC-SemPath 学习一个共享 embedding。Teacher-specific heads 是训练阶段的对齐适配器，
不是论文主张或评价的表征对象。科学对象是共享 HCC embedding。

The model uses two forces:

模型由两类信号共同塑形：

1. General morphology priors from multiple pathology foundation teachers.
   来自多个通用病理 foundation teachers 的通用 morphology priors。
2. HCC-specific prototype-mediated semantic response supervision that reshapes
   the shared embedding around HCC tissue states and attributes.
   HCC-specific prototype-mediated semantic response supervision，用于把共享 embedding
   重塑到 HCC tissue states 和 attributes 周围。

PAMT-D, Prototype-Adjudicated Multi-Teacher Distillation, is the current method
formulation. Expert prototype tiles define a fixed HCC semantic blueprint. They
are used to build teacher-space semantic prototypes and no-gradient
student-space prototype centers, not as a hard-label image classification
training set. Prototype responses are used as HCC semantic response supervision
on ordinary training tiles and as soft reliability signals for teacher
distillation. A teacher signal that conflicts with HCC prototype semantics
should be down-weighted, not blindly averaged into the shared space.

PAMT-D（Prototype-Adjudicated Multi-Teacher Distillation）是当前方法表述。专家
prototype tiles 定义固定 HCC semantic blueprint，用于构建 teacher-space semantic
prototypes 和无梯度 student-space prototype centers，而不是作为 hard-label 图像分类训练集。
Prototype response 在普通训练 tile 上作为 HCC semantic response supervision，同时作为
teacher distillation 的软可靠性信号。与 HCC prototype semantics 冲突的 teacher signal
应被降权，而不是直接平均进共享空间。

## 4. Prototype Design / Prototype 设计

Prototypes define the HCC-specific semantic axes used for representation
shaping and interpretation. They are not the final evaluation judge. They should
cover the HCC morphology space as well as the development data allows, and they
are expected to come from the training side of the study design.

Prototype 定义用于 representation shaping 和解释的 HCC 专病语义轴，不是最终评价裁判。
Prototype 应尽可能覆盖 development data 中的 HCC morphology space，并且应来自研究设计中的
训练侧数据。

Each reviewed tile receives one dominant Level-1 state and zero or more Level-2
presence attributes. Level 1 and Level 2 are parallel prototype axes, not a
parent-child hierarchy.

每个审阅 tile 对应一个主导 Level-1 state，并可包含零个或多个 Level-2 presence
attributes。Level 1 和 Level 2 是并行 prototype 语义轴，不是父子层级。

### Level 1: Primary Tissue State / Level 1：主组织状态

Level 1 is mutually exclusive and encodes the dominant tissue or lesion state.

Level 1 互斥，表示 tile 的主导组织或病变状态。

| Prototype | Scientific meaning |
| --- | --- |
| `HCC-tumor` | HCC tumor-dominant morphology, without forcing architectural subtype separation at this stage. |
| `Background-liver` | Non-neoplastic background liver parenchyma. |
| `Inflammatory-stromal` | Stromal, portal, fibrous, or interface-rich tissue where inflammatory/stromal context is dominant. |
| `Degenerative-material` | Necrosis, hemorrhage, clot, bile lake, debris, or treatment-related degenerative material as the dominant state. |

### Level 2: Non-Exclusive Morphology Attributes / Level 2：非互斥形态属性

Level 2 is multi-label and encodes morphology or tissue-context presence
attributes that can coexist within one tile and can cross Level-1 states. Empty
Level-2 labels do not imply a negative Level-1 state, and Level-1 states do not
imply fixed Level-2 positives.

Level 2 是 multi-label，表示可在同一 tile 内共存、并可跨 Level-1 state 出现的形态或组织
背景 presence attributes。L2 为空不表示 L1 阴性，L1 状态也不预设固定 L2 阳性。

| Prototype | Scientific meaning |
| --- | --- |
| `hepatocellular-parenchyma-present` | Hepatocellular parenchyma or hepatocyte-like tumor/background cells are present. |
| `necrosis-present` | Necrotic component is present. |
| `hemorrhage-present` | Hemorrhage, blood pool, clot, or erythrocyte-rich area is present. |
| `bile-pigment-present` | Bile, bile pigment, or pigment deposition is present. |
| `inflammatory-cell-present` | Inflammatory cells are present. |
| `fibrous-stroma-present` | Fibrous stroma, collagen, scar-like matrix, septa, or capsule-like stroma is present. |
| `steatosis-vacuolation-present` | Steatosis, vacuolated change, or optically clear cytoplasmic change is present. |
| `hyaline-change-present` | Hyaline or glassy degenerative material is present. |
| `vascular-structure-present` | Vascular structure, sinusoid-like space, or blood-vessel context is present. |
| `ductular-portal-present` | Ductular reaction, bile duct/ductule, or portal tract context is present. |

This taxonomy is deliberately compact. It is designed to organize HCC morphology
at a robust tile-level semantic granularity, not to exhaustively encode all
pathology entities or clinical endpoints.

该 taxonomy 有意保持紧凑，目标是在稳健的 tile-level semantic granularity 上组织 HCC
morphology，而不是穷尽所有病理实体或临床终点。

## 5. Data Separation / 数据分离

The scientific separation is:

科学分离原则是：

- training side: prototype discovery, prototype annotation, representation
  training, and model selection;
- 训练侧：prototype discovery、prototype annotation、representation training 和 model
  selection；
- external or held-out side: final representation evaluation only.
- 外部或 held-out 侧：仅用于最终 representation evaluation。

The key issue is not only data leakage. The final evaluation objective must also
avoid being identical to the training prototype objective. Otherwise the study
can look like it defines its own target and then proves the model is best at that
same target.

关键问题不只是数据泄漏；最终评价目标也不能完全等同于训练 prototype objective。否则研究会
显得像是自己定义目标，再证明模型最适合这个目标。

## 6. Evaluation / 评价

Evaluation must separate teacher imitation from HCC representation quality.

评价必须区分 teacher imitation 和 HCC representation quality。

### Teacher-Imitation QC / Teacher imitation 质控

Teacher-alignment metrics are training QC. They verify that distillation is
working, but they do not prove HCC-specific representation value.

Teacher-alignment metrics 是训练质控。它们验证 distillation 是否正常，但不能证明
HCC-specific representation value。

Examples:

- feature cosine similarity;
- relation preservation;
- nearest-neighbor overlap;
- teacher agreement/disagreement diagnostics.

### Primary Evidence: Blinded Result-Level Morphology Retrieval

The primary scientific evaluation should be blinded result-level morphology
retrieval adjudication.

主要科学评价应是结果级盲态 morphology retrieval adjudication。

This is not dense labeling of all held-out tiles. Models first generate retrieval
results; experts then evaluate the returned results without knowing which model
produced them.

这不是对 held-out tiles 做 dense labeling。各模型先产生 retrieval results；专家随后在不知
模型来源的情况下评价返回结果。

Protocol:

1. Freeze a held-out query set and gallery pool.
   固定 held-out query set 和 gallery pool。
2. Run the same top-k retrieval for HCC-SemPath, each teacher embedding, and all
   ablations.
   对 HCC-SemPath、各 teacher embedding 和所有 ablation 使用同一 query/gallery 运行 top-k
   retrieval。
3. Merge and deduplicate query-result pairs or ranked lists across models.
   合并并去重所有模型产生的 query-result pairs 或 ranked lists。
4. Hide model identity and randomize review order.
   隐藏模型身份并随机化评审顺序。
5. Ask expert reviewers to score morphology relevance for each pair or ranked
   list.
   由专家对每个 pair 或 ranked list 的 morphology relevance 进行评分。
6. Report precision@k, mean relevance@k, NDCG@k, model win rate, and confidence
   intervals.
   报告 precision@k、mean relevance@k、NDCG@k、model win rate 和置信区间。

This directly tests whether the embedding retrieves HCC-morphology-relevant
neighbors under blinded expert judgment, without requiring tens of thousands of
manual tile labels and without reducing the evaluation to prototype
reconstruction.

该设计直接检验 embedding 是否能在专家盲评下检索到 HCC morphology 相关邻居，不需要数万张
tile 的人工标签，也不把评价降格为 prototype reconstruction。

### Secondary Diagnostics / 辅助诊断

Secondary diagnostics can support interpretation but should not replace the
blinded retrieval benchmark.

辅助诊断可用于解释结果，但不能替代盲态 retrieval benchmark。

- prototype response coverage and utilization;
- prototype response coverage 与 utilization；
- Level-1 / Level-2 prototype readout on supervised development or validation
  labels;
- 在有监督 development 或 validation labels 上的 Level-1 / Level-2 prototype readout；
- morphology-group clustering or neighborhood purity where independent labels
  are available;
- 在有独立标签处计算 morphology-group clustering 或 neighborhood purity；
- cross-cohort retrieval stability;
- 跨队列 retrieval stability；
- parameter count, memory, throughput, and WSI-level processing time.
- 参数量、显存、吞吐和 WSI 级处理时间。

Clinical endpoint prediction, MIL heads, survival, recurrence, or MVI prediction
are optional downstream studies. They are not the core validation of the HCC
foundation representation claim.

临床终点预测、MIL heads、生存、复发或 MVI 预测属于可选下游研究，不是 HCC foundation
representation 主张的核心验证。

## 7. Baselines / 基线

Required comparisons:

必要对照：

1. individual teacher embeddings: GigaPath, H-optimus-1, UNI2-h, Virchow2;
1. 单个 teacher embeddings：GigaPath、H-optimus-1、UNI2-h、Virchow2；
2. simple teacher fusion where scientifically meaningful;
2. 科学上有意义的简单 teacher fusion；
3. single-teacher student distillation;
3. 单 teacher student distillation；
4. multi-teacher distillation without HCC prototype supervision;
4. 无 HCC prototype supervision 的 multi-teacher distillation；
5. multi-teacher distillation without prototype-adjudicated reliability;
5. 无 prototype-adjudicated reliability 的 multi-teacher distillation；
6. HCC prototype supervision without multi-teacher distillation;
6. 无 multi-teacher distillation 的 HCC prototype supervision；
7. full HCC-SemPath / PAMT-D.
7. 完整 HCC-SemPath / PAMT-D。

All baselines must use the same held-out query/gallery and the same blinded
result-level adjudication protocol.

所有 baselines 必须使用同一 held-out query/gallery 和同一结果级盲态评价协议。

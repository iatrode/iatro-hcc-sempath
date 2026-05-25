# HCC-SemPath Prototype Taxonomy / Prototype 分类体系

This document defines the current HCC-SemPath prototype taxonomy. It describes
how tile-level prototype responses should be interpreted during training,
retrieval, and diagnostic review workflows.

本文档定义 HCC-SemPath 当前 prototype 分类体系，用于说明训练、检索和病理复核工作流中
tile-level prototype response 的解释规则。

## Classification Rule / 分类规则

Each tile receives one dominant level-1 prototype and zero or more level-2
attribute prototypes.

每个 tile 对应一个主导的 L1 prototype，并可同时对应零个或多个 L2 attribute
prototype。

L1 is mutually exclusive. It represents the dominant tissue or lesion pattern of
the tile. The model treats L1 as a primary-state competition and optimizes it
with softmax-style objectives.

L1 互斥，表示 tile 的主导组织或病变模式。模型将 L1 作为 primary-state competition，
使用 softmax 风格目标进行约束。

L2 is non-exclusive. It represents morphology, microenvironment, degeneration,
or interface attributes that may coexist within the same tile and may appear
across different L1 categories. The model treats L2 as multi-label attributes
and optimizes them with sigmoid/BCE-style or affinity-regression objectives.

L2 非互斥，表示可在同一 tile 内共存、也可跨 L1 类型出现的形态、微环境、退变或界面属性。
模型将 L2 作为 multi-label attribute，使用 sigmoid/BCE 风格或 affinity-regression
目标进行约束。

## L1: Mutually Exclusive Primary Prototypes / 互斥主 Prototype

| Prototype | Definition |
| --- | --- |
| `HCC-trabecular` | HCC dominated by trabecular, plate-like, or cord-like architecture. |
| `HCC-solid` | HCC dominated by compact solid, nested, or weakly structured growth. |
| `HCC-pseudoglandular` | HCC dominated by pseudoglandular, acinar-like, or lumen-like architecture. |
| `HCC-mixed-pattern` | Definite HCC with mixed structural patterns that cannot be stably assigned to one HCC architecture. |
| `Background-liver` | Non-neoplastic background liver parenchyma. |
| `Fibrous-stromal` | Fibrous septa, capsule, collagen, or stromal tissue-dominant tile. |
| `Degenerative-material` | Necrosis, hemorrhage, clot, bile lake, debris, or treatment-related degenerative material-dominant tile. |
| `Indeterminate-region` | Tissue-containing region with insufficient confidence for stable tissue or lesion assignment. |
| `Artifact-non-tissue` | Blank area, severe artifact, severe out-of-focus region, contamination, or non-tissue area. |

## L2: Non-Exclusive Attribute Prototypes / 非互斥属性 Prototype

| Prototype | Definition |
| --- | --- |
| `necrotic` | Obvious necrotic component. |
| `hemorrhagic-blood-rich` | Hemorrhage, blood pool, clot, or erythrocyte-rich area. |
| `bile-pigment-rich` | Obvious bile, bile pigment, or pigment deposition. |
| `inflammatory-rich` | Inflammatory-cell-rich area. |
| `fibrotic` | Obvious fibrosis, collagen, or scar-like component. |
| `steatotic-vacuolated` | Steatosis, vacuolated change, or optically clear cytoplasmic change. |
| `interface-capsule` | Capsule, boundary, or tumor-non-tumor interface area. |

## Package Encoding / Package 编码

The taxonomy is encoded in the prototype package metadata:

```python
names = [
    "HCC-trabecular",
    "HCC-solid",
    "HCC-pseudoglandular",
    "HCC-mixed-pattern",
    "Background-liver",
    "Fibrous-stromal",
    "Degenerative-material",
    "Indeterminate-region",
    "Artifact-non-tissue",
    "necrotic",
    "hemorrhagic-blood-rich",
    "bile-pigment-rich",
    "inflammatory-rich",
    "fibrotic",
    "steatotic-vacuolated",
    "interface-capsule",
]
levels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
exclusive = [True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False]
```

Recommended groups:

```python
groups = [
    "hcc_architecture",
    "hcc_architecture",
    "hcc_architecture",
    "hcc_architecture",
    "background_liver",
    "stroma",
    "degenerative_material",
    "indeterminate",
    "artifact",
    "degeneration",
    "hemorrhage",
    "pigment",
    "inflammation",
    "fibrosis",
    "cellular_change",
    "interface",
]
```

## Interpretation Notes / 判读说明

`HCC-mixed-pattern` should be used only when the tile is confidently HCC but
its dominant architecture is mixed or unstable across the L1 HCC architecture
classes.

`HCC-mixed-pattern` 仅用于明确为 HCC、但主导结构在多个 HCC architecture 类别之间混合或
不稳定的 tile。

`Indeterminate-region` is not the same as artifact. It should contain tissue
information, but the tissue or lesion assignment is uncertain.

`Indeterminate-region` 不等于 artifact。它应包含组织信息，只是组织或病变归属置信度不足。

L2 attributes can be active across L1 classes. For example, `inflammatory-rich`
can coexist with `HCC-trabecular`, `Background-liver`, or `Fibrous-stromal`.

L2 attribute 可跨 L1 类别出现。例如 `inflammatory-rich` 可以与 `HCC-trabecular`、
`Background-liver` 或 `Fibrous-stromal` 同时存在。

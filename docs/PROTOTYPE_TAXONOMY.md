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

| Prototype | Definition / 定义 |
| --- | --- |
| `HCC-trabecular` | HCC dominated by trabecular, plate-like, or cord-like architecture. / 以梁索状、板状或条索状排列为主的 HCC。 |
| `HCC-solid` | HCC dominated by compact solid, nested, or weakly structured growth. / 以致密实性、巢状或结构感弱的生长方式为主的 HCC。 |
| `HCC-pseudoglandular` | HCC dominated by pseudoglandular, acinar-like, or lumen-like architecture. / 以假腺样、腺泡样或腔样结构为主的 HCC。 |
| `HCC-mixed-pattern` | Definite HCC with mixed structural patterns that cannot be stably assigned to one HCC architecture. Use only when the tile is confidently HCC but the dominant architecture is mixed or unstable across L1 HCC architecture classes. / 明确为 HCC，但结构模式混合，不能稳定归入单一 HCC 结构型；仅用于 HCC 置信度明确但主导结构在多个 HCC architecture 类别之间混合或不稳定的 tile。 |
| `Background-liver` | Non-neoplastic background liver parenchyma. / 非肿瘤性背景肝实质。 |
| `Fibrous-stromal` | Fibrous septa, capsule, collagen, or stromal tissue-dominant tile. / 以纤维隔、包膜、胶原或间质组织为主的 tile。 |
| `Degenerative-material` | Necrosis, hemorrhage, clot, bile lake, debris, or treatment-related degenerative material-dominant tile. / 以坏死、出血、血凝块、胆汁湖、碎屑或治疗后退变物为主的 tile。 |
| `Indeterminate-region` | Tissue-containing region with insufficient confidence for stable tissue or lesion assignment. This is not artifact; tissue information is present but category confidence is insufficient. / 有组织信息，但区域归属或病变判读置信度不足；不同于 artifact，该类应包含可见组织信息，只是类别归属不稳定。 |
| `Artifact-non-tissue` | Blank area, severe artifact, severe out-of-focus region, contamination, or non-tissue area. / 空白、严重伪影、严重失焦、污染或非组织区域。 |

## L2: Non-Exclusive Attribute Prototypes / 非互斥属性 Prototype

| Prototype | Definition / 定义 |
| --- | --- |
| `necrotic` | Obvious necrotic component. / 坏死成分明显。 |
| `hemorrhagic-blood-rich` | Hemorrhage, blood pool, clot, or erythrocyte-rich area. / 出血、血池、血凝块或红细胞丰富。 |
| `bile-pigment-rich` | Obvious bile, bile pigment, or pigment deposition. / 胆汁、胆色素或色素沉积明显。 |
| `inflammatory-rich` | Inflammatory-cell-rich area; can coexist with HCC, background liver, or stromal L1 classes. / 炎症细胞丰富；可与 HCC、背景肝或纤维间质等 L1 类别同时存在。 |
| `fibrotic` | Obvious fibrosis, collagen, or scar-like component. / 纤维化、胶原或瘢痕样成分明显。 |
| `steatotic-vacuolated` | Steatosis, vacuolated change, or optically clear cytoplasmic change. / 脂肪变、空泡样改变或胞质空亮改变。 |
| `interface-capsule` | Capsule, boundary, or tumor-non-tumor interface area. / 包膜、边界或肿瘤-非肿瘤交界区域。 |

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

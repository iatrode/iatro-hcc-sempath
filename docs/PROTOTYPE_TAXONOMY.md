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
| `HCC-tumor` | HCC tumor-dominant tile, regardless of local architecture subtype. / 以 HCC 肿瘤成分为主的 tile，不再细分局部结构亚型。 |
| `Background-liver` | Non-neoplastic background liver parenchyma. / 非肿瘤性背景肝实质。 |
| `Inflammatory-stromal` | Stromal, portal, fibrous, or interface-rich tile with inflammatory/stromal context as the dominant state. / 以间质、汇管区、纤维成分或界面区域为主，并带炎症/间质背景的 tile。 |
| `Degenerative-material` | Necrosis, hemorrhage, clot, bile lake, debris, or treatment-related degenerative material-dominant tile. / 以坏死、出血、血凝块、胆汁湖、碎屑或治疗后退变物为主的 tile。 |

## L2: Non-Exclusive Attribute Prototypes / 非互斥属性 Prototype

| Prototype | Definition / 定义 |
| --- | --- |
| `hepatocellular-parenchyma-present` | Hepatocellular parenchyma or hepatocyte-like tumor/background cells are present. / 可见肝细胞性实质或肝细胞样肿瘤/背景细胞。 |
| `necrosis-present` | Necrotic component is present. / 可见坏死成分。 |
| `hemorrhage-present` | Hemorrhage, blood pool, clot, or erythrocyte-rich area is present. / 可见出血、血池、血凝块或红细胞丰富区域。 |
| `bile-pigment-present` | Bile, bile pigment, or pigment deposition is present. / 可见胆汁、胆色素或色素沉积。 |
| `inflammatory-cell-present` | Inflammatory cells are present. / 可见炎症细胞。 |
| `fibrous-stroma-present` | Fibrous stroma, collagen, scar-like matrix, septa, or capsule-like stroma is present. / 可见纤维间质、胶原、瘢痕样基质、纤维隔或包膜样间质。 |
| `steatosis-vacuolation-present` | Steatosis, vacuolated change, or optically clear cytoplasmic change is present. / 可见脂肪变、空泡样改变或胞质空亮改变。 |
| `hyaline-change-present` | Hyaline or glassy degenerative material is present. / 可见透明变或玻璃样退变物。 |
| `vascular-structure-present` | Vascular structure, sinusoid-like space, or blood vessel context is present. / 可见血管结构、窦样腔隙或血管相关背景。 |
| `ductular-portal-present` | Ductular reaction, bile duct/ductule, or portal tract context is present. / 可见胆管/小胆管、胆管反应或汇管区背景。 |

## Package Encoding / Package 编码

The taxonomy is encoded in the prototype package metadata:

```python
names = [
    "HCC-tumor",
    "Background-liver",
    "Inflammatory-stromal",
    "Degenerative-material",
    "hepatocellular-parenchyma-present",
    "necrosis-present",
    "hemorrhage-present",
    "bile-pigment-present",
    "inflammatory-cell-present",
    "fibrous-stroma-present",
    "steatosis-vacuolation-present",
    "hyaline-change-present",
    "vascular-structure-present",
    "ductular-portal-present",
]
levels = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
exclusive = [True, True, True, True, False, False, False, False, False, False, False, False, False, False]
```

Recommended groups:

```python
groups = [
    "hcc_tumor",
    "background_liver",
    "inflammatory_stroma",
    "degenerative_material",
    "hepatocellular_parenchyma",
    "necrosis",
    "hemorrhage",
    "pigment",
    "inflammation",
    "fibrous_stroma",
    "steatosis_vacuolation",
    "hyaline_change",
    "vascular_structure",
    "ductular_portal",
]
```

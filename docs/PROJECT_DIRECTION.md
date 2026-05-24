# HCC-SemPath Project Direction / 项目方向

This document is the public-facing project direction for HCC-SemPath. It keeps
the repository centered on one goal: a lightweight vertical pathology
representation model for hepatocellular carcinoma.

本文档是 HCC-SemPath 面向公开发布的项目方向说明。仓库主线应始终围绕一个目标：
构建面向肝细胞癌的轻量级垂直病理表征模型。

## 1. Research Question / 研究问题

HCC-SemPath asks whether a compact student model can learn a reusable
HCC-specific embedding space that is more useful for hepatocellular carcinoma
computational pathology than directly using general pathology foundation model
embeddings.

HCC-SemPath 关注的问题是：轻量 student 能否学习一个可复用的 HCC 专病 embedding
space，使其在肝细胞癌计算病理任务中比直接使用通用病理基础模型 embedding 更有效。

## 2. Core Claim / 核心主张

HCC-SemPath should be presented as a lightweight vertical representation model,
not as a general-purpose pathology foundation model and not as a single
diagnostic, prognostic, captioning, or VQA model.

HCC-SemPath 应被定位为轻量级垂直表征模型，而不是通用病理基础模型，也不是单一诊断、
预后、图像描述或视觉问答模型。

The released representation is the shared HCC embedding `z_hcc`. Teacher
alignment heads are training-time adapters.

最终复用和发布的表征是共享 HCC embedding `z_hcc`。Teacher alignment heads 只是训练阶段的适配器。

## 3. Core Hypothesis / 核心假设

General pathology foundation models contain strong morphology priors, but their
embedding spaces are not optimized around HCC-specific histopathology semantics.
A lightweight student can integrate heterogeneous teacher priors and then be
reshaped by HCC-specific weak supervision into a compact disease-oriented
embedding space.

通用病理基础模型包含强形态学先验，但其 embedding space 并不是围绕 HCC 专病组织病理
语义优化。轻量 student 可以整合异质 teacher 先验，并通过 HCC 专病弱监督进一步重塑为
紧凑的 disease-oriented embedding space。

## 4. Model Strategy / 模型思路

The main architecture is:

```text
tile image
  -> lightweight shared student encoder
  -> z_hcc
       -> teacher-specific head A
       -> teacher-specific head B
       -> teacher-specific head C
```

Stage 1 uses multi-teacher distillation to inject general pathology morphology
priors. Stage 2 uses HCC-specific weak supervision to shape `z_hcc` around HCC
morphology and tissue semantics.

第一阶段使用多教师蒸馏注入通用病理形态先验。第二阶段使用 HCC 专病弱监督塑造
`z_hcc`，使其围绕 HCC 形态和组织语义组织。

The weak supervision signals should act on the shared embedding space rather
than only making the student imitate teacher features more closely. Suitable
public-safe signal types include expert-defined morphology anchors, weak region
labels, curated retrieval sets, structured pathology descriptors, and
slide- or region-level disease-domain labels.

弱监督信号应直接作用于共享 embedding space，而不是只让 student 更接近 teacher
feature。适合公开描述的信号包括专家定义形态锚点、弱区域标签、人工整理检索集合、
结构化病理描述，以及切片级或区域级专病标签。

## 5. Data Organization / 数据组织

Training data should be organized around patient- or WSI-level splits. Tile
packages, teacher feature packages, weak labels, and semantic anchors should be
joined by stable tile identifiers and reproducible WSI coordinates.

训练数据应按 patient 或 WSI 级别划分。Tile package、teacher feature package、弱标签和
semantic anchors 通过稳定 tile identifier 与可复现 WSI 坐标对齐。

The public repository should contain schemas, configuration templates, synthetic
fixtures, aggregate benchmark summaries, and reproducibility scripts. It should
not contain private WSIs, production tile packages, teacher feature packages,
large checkpoints, patient-identifiable manifests, or institutional file paths.

公开仓库应包含 schema、配置模板、合成测试数据、汇总 benchmark 结果和复现脚本。不应包含
私有 WSI、生产级 tile package、teacher feature package、大型 checkpoint、可识别患者身份的
manifest 或机构内部路径。

## 6. Validation Design / 验证设计

Evaluation must separate teacher imitation from HCC representation value.

评估必须区分 teacher imitation 与 HCC representation value。

Teacher-alignment metrics are quality-control metrics:

- feature cosine similarity;
- normalized feature MSE;
- pairwise relation preservation;
- nearest-neighbor overlap;
- teacher-specific retrieval consistency.

HCC representation metrics are the main scientific evidence:

- expert-reviewed HCC morphology retrieval;
- HCC semantic anchor consistency;
- morphology-group clustering or neighborhood purity;
- cross-cohort stability;
- lightweight downstream adaptation;
- parameter count, memory, throughput, and WSI-level processing time.

Required baselines are individual teacher embeddings, single-teacher
distillation, multi-teacher distillation without HCC weak supervision, HCC weak
supervision without multi-teacher distillation, and the full HCC-SemPath model.

必要对照包括单个 teacher embedding、单教师蒸馏、无 HCC 弱监督的多教师蒸馏、无多教师蒸馏的
HCC 弱监督，以及完整 HCC-SemPath 模型。

## 7. Expected Contribution / 预期贡献

The contribution is a public-safe framework for learning a lightweight vertical
HCC pathology representation: heterogeneous pathology teachers provide
morphology priors, HCC weak supervision reshapes the shared embedding, and the
resulting compact model is evaluated as a reusable representation for HCC
computational pathology workflows.

预期贡献是一套可公开发布的轻量级 HCC 垂直病理表征框架：异质病理 teacher 提供形态学
先验，HCC 弱监督重塑共享 embedding，最终得到一个紧凑模型，并作为 HCC 计算病理工作流中的
可复用表征进行评估。

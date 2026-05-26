# HCC-SemPath Current Status / 当前状态

This document summarizes the current public-facing technical direction of HCC-SemPath. It is intended to support future open-source release, reproducible experimentation, and manuscript planning.

本文档总结 HCC-SemPath 当前面向公开发布的技术方向，用于支持后续开源、可复现实验和论文规划。

## 1. Project Positioning / 项目定位

HCC-SemPath is being developed as a lightweight vertical pathology representation model for hepatocellular carcinoma (HCC). The primary output is a compact reusable embedding space for HCC histopathology, not a general-purpose pathology foundation model and not a single diagnostic, prognostic, captioning, or visual question answering model.

HCC-SemPath 的目标是构建面向肝细胞癌（HCC）的轻量级垂直病理表征模型。模型的主要输出是紧凑、可复用的 HCC 组织病理 embedding space，而不是通用病理基础模型，也不是单一诊断、预后、图像描述或视觉问答模型。

The project is positioned as a lightweight vertical representation effort: it aims to define and learn an HCC-oriented semantic space that can support retrieval, clustering, weakly supervised learning, downstream adaptation, and computational pathology workflows with a compact student model.

本项目定位为轻量级垂直表征工作：目标是通过紧凑 student 定义并学习一个 HCC 专病语义空间，用于支持检索、聚类、弱监督学习、下游适配和计算病理工作流。

## 2. Technical Direction / 技术路线

### 2.1 Multi-teacher distillation as morphology prior / 多教师蒸馏作为形态先验

The training framework is designed to use multiple pathology foundation models as teachers. These teachers may encode different objectives, spatial contexts, and morphology priors. Their feature spaces should therefore not be directly averaged or treated as a single homogeneous target.

训练框架计划使用多个病理基础模型作为 teacher。这些 teacher 可能具有不同的训练目标、空间上下文建模方式和形态学先验。因此，不应直接平均它们的 feature space，也不应把它们当作单一同质监督目标。

The planned architecture uses one shared student encoder and teacher-specific projection heads during training:

计划中的结构是在训练阶段使用一个共享 student encoder 和多个 teacher-specific projection heads：

```text
image tile
  -> shared student encoder
  -> z_hcc
       -> head_gigapath  -> GigaPath feature space
       -> head_uni2_h    -> UNI2-h feature space
       -> head_virchow2  -> Virchow2 feature space
```

The teacher-specific heads are training-time alignment modules. The reusable output of the model is the shared HCC embedding `z_hcc`.

teacher-specific heads 是训练阶段的对齐模块。模型最终复用的是共享 HCC embedding，即 `z_hcc`。

### 2.2 HCC-specific weak supervision / HCC 专病弱监督

Multi-teacher distillation is not the final objective. It provides generic morphology priors. The final representation should be reshaped by HCC-specific weak supervision so that the embedding geometry better reflects HCC histopathology.

多教师蒸馏不是最终目标，而是提供通用形态学先验。最终表征需要通过 HCC 专病弱监督进一步重塑，使 embedding geometry 更符合 HCC 组织病理语义。

Potential weak supervision sources include region-level morphology signals, slide-level labels, structured pathology descriptions, expert-defined semantic prototypes, weak region labels, or other HCC-specific signals. The repository direction now treats prototype as the only public term for this mechanism.

潜在弱监督来源包括区域级形态信号、切片级标签、结构化病理描述、专家定义语义 prototype、弱区域标签，以及其他 HCC 专病信号。仓库公开表述统一使用 prototype 描述该机制。

The working hypothesis is that a compact student can outperform a general teacher on HCC-specific representation tasks when the student is trained to organize HCC-relevant morphology, rather than simply imitate generic teacher embeddings.

当前核心假设是：当 student 学习组织 HCC 相关形态语义，而不是仅仅模仿通用 teacher embedding 时，轻量 student 有机会在 HCC 专病表征任务上超过通用 teacher。

## 3. Boundary With Related Work / 与相关工作的边界

HCC-SemPath should not be presented as the first multi-teacher distillation method, the first pathology foundation model compression method, or a conventional HCC diagnostic model.

HCC-SemPath 不应被表述为首个多教师蒸馏方法、首个病理基础模型压缩方法，或一个常规 HCC 诊断模型。

The intended contribution is a lightweight disease-specific representation framework:

预期贡献是一个轻量级专病表征框架：

- heterogeneous pathology foundation teachers are used as morphology priors;
- 异质病理基础模型 teacher 被用作形态学先验；
- a shared HCC embedding space is learned;
- 学习一个共享 HCC embedding space；
- HCC-specific weak supervision reshapes the representation beyond teacher imitation;
- HCC 专病弱监督将表征从 teacher imitation 推向专病语义空间；
- the resulting embedding is evaluated as a reusable representation for HCC computational pathology.
- 最终 embedding 作为 HCC 计算病理中的可复用表征进行评估。

Known adjacent directions include multi-teacher knowledge distillation, pathology foundation model distillation/compression, general pathology foundation models, and HCC-oriented pathology MLLMs. These should be cited as related work while maintaining a clear distinction between diagnostic/VQA models and reusable embedding foundation models.

相近方向包括多教师知识蒸馏、病理基础模型蒸馏/压缩、通用病理基础模型，以及 HCC-oriented pathology MLLM。这些工作应作为相关工作引用，同时明确区分诊断/问答模型与可复用 embedding 基座模型。

## 4. Current Model Scale / 当前模型规模

The current student configuration is:

当前 student 配置为：

```yaml
backbone_name: vit_small_patch14_reg4_dinov2.lvd142m
teacher_dim: 1536
pretrained: true
```

The `reg4` backbone is the DINOv2 ViT-S/14 variant with four register tokens.

`reg4` backbone 是带 4 个 register token 的 DINOv2 ViT-S/14 变体。

Estimated trainable parameters:

估计可训练参数量：

- ViT-S/14 DINOv2 backbone: approximately 21M parameters.
- ViT-S/14 DINOv2 backbone：约 21M 参数。
- Projection head from 384 to 1536 dimensions: approximately 0.592M parameters.
- 384 到 1536 维 projection head：约 0.592M 参数。
- Total: approximately 21.6M trainable parameters.
- 合计：约 21.6M 可训练参数。

The `lvd142m` suffix refers to the DINOv2 pretraining data scale and should not be interpreted as the model parameter count.

`lvd142m` 后缀表示 DINOv2 预训练数据规模，不应被理解为模型参数量。

## 5. Planned Data Scale / 计划数据规模

The current planning scale is:

当前计划规模为：

```text
900 WSIs x approximately 20,000 tiles per WSI ~= 18,000,000 tiles
```

Approximate storage planning:

粗略存储规划：

- compressed image tiles: approximately 230 GB at about 13 KB per tile;
- 压缩图像 tile：按约 13 KB/tile 估计，约 230 GB；
- teacher features: approximately 110 GB for 1536-dimensional float32 features;
- teacher feature：1536 维 float32 约 110 GB；
- total training inputs: approximately 340 GB or more;
- 训练输入合计：约 340 GB 或更高；
- recommended working storage: at least 500 GB after indexes, outputs, checkpoints, and intermediate files.
- 考虑索引、输出、checkpoint 和中间文件后，建议至少预留 500 GB。

Approximate training steps:

粗略训练步数：

```text
batch 128: 140,625 steps / epoch
batch 256: 70,313 steps / epoch
batch 512: 35,157 steps / epoch
```

Full-scale training should be planned around short runs first, such as one full epoch for throughput and convergence checks, followed by 3 to 5 epoch training plans when the data pipeline is stable.

全量训练应先按短程运行规划，例如先完成 1 个 full epoch 以检查吞吐和收敛，再在数据管线稳定后规划 3 到 5 个 epoch。

## 6. Compute Notes / 计算资源判断

Apple M-series local machines are suitable for workflow validation, small-scale training, data contract checks, and debugging. They are not recommended for full-scale training over approximately 18M tiles.

Apple M 系列本机适合流程验证、小规模训练、数据合同检查和调试，不建议用于约 18M tile 的正式全量训练。

A V100 32GB GPU should be able to train the current student model, but throughput will depend heavily on mixed precision, batch size, storage bandwidth, tile decoding speed, and validation strategy.

V100 32GB 应能训练当前 student 模型，但实际吞吐高度依赖混合精度、batch size、存储带宽、tile 解码速度和验证策略。

Implemented training-system foundations:

已实现的训练系统基础：

- shared HCC embedding with teacher-specific projection heads;
- 共享 HCC embedding 与 teacher-specific projection heads；
- multi-teacher distillation losses and teacher-specific metrics;
- 多 teacher 蒸馏损失与 teacher-specific 指标；
- CUDA mixed precision training with autocast and gradient scaling;
- CUDA 混合精度训练，包括 autocast 与 gradient scaling；
- per-WSI multi-package dataset loading without changing the production image-tile IAC format;
- 在不改变生产 image-tile IAC 格式的前提下支持 per-WSI multi-package dataset；
- training manifest construction for development sources plus a public source with held-out external validation;
- 支持 development source 与 public source held-out external validation 的 training manifest 构建；
- convention-based teacher feature package resolution from manifest WSI stems.
- 基于 manifest WSI stem 与命名约定解析 teacher feature package。
- training preflight checks that reject stale or removed teacher entries, including H-optimus-1/H1.
- 训练前配置检查会拒绝残留或已移除的 teacher 条目，包括 H-optimus-1/H1。

Important engineering requirements before full-scale training:

正式全量训练前的重要工程要求：

- use fixed or sampled validation subsets for frequent validation;
- 高频验证时使用固定或抽样 validation subset；
- add training-time WSI-window or feature-cache-aware sampling if measured I/O locality becomes the bottleneck;
- 若实测 I/O locality 成为瓶颈，再加入训练期 WSI-window 或 feature-cache-aware sampling；
- add HCC-specific representation evaluation beyond teacher imitation before manuscript-grade conclusions;
- 在论文级结论前补齐超越 teacher imitation 的 HCC 专病表征评估；
- benchmark teacher-feature storage alternatives on real extracted features before changing the feature package layout.
- 在修改 feature package layout 前，基于真实提取特征评估不同 teacher-feature 存储方案。

## 7. Repository Data Policy / 仓库数据策略

The repository should contain public-facing specifications, code, configuration templates, schemas, small fixtures, and non-sensitive aggregate summaries.

仓库应保存面向公开发布的规范、代码、配置模板、schema、小型测试数据和非敏感汇总统计。

Recommended to include in git:

建议进入 git 的内容：

- project positioning and design documents;
- 项目定位和设计文档；
- training and evaluation configuration templates;
- 训练和评估配置模板；
- prototype package format documentation;
- prototype package 格式文档；
- manifest schemas;
- manifest schema；
- teacher metadata schemas;
- teacher 元数据 schema；
- HCC semantic prototype schemas;
- HCC 语义 prototype schema；
- small smoke-test fixtures;
- 小型 smoke-test fixtures；
- benchmark summary tables without protected health information;
- 不含受保护健康信息的 benchmark 汇总表；
- reproducibility scripts and documentation.
- 可复现脚本和文档。

Should not be committed to normal git history:

不应进入普通 git 历史的内容：

- raw WSIs;
- 原始 WSI；
- large tile packages;
- 大规模 tile package；
- large teacher feature packages;
- 大规模 teacher feature package；
- per-tile large tables at production scale;
- 生产规模的每 tile 大表；
- large checkpoints;
- 大型 checkpoint；
- manifests containing patient-identifiable information or real clinical file paths.
- 包含可识别患者信息或真实临床文件路径的 manifest。

Large artifacts should be managed through an external storage system or artifact registry, with only schemas, checksums, public summaries, and reproduction scripts tracked in the repository.

大型 artifact 应通过外部存储或 artifact registry 管理，仓库中只追踪 schema、checksum、公开汇总和复现脚本。

## 8. Near-term Roadmap / 近期路线图

Near-term development should focus on:

近期开发重点：

1. use the training manifest schema and cohort-building workflow for real dry runs;
1. 使用 training manifest schema 与 cohort 构建流程进行真实 dry run；
2. fixed or sampled validation subset export for frequent validation;
2. 导出固定或抽样 validation subset，用于高频验证；
3. representation evaluation protocols beyond teacher imitation;
3. 建立超越 teacher imitation 的表征评估协议；
4. benchmark teacher-feature storage alternatives after real feature extraction;
4. 在真实 feature 提取后评估 teacher-feature 存储方案；
5. bilingual open-source documentation for public release.
5. 为未来公开发布准备中英双语文档。

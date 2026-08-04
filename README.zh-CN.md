# HCC-SemPath

[![Source](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/iatrode/iatro-hcc-sempath) [![Hugging Face](<https://img.shields.io/badge/Hugging%20Face-gated%20model-ffcc4d?logo=huggingface&logoColor=black>)](https://huggingface.co/iatrode/iatro-hcc-sempath) [![ModelScope](<https://img.shields.io/badge/ModelScope-gated%20model-624aff>)](https://modelscope.cn/models/iatrode/iatro-hcc-sempath) [![PyPI](https://img.shields.io/pypi/v/hcc-sempath?include_prereleases)](https://pypi.org/project/hcc-sempath/) [![Python](https://img.shields.io/pypi/pyversions/hcc-sempath)](https://pypi.org/project/hcc-sempath/) [![CI](https://github.com/iatrode/iatro-hcc-sempath/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/iatrode/iatro-hcc-sempath/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

**Gated 模型权重：** [Hugging Face](https://huggingface.co/iatrode/iatro-hcc-sempath) | [ModelScope](https://modelscope.cn/models/iatrode/iatro-hcc-sempath)。
两个模型仓库分发同一份 SemPath 模型资产。PyPI 包仅包含建模代码和 CLI；临床资产、
教师权重及患者级输出均不公开分发。

HCC-SemPath 是一个 HCC 专用病理表征模型。它将四个冻结的病理基础模型蒸馏到一个
DINOv2-S/14 学生模型中，并使用一个规模小、固定且由病理医师标注的原型库约束所得
表征。公开包提供的是一套完整的建模工具链：构建图像 tile 与教师特征资产、支持分类
和空间标注、构建固定监督库、训练和评估学生模型，以及导出能够还原到原始切片坐标的
tile 级预测结果。

模型包含两条并行的监督输出：

- 七分类 HCC 组织/分化原型读出；
- 由 point、circle 和 brush 注释约束的十一成分空间读出。

科学设计与实现口径以
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md) 为准。论文实验协议以及
公开资产与受控资产的边界见 [`experiments/README.md`](experiments/README.md)。

> **仅限研究用途。** HCC-SemPath 不是诊断设备；未经独立验证和相应机构审批，不得
> 用于临床决策。

## 目录

- [科学契约](#科学契约)
- [模型契约](#模型契约)
- [发布与数据边界](#发布与数据边界)
- [安装](#安装)
- [命令结构](#命令结构)
- [快速开始：使用发布模型推理](#快速开始使用发布模型推理)
- [完整建模流程](#完整建模流程)
- [标注语义](#标注语义)
- [训练与检查点选择](#训练与检查点选择)
- [预测格式与空间还原](#预测格式与空间还原)
- [配置文件](#配置文件)
- [测试](#测试)
- [仓库结构](#仓库结构)
- [许可证](#许可证)

## 科学契约

HCC-SemPath 验证的核心假设是：当多教师蒸馏与一个覆盖目标分类和空间特征范围的小型
专家监督库相结合时，一个紧凑的 HCC 专用学生模型能够保留多个大型病理教师中互补的
形态学信息。

该契约由五个固定部分组成。

1. **四个冻结教师坐标系。** GigaPath、H-optimus-1、UNI2-h 和 Virchow2 的特征在
   训练前离线缓存。SemPath 训练不更新教师权重或教师特征提取器。
2. **一个共享学生模型。** DINOv2-S/14 生成共同的 HCC 表征。在约 20 倍镜、
   0.5 微米/像素条件下，一个 14 像素 patch 横跨约 7 微米，使学生 token 的空间尺度
   接近一个免疫细胞的直径。
3. **两个相互独立的监督轴。** 分类原型描述整体 HCC 分化/组织身份；空间原型描述局部
   生物学成分。两者不能相互替代。
4. **固定完整库原型。** PAMT-D 原型是对完整、冻结专家库计算得到的当前学生表征精确
   质心，不是 minibatch 指数移动平均。
5. **独立的检查点监督。** 验证集专家库用于选择检查点，不加入训练优化器，也不是
   教师蒸馏训练数据。

训练分类库用于证明小样本专家标注已达到所需的特征空间覆盖范围。验证监督库则为持续
动态刷新的蒸馏轨迹提供外部监督停止信号。因此，私有研究数据不应与操作这些数据的
公开代码路径混为一谈。

## 模型契约

### 分类类别

整体分类读出包含七个互斥类别：

1. `HCC-tumor-well-differentiated`：高分化 HCC；
2. `HCC-tumor-moderately-differentiated`：中分化 HCC；
3. `HCC-tumor-poorly-differentiated`：低分化 HCC；
4. `Background-liver`：背景肝组织；
5. `Inflammatory-stromal`：炎性间质；
6. `Hemorrhage-necrosis`：出血/坏死；
7. `Artifact-contamination`：人工污染/制片伪影。

### 空间成分

空间读出包含十一个可同时出现的非互斥成分：

1. `hepatocellular-parenchyma`：肝细胞/肝实质；
2. `necrosis`：坏死；
3. `hemorrhage`：出血；
4. `bile-pigment`：胆色素/淤胆；
5. `inflammatory-cell`：炎症细胞；
6. `fibroblast`：成纤维细胞；
7. `fibrous-stroma`：纤维间质；
8. `steatosis-vacuolation`：脂肪变/空泡；
9. `small-vessel`：小血管；
10. `large-vessel`：大血管；
11. `ductular-portal`：胆管/汇管区结构。

`fibroblast` 是细胞定位与密度目标；`fibrous-stroma` 是连续细胞外区域目标。小血管与
大血管必须分开，因为后者的身份和范围依赖跨多个空间网格的结构上下文。

### 训练阶段

训练从仅教师表征对齐开始。到达配置的 global-step ramp 后，分类和空间监督同时进入。
实现不会把验证监督悄悄加入优化器 loss；验证读出只在前向计算后参与检查点选择。

## 发布与数据边界

仓库明确分离可复用建模代码与受控研究资产。

| 资产 | 公开源码仓库 | 由使用方提供 |
|---|---:|---:|
| CLI、模型、loss、训练引擎、预测读取器 | 是 | 否 |
| 示例配置与格式契约 | 是 | 否 |
| 标注 UI 与监督资产构建器 | 是 | 否 |
| 论文特定的协议脚本和配置 | 是，位于 `experiments/` | 否 |
| 诊断性 WSI 与派生图像 tile | 否 | 是 |
| 患者/病例标识与标注状态 | 否 | 是 |
| 四教师权重及 gated 访问权限 | 否 | 是 |
| 教师特征 IAC 缓存 | 否 | 是 |
| 冻结的训练/验证监督记录 | 否 | 是 |
| 学生检查点与发布模型包 | 单独 gated 发布 | 是 |

标注状态可能包含本地路径或病例标识，必须保存在 Git 忽略的本地工作区或机构控制的
私有仓库中。HCC-SemPath 不下载或再分发四个教师模型；使用方必须分别向原始提供方
申请访问，并遵守各模型的许可条款。

## 安装

HCC-SemPath 支持 Python 3.10 及以上版本。通过 PyPI 安装最新公开预览版：

```bash
python -m pip install --upgrade pip
python -m pip install --pre hcc-sempath
hcc-sempath --help
```

从源码进行开发时，在源码检出目录安装 editable 包及完整开发工具链：

```bash
python -m pip install -e ".[dev]"
```

`pyproject.toml` 将两个 IatroCache 依赖约束在兼容的公开版本系列内。应整体安装项目
解析出的依赖，不应单独覆盖其中任一包的版本。

WSI 读取要求 OpenSlide 在当前 Python 环境中可用。GPU 训练还需要与宿主机匹配的
PyTorch/CUDA 组合；HCC-SemPath 不会替使用方创建或选择 CUDA 环境。

## 命令结构

安装后 CLI 提供八个稳定的顶层工作流：

```text
hcc-sempath build       构建可复用 tile、特征、manifest 与监督资产
hcc-sempath download    将 gated 发布模型下载到本地模型缓存
hcc-sempath annotate    启动分类/空间联合标注工作区
hcc-sempath train       训练或恢复 SemPath 学生模型
hcc-sempath evaluate    按解析后的配置评估检查点
hcc-sempath export      构建仅含推理能力的发布模型包
hcc-sempath infer       导出可还原到切片坐标的 tile 级预测
hcc-sempath benchmark   测试发布模型的推理性能
```

`build` 命名空间包括：

```text
hcc-sempath build tiles
hcc-sempath build teacher-features
hcc-sempath build training-cache
hcc-sempath build manifest
hcc-sempath build supervision
```

每个命令的权威参数以自身 `--help` 输出为准：

```bash
hcc-sempath build tiles --help
hcc-sempath build teacher-features --help
hcc-sempath build manifest --help
hcc-sempath annotate --help
hcc-sempath train --help
hcc-sempath infer --help
```

## 快速开始：使用发布模型推理

Gated 访问获批后只需下载一次发布模型。命令在中国公网 IP 下选择 ModelScope，其他
地区选择 Hugging Face；也可通过 `--hub` 显式指定。

```bash
hcc-sempath download

hcc-sempath infer \
  --input /path/to/case.svs \
  --output /path/to/predictions
```

`infer` 从本地模型缓存解析已下载的发布模型。其他位置的发布模型可用
`--model` 选择；`--hub {hf,modelscope}` 与 `--cache-dir` 用于选择特定本地缓存。

输入可以是：

- `<name>.tile.path.iac`；
- 单张 224x224 PNG、JPEG、WebP、BMP 图像；
- 单个 WSI 或含 `.svs`、`.mrxs`、`.ndpi`、`.scn`、`.tif`、`.tiff` 的目录。

WSI 会先按 `--target-mpp` 降采样，依次经过低分辨率组织 mask 和 tile 级组织过滤，并持久化为
`<name>.tile.path.iac`；模型结果写为 `<name>.pred.path.iac`。WSI 转换和模型推理两个
阶段默认都显示进度。

若需要避免额外的整数离散化，使用 float16；若更关注文件体积，可使用带明确范围的
整数编码：

```bash
hcc-sempath infer \
  --input /path/to/case.mrxs \
  --output /path/to/predictions \
  --target-mpp 0.5 \
  --min-tissue-fraction 0.10 \
  --spatial-dtype float16 \
  --batch-size 128 \
  --workers 8
```

WSI 无法读取可信 MPP 时使用 `--native-mpp`/`--native-mpp-y`。默认不会覆盖已有 tile
或预测输出；只有显式提供 `--overwrite` 才会替换。`--no-progress` 用于关闭进度条。

## 完整建模流程

以下是公开工具链支持的完整流程。所有路径均为占位符；仓库不附带任何受控研究资产。

### 1. 构建无损图像 tile 包

输入可以是单个 WSI 或 WSI 目录。MPP 必须能从切片元数据读取，或者由参数显式提供。

```bash
hcc-sempath build tiles \
  --input /controlled/wsis \
  --output /assets/tiles \
  --target-mpp 0.5 \
  --tile-size 224 \
  --min-tissue-fraction 0.30 \
  --workers 8 \
  --lossless
```

关键输入：

- `--patient-id`、`--slide-id`、`--split` 写入稳定的研究身份；
- `--native-mpp`/`--native-mpp-y` 用于缺少可信 MPP 的切片；
- `--max-tiles` 和 `--limit` 只用于有意设置边界的试运行；
- `--tcga-patient-id` 启用受支持的 TCGA 患者编号解析；
- 只有 `--overwrite` 才允许替换已有包。

输出：

- 每张切片/每个源文件一个 `<name>.tile.path.iac`；
- 目录模式下生成 `packages.csv`、`batch_summary.json` 和
  `batch_progress.json`；
- 开启 `--qc` 时生成可选的单切片 QC 图。

验收条件：

- IAC record 数等于保留的 tile 数；
- patient/slide 身份以及 level-0 `x`/`y` 坐标存在；
- MPP、tile 大小与源切片层级几何关系一致；
- 在提取教师特征前确认组织过滤效果和 QC 抽样。

### 2. 构建四教师合并特征包

公开构建器固定运行 GigaPath、H-optimus-1、UNI2-h 和 Virchow2，核对四者的逐行
对应关系，并为每个 tile 包写出一个合并特征包。

```bash
hcc-sempath build teacher-features \
  --input /assets/tiles \
  --output /assets/features \
  --device cuda \
  --precision bf16 \
  --batch-size 128 \
  --num-workers 8 \
  --validate-output
```

`--pretrained`、`--compile`、precision、feature dtype 和 device 均为显式选项。单教师包
只作为 staging 中间态；合并输出通过 record 数、tile-ID、维数、字节数和抽样精确值
校验后会自动清理。任务中断时保留 staging，供下次继续。

输出：

- 每个 tile 包对应一个含四教师向量的 `<name>.feat.path.iac`；
- 描述合并包及各教师维数的 `feature_build_manifest.json`。

验收条件：

- 合并特征包与源 tile 包的 record 数和 tile-ID 顺序完全一致；
- 四个教师维数全部存在；
- 合并字节能够精确复现通过校验的 staging 向量。

Virchow2 使用 class token 与 patch token 均值的拼接特征；这是冻结的特征契约，
不是训练时可切换的选项。

### 3. 可选：按训练顺序准备合并缓存

`teacher-features` 已经直接写出合并特征。只有研究训练流程还需要确定性配对行顺序时，
才运行 `training-cache`：

```bash
hcc-sempath build training-cache \
  --tile-root /assets/tiles \
  --feature-root /assets/features \
  --seed 13 \
  --workers 8
```

特征包保留在 `<feature-root>/<dataset>/<name>.feat.path.iac`。命令会检查 record 数、
tile-ID 顺序、维数、字节数并抽样验证精确相等，然后准备 tile/feature 配对行顺序。已有
缓存可使用 `--validate-only` 检查。

### 4. 构建患者/切片隔离的 manifest

```bash
hcc-sempath build manifest \
  --dev-source internal=/assets/tiles/internal \
  --public-source external=/assets/tiles/external \
  --feature-root /assets/features \
  --teacher gigapath \
  --teacher h_optimus_1 \
  --teacher uni2_h \
  --teacher virchow2 \
  --split-key patient_id \
  --val-frac 0.10 \
  --seed 13 \
  --check-artifacts \
  --output /controlled/manifests/hcc_sempath.yaml
```

有患者身份时优先使用 `--split-key patient_id`；元数据结构不同的数据可选择
`slide_id` 或 `stem`。同一个分组不会被拆分到 train 和 validation。

验收条件：

- 同一个分组键不会出现在多个 development split；
- 外部验证与 development 数据严格分离；
- `--check-artifacts` 能为每一行解析 tile 和四教师包；
- 输出 YAML 的摘要数量符合预定队列边界。

### 5. 建立受控标注状态

在同一个 tile 源上启动分类和空间联合工作区：

```bash
hcc-sempath annotate \
  --input /assets/tiles \
  --classification-state /controlled/annotations/classification.json \
  --spatial-state /controlled/annotations/spatial.json \
  --host 127.0.0.1 \
  --port 8765
```

远程访问应通过带鉴权的机构隧道或反向代理提供。`--no-auth` 只适用于明确隔离的临时
可信网络，不适用于公开服务。

UI 分别维护分类和空间状态。UI 中创建的版本位于
`<state-stem>.versions/<version-id>.json`，并有同名版本索引。稳定 label ID 不随显示名
变更；CSV 同时导出 ID 和显示名。

若要进行边界固定的复审，应显式提供有序 review manifest：

```bash
hcc-sempath annotate \
  --input /assets/tiles \
  --classification-state /controlled/annotations/classification.json \
  --spatial-state /controlled/annotations/spatial.json \
  --classification-review-manifest /controlled/review/classification.json \
  --spatial-review-manifest /controlled/review/spatial.json \
  --review-existing
```

每个 review manifest 绑定稳定 `review_id`，以及有序 tile/IAC/row 记录。完成状态按
`review_id` 保存；导航在清单末尾停止，不会静默继续抽取新 tile。

### 6. 构建固定分类监督库

```bash
hcc-sempath build supervision \
  --annotation-json /controlled/annotations/classification.json \
  --validation-annotation-json /controlled/annotations/classification_val.json \
  --training-manifest /controlled/manifests/hcc_sempath.yaml \
  --source-split train \
  --target-per-class 400 \
  --output-dir /controlled/supervision/classification
```

构建器在四个教师空间中采用受限贪心 facility coverage，从 accepted 标签中选择固定、
平衡的监督库。输出包括：

- 四个 `{teacher}_hcc_semantic_prototypes.pt` 原型注册表；
- `hcc_prototype_supervision_manifest.csv`；
- `prototype_assets_summary.json`。

可选的验证标注会进行 schema 校验并保持为独立检查点选择资产，不会合并进训练库。

验收条件：

- 七类均以预期的稳定 label ID 存在；
- 每个 accepted tile 能唯一解析到训练 manifest 和四教师记录；
- 训练与验证监督边界不存在非预期的 tile 重复；
- 信息增益只由 accepted 阳性成员相对于固定候选参考池计算。

空间标注必须保留几何信息，并由训练配置中的 spatial manifest 路径直接读取；
`build supervision` 不会把它压平成分类原型注册表。

### 7. 审计空间监督库覆盖度

论文协议提供一个被跟踪的审计工具：

```bash
python experiments/scripts/roi_information_curve.py \
  --annotation-json /controlled/annotations/spatial.json \
  --teacher-feature-packages \
    'gigapath=/assets/features/gigapath/*.iac,h_optimus_1=/assets/features/h_optimus_1/*.iac,uni2_h=/assets/features/uni2_h/*.iac,virchow2=/assets/features/virchow2/*.iac' \
  --output-root /controlled/audits/spatial_information_curve
```

该曲线在固定参考定义下衡量 accepted 阳性 tile 新增的边际特征空间信息。无关负例或
总体候选数量的变化不应改变曲线。单教师曲线用于诊断；预设的四教师汇总平台值是停止
读数。每份报告记录源标注 SHA-256 和生成时间，从而能识别过期报告。

### 8. 解析配置并训练学生模型

复制模板并替换所有占位路径：

```bash
cp configs/manifest.example.yaml /controlled/configs/manifest.yaml
cp configs/train.example.yaml /controlled/configs/train.yaml

hcc-sempath train --config /controlled/configs/train.yaml
```

需要时从完整训练状态恢复：

```bash
hcc-sempath train \
  --config /controlled/configs/train.yaml \
  --resume /outputs/hcc_sempath/checkpoints/last.pt
```

检查点保存解析后的配置、模型、优化器、scheduler、RNG 状态、动态原型刷新位置和终止
epoch 契约。延长已完成训练时，必须在传入配置中增大 `train.epochs`；resume 不会静默
产生新的终止 epoch。

输出目录包括：

- `resolved_config.json`；
- `step_metrics.csv`、`development_metrics.csv`、`selection_metrics.csv` 和
  epoch 级 `metrics.csv`；
- TensorBoard events；
- `checkpoints/last.pt`、`checkpoints/best.pt` 以及配置的诊断检查点；
- `summary.json`。

验收条件：

- 解析后的 manifest 与所有监督资产 digest 对应预定 run；
- teacher-only 和 joint ramp 在配置的 global step 启动；
- 训练和验证专家记录在优化器使用上严格分离；
- selection probe 输出非空 teacher、classification 和 spatial 指标；
- `best.pt` 由声明的联合分数选择，而不是单独按训练 loss 选择。

### 9. 评估并校准保留的检查点

```bash
hcc-sempath evaluate \
  --config /controlled/configs/train.yaml \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --split val
```

教师对齐 loss 只有在 cohort、教师、特征契约和归一化方式一致时才能横向比较。分类和
空间验证 loss 只在固定监督库上报告。空间未标注区域是 unknown，不能自动转为负例。

若需要校准生物学测量值，应使用患者/切片隔离的独立资产，且记录必须明确声明计数和
测量完整性：

```bash
python experiments/scripts/calibrate_spatial_decoder.py \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --annotation /controlled/annotations/spatial_calibration.json \
  --validation-split val \
  --output-calibration /outputs/hcc_sempath/spatial_calibration.json \
  --output-report /outputs/hcc_sempath/spatial_validation_report.json
```

普通弱监督 point/circle/brush 不代表穷尽式计数真值。因此校准资产必须针对适用成分
显式提供 `roi_count_complete` 和 `roi_measurement_complete`。

### 10. 导出仅推理发布包

```bash
hcc-sempath export \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --spatial-calibration /outputs/hcc_sempath/spatial_calibration.json \
  --output /outputs/hcc_sempath/release
```

发布目录包含：

- `model.safetensors`；
- `config.json`。

导出会移除教师 heads 和优化器状态，保留共享 encoder、分类 head、空间 head、类别/
成分 schema、输出几何与可选校准契约。写出前会核对校准文件与所选模型 digest 和空间
stride 是否一致。

## 标注语义

Point、circle 和 brush 不是可以互换的三种栅格工具。

- **Point：** 一个可信的生物学中心。对于炎症细胞、血细胞、成纤维细胞等细胞目标，
  通常表示细胞或细胞核中心；胆色素 point 可以表示很小的局部沉积灶。
- **Circle：** 中心加近似局部范围，适合空泡、血管腔等紧凑目标。
- **Brush：** 沿目标外轮廓绘制的区域或密集场。对于密集炎症细胞，它表示无法逐个标记
  中心；对于纤维间质，它表示连续面积。

Loss 构建必须保留这些差异。Point 不能使用和 brush polygon 相同的规则扩展；brush
也不代表同一位置重复出现的一组 point 实例。

未标记的成分默认是 **unknown**。只有显式负例才是 negative；穷尽式评估还必须有
completeness 声明。这一规则避免在医师未审阅的区域惩罚合理预测。

完整几何和 loss 语义见
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md)。

## 训练与检查点选择

### 默认选择原则

完整库动态原型刷新会使优化器可见 loss 在空间有效性已经饱和后继续下降。因此
HCC-SemPath 同时保留两类证据：

- 教师对齐用于确认学生仍保留基础表征；
- 固定验证集分类和空间监督用于判断模型在专家定义目标上是否仍有效。

论文 A0 协议使用预先指定并归一化的联合分数，包含直接教师保持、七分类验证 loss 与
十一成分空间验证 loss。各项相对同一个初始化基线归一化，所有监督 ramp 开启后才允许
选择和剪枝。论文实验的精确权重、hash、搜索预算与消融边界位于
[`experiments/`](experiments/README.md)；它们不是其他数据集上的隐式默认值。

### 论文 A0 搜索

先在不启动 trial 的情况下检查固定资产：

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 0
```

使用单个 coordinator 执行冻结的搜索预算：

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 10 \
  --study-trials 10
```

多个独立 trial 可分别占用 GPU，并共享一个异步 TPE coordinator：

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 10 \
  --study-trials 10 \
  --parallel-trials 4 \
  --devices 0,1,2,3
```

正式 study 通过 digest 绑定源码、解析后的 IAC 包、模型初始化、监督 manifest 和原型
注册表。失败或遗留的 trial 不会被静默当作完整证据。导出的 `best_config.yaml` 记录
所选 trial、参数、检查点和相关 digest，供后续消融使用。

## 预测格式与空间还原

`hcc-sempath infer` 写出 payload schema 为 `hcc_sempath_tile_predictions` 的版本化
IatroCache 包。每个源 tile 保存：

- 稳定的 package、dataset、split、slide、row 和 tile 标识；
- level-0 tile `x`/`y`、源 MPP 与坐标方向；
- float16 格式的七分类最终概率；
- 十一成分的 instance-response 和 abundance-response 网格；
- 网格尺寸、stride、patch size、padding 以及 model-pixel 到 level-0 的比例；
- 模型/检查点 digest 与源 index digest。

空间网格可编码为 `float16`、`uint16` 或 `uint8`。整数模式使用包头声明的有界编码，
体积更小，但在不能接受不可逆量化时不能替代 float16。

应通过包 API 读取预测，不能把 payload 当作无版本 NumPy blob：

```python
from hcc_sempath.inference.predictions import (
    PredictionPackageReader,
    grid_cell_center_level0,
)

with PredictionPackageReader("slide.pred.path.iac") as reader:
    prediction = reader.read_at(0)
    index = reader.index_table.slice(0, 1).to_pylist()[0]
    center_x, center_y = grid_cell_center_level0(
        reader.header,
        tile_x=int(index["tile_x"]),
        tile_y=int(index["tile_y"]),
        row=4,
        column=7,
    )
```

读取器会校验 IAC 容器并还原编码后的空间数组。`grid_cell_center_level0` 使用文件中
保存的几何参数，因此下游空间分析不需要猜测 tile offset、stride 或 level scale。

## 配置文件

仓库维护两个模板：

- [`configs/manifest.example.yaml`](configs/manifest.example.yaml)：cohort、split、tile 和
  教师特征位置；
- [`configs/train.example.yaml`](configs/train.example.yaml)：模型、loss、监督、优化、
  checkpoint 与 validation probe 设置。

实际运行应将模板复制到源码树外，并替换所有路径。不得提交机器特定绝对路径、访问
token、患者标识、缓存清单或私有标注位置。

训练启动时会将完整解析结果写入 `resolved_config.json`。该文件而非修改前的模板才是
准确的运行记录。Resume 会检查已存契约，不会静默接受不兼容的模型或数据布局。

## 测试

在项目环境中执行完整契约测试：

```bash
python -m pytest -q
```

不构建资产时可直接检查 CLI：

```bash
hcc-sempath --help
hcc-sempath build --help
hcc-sempath infer --help
```

长训练前至少完成以下验收：

1. tile 与四教师 IAC 校验通过；
2. manifest 分组隔离通过；
3. 分类和空间监督 digest 能正确解析；
4. 一次短训练 probe 能写出非空 teacher/classification/spatial 指标；
5. 一个小型预测包能在本地读取并还原到正确源坐标。

## 仓库结构

```text
src/hcc_sempath/     安装后的模型、数据、训练、标注和 I/O 代码
configs/             公开 manifest 与训练配置模板
docs/                维护中的科学和实现契约
experiments/         论文特定的协议、配置与派生研究工具
tests/               unit、schema、resume、CLI 与 integration 契约测试
```

`experiments/` 保留理解论文结果所需的协议，但不属于默认安装命令面。该目录不得包含
机器特定启动脚本、检查点、私有标注、教师缓存或临时报告。

## 许可证

源码和文档以 [CC BY-NC-ND 4.0](LICENSE) 发布，允许在署名条件下进行非商业使用。
可以为非商业目的制作修改版本，但不得再分发修改版本。

第三方依赖和教师模型仍受各自原始许可证与访问协议约束。SemPath 检查点不授予任何
gated 教师模型的访问权，也不包含对这些教师权重的再分发。最终学生权重发布后，将在
单独的 gated 模型仓库中按其声明的发布条款提供。

# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## 项目概述

这是葡萄膜炎（Uveitis）FA 荧光素眼底血管造影图像分割的毕业设计项目。任务是将眼底图像中的病灶渗漏区域分割为两类病灶（lesion_1 / lesion_2），论文叙事以从零训练的 U-Net 为基线，DINOv3 预训练 backbone 为强 baseline。

- 项目根目录：`/root/autodl-tmp/Uveitis-Segmentation-bs`（部分脚本中硬编码为 `/root/autodl-tmp/bs`，二者指向同一目录）
- GPU：NVIDIA Tesla V100-PCIE-32GB
- Python 3.12 / PyTorch 2.5.1+cu124 / CUDA 12.4

## 常用命令

所有命令在项目根目录下执行。

```bash
# 环境检查
python scripts/check_environment.py

# 数据集索引统计（每折有多少图像/掩码配对）
python scripts/index_dataset.py

# 运行测试（pyproject.toml 已配置 pythonpath=src）
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python -m pytest tests/test_config.py -v   # 单个测试文件

# 诊断 CLI（打印配置、数据集摘要，不做训练）
PYTHONPATH=src python -m bs.cli --config configs/default.yaml

# ---- 训练 ----

# 论文 U-Net 基线（从零训练，4分类，512x512）
python scripts/train_paper_unet.py --config configs/paper_unet_itksnap.yaml --fold f1

# DINOv3 ViT-B/16 多标签训练（768x768，核心实验）
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_vitb16_multilabel_itksnap.yaml \
    --fold f1 --batch-size 8 --grad-accum-steps 1

# DINOv3 ConvNeXt-Tiny 多标签训练
python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
    --variant tiny --fold f1 --batch-size 12 --grad-accum-steps 1

# 后台串行跑多折/多模型（生产用法）
nohup bash scripts/run_both_dinov3.sh &
nohup bash scripts/run_5fold_resume.sh &
```

训练脚本的 `--fold` 指定单折（f1-f5），`--epochs`、`--batch-size`、`--learning-rate` 等命令行参数会覆盖 YAML 配置中对应字段（由各脚本的 `resolve_config()` 处理）。

## 高层架构

### 双指标 / 双损失体系（最容易混淆的点）

项目同时存在两套评估与训练范式，切换依据是模型输出通道数和配置中的 `num_outputs` / `num_classes`：

1. **4 分类多类（multiclass）** — 用于论文 U-Net 基线
   - 掩码标签 0/1/2/3，模型输出 4 通道，`argmax` 取预测类
   - 损失：`DiceCrossEntropyLoss`（Dice + 加权交叉熵）
   - 指标：`SegmentationMetrics`（per-class IoU/Dice、pixel_acc）

2. **2 通道多标签（multilabel / paper metric）** — 用于 DINOv3 实验，是当前主线
   - 将 4 类掩码拆解为 2 个二值病灶掩码：`lesion_1 = (label==1 | label==3)`，`lesion_2 = (label==2 | label==3)`
   - 模型输出 2 通道，`sigmoid` + 阈值（默认 0.5）取预测
   - 损失：`AsymmetricFocalTverskyBCE`（非对称 Focal Tversky + 加权 BCE，处理极端类别不平衡）
   - 指标：`PaperDice`（论文定义的两病灶 Dice + macro Dice）
   - 转换逻辑在 `multilabel.py:masks_to_paper_targets()`

### 掩码编码格式

掩码使用 ITK-SNAP RGB 调色板编码，需通过 `dataset.py:decode_mask_array()` 解码。颜色映射定义在 `RGB_LABEL_COLORS`：
- 黑色 `(0,0,0)` → 0（背景）
- 红 `(255,64,64)` → 1
- 绿 `(64,210,110)` → 2
- 蓝 `(80,150,255)` → 3

数据集有两种掩码目录：`mask/`（含 HRNet 结果）和 `mask_only_itksnap/`（纯 ITK-SNAP 标注，当前主线配置使用后者）。文件格式为 `.nii.gz` 或图像格式。

### 模型架构

| 模型 | 文件 | backbone | 解码器 | 输出 |
|------|------|----------|--------|------|
| PaperUNet | `paper_unet.py` | 无（从零训练） | 标准 U-Net 编解码器 | 4 通道 |
| DinoV3SegmentationModel | `model.py` | DINOv3 ViT-B/16 | TokenFPNHead（token FPN） | 2 通道 |
| DinoV3ConvNeXtSegmentationModel | `convnext_seg.py` | DINOv3 ConvNeXt-Tiny/Small | ConvNeXtFPNDecoder（特征金字塔） | 2 通道 |

ViT 模型取 transformer 第 `[2,5,8,11]` 层中间特征，reshape 成空间 token map 后送入轻量 FPN 解码器。ConvNeXt 模型取 4 个 stage 的多尺度特征做 FPN 融合。解码器刻意保持轻量以在单卡 V100 32GB 上跑全分辨率 768x768。

### DINOv3 backbone 加载方式

DINOv3 源码 vendored 在 `backbone/dinov3/`，**不是 pip 包**。模型类在 `__init__` 中通过 `sys.path.insert(0, dinov3_code_dir)` 动态加载，预训练权重从 `weights/` 目录读取并 `load_state_dict(strict=True)`。backbone 支持冻结/部分解冻（`freeze_backbone` + `unfreeze_last_blocks`）。

### 数据流与路径解析

- `paths.py:project_path()` 基于包安装位置（`src/bs/` 上两级）解析相对路径为绝对路径
- 训练脚本独立使用 `PROJECT_ROOT = Path(__file__).resolve().parents[1]` 并 `sys.path.insert(0, str(PROJECT_ROOT / "src"))`
- 数据集根：`dataset/dataset/split_dataorigin`，5 折结构 `img/{f1..f5}` + `mask_only_itksnap/{f1..f5}`
- 像素分布极度不平衡：背景 ~94%，标签1 ~5-6%，标签3 ~0.1%，标签2 极罕见

### 数据增强

`augmentations.py` 是纯 PyTorch 自实现增强管道（无 albumentations 依赖），通过 YAML 配置驱动。支持翻转、仿射变换、前景感知裁剪（`foreground_resized_crop`）、亮度/对比度、Gamma、高斯噪声/模糊、粗粒度 Dropout、`OneOf`、`RandomOrder` 组合。每个增强块有 `prob` 和 `strength` 参数控制触发概率与强度。`paper_dataset.py` 则有独立的简化增强（仅翻转+旋转），用于论文复现。

### 训练脚本结构

训练脚本（`scripts/train_*.py`）是自包含的完整训练循环，每个 400-630 行，共享相同模式：
1. `parse_args()` → `load_config()` → `resolve_config()`（命令行覆盖 YAML）
2. `discover_samples()` 按折索引样本
3. 构建 `WeightedRandomSampler`（按病灶存在性加权采样，`lesion1_sample_weight` / `lesion2_sample_weight`）
4. AMP 训练循环 + cosine LR 调度 + 梯度裁剪
5. 每折输出到 `runs/<run-name>/<fold>/`，包含 checkpoint、TensorBoard 日志、CSV 指标、预测可视化

### 输出目录约定

```
runs/<run-name>/<fold>/
  checkpoints/{best.pt, latest.pt}
  logs/           # TensorBoard
  train.log
  metrics.csv
  predictions/    # 可视化图
```

## 关键约定

- `dataset/` 目录只读，不修改、不移动、不删除
- 训练输出统一放 `runs/`，checkpoint 和日志在其中按 run-name/fold 组织
- AMP（混合精度）训练默认开启，如遇 NaN 需降低学习率或更换 seed
- 5 折训练通过调度脚本（`run_5fold_resume.sh` 等）串行执行，`--fold` 逐折运行，`nohup` 后台执行
- 测试代码使用 `PYTHONPATH=src` 运行（pyproject.toml 已配置 pytest 的 pythonpath）
- DINOv3 backbone 权重文件较大，放在 `weights/` 目录，已在 `.gitignore` 中忽略

## 关键配置文件

| 配置 | 用途 |
|------|------|
| `configs/default.yaml` | 默认 4 分类 DINOv3 ViT 配置（诊断用） |
| `configs/dinov3_vitb16_multilabel_itksnap.yaml` | ViT-B/16 多标签主线实验（768x768） |
| `configs/dinov3_vitb16_multilabel_itksnap_640.yaml` | ViT-B/16 输入尺寸消融（640x640） |
| `configs/dinov3_convnext_tiny_multilabel_itksnap.yaml` | ConvNeXt-Tiny 多标签实验 |
| `configs/paper_unet_itksnap.yaml` | 论文 U-Net 复现基线（512x512, SGD, 100 epochs） |
| `configs/unet_multilabel_optimized.yaml` | 优化版多标签 U-Net |
# Project Instructions

- This repository root is `/root/autodl-tmp/bs`.
- All graduation-design source code, configs, scripts, docs, tests, and experiment outputs should stay under this directory.
- Do not write project files directly under `/root/autodl-tmp` or other workspace folders unless the user explicitly asks.
- Treat `dataset/` as read-only input data. Do not move, rename, delete, or rewrite dataset files.
- Keep generated model weights, logs, plots, and temporary artifacts under `outputs/`.
- Prefer Python 3.12 and the existing PyTorch CUDA environment unless the user asks to create a separate environment.

## Current Machine Summary

- CUDA reported by `nvidia-smi`: 13.0
- CUDA toolkit `nvcc`: 12.4.131
- PyTorch: 2.5.1+cu124
- Python: 3.12.3


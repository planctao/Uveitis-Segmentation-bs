# 实验结果总表

> 本文件用于集中展示可引用的实验结果；`EXPERIMENT_LOG.md` 保留更详细的训练流水账、方法说明和修正过程。  
> **当前正式结论优先引用“干净验证集”表格**；修正前结果仅作数据卫生审计和历史对照。

## 更新规范

- **新增实验**：优先追加到“实验总览（持续更新）”，若属于主线结果再同步更新“当前可信主表”。
- **指标口径**：多标签分割统一报告 `dice_1`、`dice_2`、`macro=(dice_1+dice_2)/2`；若使用 threshold sweep，需在备注写明阈值。
- **数据卫生**：从 2026-07-10 起，验证集默认剔除 `_aug` 离线增强副本（`exclude_val_augmented=True`），以避免同一病例重复计入验证集。
- **权重路径**：训练权重位于 `runs/<run-name>/<fold>/checkpoints/`，不纳入 git。

## 当前可信主表（干净 f1 验证集）

> f1 验证集：原 494 张，剔除 `_aug` 后 444 张；训练集保留增强副本。  
> 这张表是目前论文/答辩应优先引用的单折对比结果。

| 排名 | 方法 | Run Name | Head / Dynamics | Fold | Val Set | Best Epoch | Thr | Dice-1 | Dice-2 | Macro Dice | 结论 |
|---:|------|----------|-----------------|------|---------|-----------:|-----|-------:|-------:|-----------:|------|
| 1 | **RDH-PDE（Perona-Malik）** | `diffleak_f1_rdh_clean` | `rdh/pde` | f1 | clean 444 | 24 | 0.90/0.90 | 0.7958 | **0.7702** | **0.7830** | 当前最强；更轻、更可解释，主线优先 |
| 2 | S3RD（Mamba/SSM） | `diffleak_f1_s3rd_clean` | `rdh/ssm` | f1 | clean 444 | 17 | 0.80/0.80 | **0.7970** | 0.7643 | 0.7807 | 与 RDH 几乎打平；dice-1 最高，但结构更复杂 |
| 3 | ConvNeXt baseline | `diffleak_f1_baseline_clean` | `conv` | f1 | clean 444 | 16 | 0.90/0.90 | 0.7906 | 0.7632 | 0.7769 | 干净对照基线 |

### 当前结论

- **RDH-PDE 仍是当前最强单折结果**：macro 0.7830，比 baseline 高 0.61pp。
- **S3RD 与 RDH-PDE 几乎打平**：0.7807 vs 0.7830，仅低 0.23pp；之前“Mamba明显差”的结论由脏验证集放大导致，不再作为正式结论。
- **论文叙事建议**：物理扩散 RDH 与数据驱动 Mamba 传播性能接近，但 RDH 参数更少、可解释性更强、对小样本更稳，适合作为主线；S3RD 作为前沿 SSM 探索消融。
- **仍需 5 折确认**：目前只有 f1 干净单折，最终主表应补充 5-fold mean±std。

## Backbone 对比：ConvNeXt-Tiny vs ViT-B/16（f1 干净验证集）

> 同一套头（conv / RDH-PDE / S3RD）分别接在两种 DINOv3 backbone 上的对比。ViT-B/16 权重由 ModelScope `facebook/dinov3-vitb16-pretrain-lvd1689m` 经 `scripts/convert_dinov3_vit_weights.py` 转换。
> ⚠️ ViT+RDH / ViT+S3RD 为**手动提前终止**（分别跑到 ep25 / ep24，未满 30），但趋势已明确（RDH 缓慢爬升但追不上自身 baseline，S3RD 在 ep3 见顶后一路过拟合）。

| Backbone | 解码特征分辨率 | conv baseline | RDH-PDE | S3RD |
|----------|:---:|:---:|:---:|:---:|
| **ConvNeXt-Tiny**（主线） | 192×192 | 0.7769 | **0.7830** ↑ | 0.7807 ↑ |
| ViT-B/16 | 48×48 | 0.7350 | 0.7101 ↓ | 0.6398 ↓ |

ViT 各头明细（best sweep macro）：

| 方法 | Run Name | Head | Best Epoch | Dice-1 | Dice-2 | Macro | 备注 |
|------|----------|------|-----------:|-------:|-------:|------:|------|
| ViT baseline | `diffleak_f1_vitb16_clean` | conv | 24 | 0.7194 | 0.7507 | 0.7350 | 跑满 30ep |
| ViT + RDH-PDE | `diffleak_f1_vitb16_rdh_clean` | rdh/pde | 25 | 0.7325 | 0.6877 | 0.7101 | ep25 手动终止 |
| ViT + S3RD | `diffleak_f1_vitb16_s3rd_clean` | rdh/ssm(stride2) | 3 | 0.7017 | 0.5779 | 0.6398 | ep3 见顶, ep24 终止 |

**结论**：
- **ViT-B/16 整体弱于 ConvNeXt-Tiny**（baseline 0.7350 vs 0.7769，低 ~4.2pp），与历史一致。
- **RDH/S3RD 在 ViT 上不复现增益，反而掉点**（与 ConvNeXt 上“接 RDH 涨点”相反）。
- **归因**：RDH 的物理扩散演化依赖高分辨率多尺度特征（ConvNeXt 192×192）；ViT 的 48×48 粗 token 上，种子→逐格扩散的空间精度不足，上采样到 768 后边界糊掉，稀有类 lesion_2 掉得最多。
- **用途**：此对比正面支撑论文主线选 ConvNeXt 而非 ViT，并说明 RDH/S3RD 的增益与 backbone 特征分辨率强相关。

## 数据卫生修正对照

> 修正前 f1 验证集混入 `_aug` 离线增强副本，badcase 分析发现病例 `7474` 及其多个增强副本重复计入验证集，且被 S3RD 全漏检，导致 S3RD 的 lesion_2 劣势被人为放大。

| 方法 | 修正前 Macro（脏验证集 494） | 修正后 Macro（干净验证集 444） | 变化 | 解释 |
|------|-----------------------------:|-------------------------------:|-----:|------|
| RDH-PDE | 0.7913 | 0.7830 | -0.0083 | 剔除重复增强后绝对值下降，但仍最强 |
| S3RD | 0.7786 | 0.7807 | +0.0021 | 移除 7474 重复漏检后，劣势显著缩小 |
| baseline | 0.7752 | 0.7769 | +0.0017 | 基线基本稳定 |
| **RDH-S3RD 差距** | **+0.0127** | **+0.0023** | **-0.0104** | 由“明显差”修正为“几乎打平” |

## DiffLeak / 结构头历史单折消融（修正前，保留作审计）

> 以下结果来自验证集剔除 `_aug` 前的 f1 消融；已不作为正式最终结论，但可用于展示实验探索过程。

| # | 方法 | Run Name | Head | DALS | DSB | Fold | Best Epoch | Dice-1 | Dice-2 | Macro Dice | 备注 |
|---:|------|----------|------|------|-----|------|-----------:|-------:|-------:|-----------:|------|
| 1 | ConvNeXt baseline | `diffleak_f1_baseline` | conv | - | - | f1 | 24 | 0.7967 | 0.7538 | 0.7752 | 修正前基线 |
| 2 | + DSB | `diffleak_f1_dsb` | conv | - | Y | f1 | 16 | 0.7941 | 0.7688 | 0.7814 | 软边界监督提升稀有类 |
| 3 | + DALS | `diffleak_f1_dals` | conv | Y | - | f1 | 26 | 0.7986 | 0.7793 | 0.7889 | DALS 单独涨点明显 |
| 4 | + DALS + DSB | `diffleak_f1_full` | conv | Y | Y | f1 | 27 | 0.7972 | 0.7771 | 0.7872 | 叠加后略低于 DALS 单独 |
| 5 | RDH-PDE | `diffleak_f1_rdh_only` | rdh/pde | - | - | f1 | 24 | 0.7955 | 0.7872 | 0.7913 | 修正前最高；受验证集污染影响 |
| 6 | S3RD | `diffleak_f1_s3rd` | rdh/ssm | - | - | f1 | 26 | 0.7958 | 0.7614 | 0.7786 | 修正前被 7474 重复副本显著拉低 |

## 早期模型探索记录

| # | 日期 | 方法 | Run Name | Backbone / 结构 | Fold | Epochs | Dice-1 | Dice-2 | Macro / 关键指标 | 结论 |
|---:|------|------|----------|-----------------|------|--------|-------:|-------:|------------------:|------|
| 1 | 2026-06-15 | ViT baseline | `dinov3_vitb16_1fold_768_20260615_1537` | DINOv3 ViT-B/16 + TokenFPN | f1 | 30 | 0.7221 | 0.7453 | 0.7337 | ViT 单折基线 |
| 2 | 2026-06-17 | WBE v1 | `dinov3_wbe_f1` | ViT-B/16 + Wavelet Boundary Enhance | f1 | 30 | 0.7179 | 0.7318 | 0.7248 | 未超 ViT baseline |
| 3 | 2026-06-17 | WBE v2 | `dinov3_wbe_v2_f1` | ViT-B/16 + SNR/Structure Attention | f1 | 30 | 0.7194 | 0.7312 | 0.7253（sweep 0.7492） | 过拟合，仍未稳定超基线 |
| 4 | 2026-06-25 | SAM2-MAE 分割微调 | `mae_sam2_ft_f1_20260625_1404` | SAM2 Hiera-Small | f1 | ~20 | 0.6503 | 0.5969 | 0.6236 | 明显弱于 DINOv3 ConvNeXt |
| 5 | 2026-06-25 | DINOv3 荧光 MAE + 分割 | `dinov3_mae_ft_f1_20260625_1718` | DINOv3 ViT-B/16 | f1 | 30 | 0.6907 | 0.6730 | 0.6818 | MAE 域适配导致 backbone 漂移 |
| 6 | 2026-06-26 | SAM2 FPN Neck | `dinov3_vitb16_fpn_f1` | ViT-B/16 + SAM2 FPN | f1 | 30 | 0.7098 | 0.6779 | 0.6939（sweep 0.6984） | 虚拟金字塔未超基线 |
| 7 | 历史对照 | ConvNeXt-Tiny 4折均值 | - | DINOv3 ConvNeXt-Tiny | f1-f4 | - | - | - | 0.7710 | 旧主线强 baseline |

## 预训练 / 非分割阶段记录

| # | 日期 | Run Name | 任务 | Backbone | Fold | Epochs | 关键指标 | 后续用途 |
|---:|------|----------|------|----------|------|--------|----------|----------|
| 1 | 2026-06-25 | `mae_sam2_f1_20260625_1404` | MAE 自监督预训练 | SAM2 Hiera-Small | f1 | 50 | best_mae_val_loss=0.2365 | 用于 SAM2 分割微调 |
| 2 | 2026-06-25 | `dinov3_mae_f1_20260625_1718` | 荧光 MAE 自监督预训练 | DINOv3 ViT-B/16 | f1 | 30 | best_mae_val_loss=0.3984 | 用于 DINOv3 分割微调 |

## 可视化与分析产物

| 类型 | 路径 | 内容 |
|------|------|------|
| RDH vs S3RD badcase 图（修正前） | `runs/compare_rdh_s3rd/compare.png` | 原图 / GT / RDH-PDE / S3RD 四联图，暴露 7474 重复副本问题 |
| 对比脚本 | `scripts/compare_rdh_s3rd.py` | 逐图预测、计算 dice、按 badcase 排序并生成对比图；当前已默认剔除 `_aug` |
| RDH 可解释演化图 | `runs/rdh_vis/rdh_visualization.png` | 种子、传导、演化过程可视化 |
| DALS 合成可视化 | `runs/dals_vis/dals_visualization.png` | 扩散外观合成效果诊断 |

## 实验总览（持续更新模板）

> 后续新实验建议追加到此表；若结果成为主线，再同步到前面的正式主表。

| 日期 | Run Name | 方法/改动 | Config | Fold(s) | Val Set | Epochs | Dice-1 | Dice-2 | Macro | 权重路径 | 结论/下一步 |
|------|----------|-----------|--------|---------|---------|--------|-------:|-------:|------:|----------|-----------|
| YYYY-MM-DD | `run_name` | 简述新增模块/参数 | `configs/xxx.yaml` | f1/f1-f5 | clean/raw | 30 | 0.0000 | 0.0000 | 0.0000 | `runs/.../checkpoints/best.pt` | 结论与是否进入主线 |

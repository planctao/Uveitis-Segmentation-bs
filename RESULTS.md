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

## VA-RDH：血管锚定各向异性反应扩散头（2026-07-23，torch 2.9 同环境）

> **创新点**：把 RDH-PDE 的各向同性 Perona-Malik 标量传导升级为由图像**结构张量**导出的**相干增强各向异性扩散张量**（Weickert CED）。渗漏沿血管相干方向强扩散、跨血管方向弱扩散，对应“FA 渗漏源自受损血管并沿血管树蔓延”的临床先验；仅新增 **1 个可学习相干对比标量**，推理零额外开销。诊断发现原 RDH 在 K=8 步时“演化过细微、物理欠表达”，故取 **K=16** 步充分表达扩散。
> ⚠️ 本批次为 torch 2.9 同环境重跑；RDH-PDE 同环境 f1=0.7828（复现历史 0.7830）。所有对比为 clean 验证集 best per-lesion 阈值 sweep macro。

**f1 变体消融（择优 K=16）**

| 方法 | Head | best sweep macro | dice_1 | dice_2 |
|------|------|-----------------:|-------:|-------:|
| conv baseline（同环境） | conv | 0.7799 | 0.7967 | 0.7632 |
| RDH-PDE isotropic（锚点） | rdh/pde | 0.7828 | 0.7953 | 0.7702 |
| VA-RDH（aniso, K=8） | rdh/pde | 0.7821 | 0.7935 | 0.7707 |
| **VA-RDH（aniso, K=16）** | rdh/pde | **0.7872** | 0.7943 | **0.7802** |
| VA-RDH struct σ=4 | rdh/pde | 0.7866 | 0.7954 | 0.7778 |
| VA-RDH across（跨血管外渗） | rdh/pde | 0.7843 | 0.7983 | 0.7704 |
| VA-RDH contrast=0.3 | rdh/pde | 0.7776 | 0.7888 | 0.7664 |
| RDH-PDE + FIC（自监督成像一致） | rdh/pde | 0.7670 | 0.7864 | 0.7475 |
| VA-RDH + FIC | rdh/pde | 0.7718 | 0.7821 | 0.7615 |

**5 折对比（VA-RDH K=16 vs RDH-PDE isotropic，clean val，best sweep macro）**

| Fold | VA-RDH K=16 macro | dice_1 | dice_2 | RDH-iso macro | dice_1 | dice_2 |
|------|------------------:|-------:|-------:|--------------:|-------:|-------:|
| f1 | 0.7872 | 0.7943 | 0.7802 | 0.7828 | 0.7953 | 0.7702 |
| f2 | 0.7877 | 0.8217 | 0.7537 | 0.7830 | 0.8214 | 0.7446 |
| f3 | 0.8153 | 0.8151 | 0.8155 | 0.8149 | 0.8153 | 0.8146 |
| f4 | 0.7559 | 0.7990 | 0.7128 | 0.7591 | 0.8010 | 0.7172 |
| f5 | 0.7889 | 0.8196 | 0.7582 | 0.7888 | 0.8229 | 0.7548 |
| **mean±std** | **0.7870±0.0188** | 0.8100 | **0.7641** | 0.7857±0.0178 | 0.8112 | 0.7603 |
| **Δ (VA−iso)** | **+0.13pp** | −0.12pp | **+0.38pp** | | | |

Run: `runs/va_rdh_iters16_5fold/{f2..f5}` + `runs/va_rdh_iters16_f1/f1`；对照 `runs/rdh_iso_5fold/{f2..f5}` + `runs/rdh_pde_ref_t29_f1/f1`。Config: `configs/dinov3_convnext_tiny_va_rdh_iters16.yaml`。

### VA-RDH 结论（诚实）

- **VA-RDH（K=16）在稀有类 dice_2 上给出小而方向一致的提升**：5 折均值 **+0.38pp**（4/5 折更优，仅 f4 略低 0.44pp），macro **+0.13pp**，dice_1 基本持平（−0.12pp）。
- 提升幅度在折间 std（±1.9pp）以内，**非强显著**；但方向稳定、只加 1 个标量、可解释（可导出血管取向扩散张量与演化过程），且**不损伤 dice_1**，符合“小样本上物理先验 > 复杂模块”的主线叙事。
- **K 的作用明确**：K=8 与 baseline 打平，K=16 才涨点——印证“物理欠表达”诊断，可作为消融卖点。
- **FIC（荧光成像自监督一致）为诚实负结果**：RDH+FIC −1.58pp、VA-RDH+FIC −1.03pp。自监督成像一致项把预测拉向高亮非渗漏结构，在 ~2k 小数据上过强；与本项目“复杂/数据驱动模块易过拟合”的历史一脉相承，作为负消融记录。
- **推理侧 flip-TTA（h/v/hv 平均）叠加，已完成**：VA-RDH K16 + TTA 5 折 sweep macro=**0.7890±0.0206**（dice_1 0.8121, dice_2 0.7658），TTA 相对无 TTA **+0.20pp**，相对 RDH-PDE 各向同性基线（0.7857）**+0.33pp**；逐折 f1 0.7928 / f2 0.7934 / f3 0.8164 / f4 0.7523 / f5 0.7898（f4 折 TTA 略降，其余折均升）。Config: `configs/eval_va_rdh_iters16_tta.yaml`；命令加 `--disable-postprocess --disable-fov-mask` 走快速 TTA+阈值 sweep 路径。
- **剩余可选杠杆（未做）**：形态学/FOV 后处理网格搜索（CPU 慢、增益边际）；RDH-PDE⊕VA-RDH 双扩散几何异构集成（需给 `evaluate_ensemble_postprocess.py` 补 VA-RDH 参数接线）。二者均可作为进一步提升的备选。

## 边缘检测创新 Wave 2：Oriented-PDC / GAC / EOC（2026-07-24，torch 2.9 同环境，f1）

> 在既有边缘线（CPDC 边缘头 + 软边缘监督 + Boundary-DoU，Wave 1 最佳 edge-soft 0.7832）之上继续创新，落地三条**新**边缘检测方向（均低参数、强先验）：
> 1. **Oriented-PDC**：把各向同性 CPDC 扩展为 **CPDC+APDC(角向)+RPDC(径向)** 多方向像素差分卷积并联组（PiDiNet 全集），捕捉有取向/径向的血管与渗漏边界。代码 `src/bs/edge.py`（`AngularDifferenceConv2d`/`RadialDifferenceConv2d`/`PDCBank`）。
> 2. **GAC**：边缘指示 `g=1/(1+|∇I|²/κ²)` 驱动的可微**测地主动轮廓**边界演化头（边缘停止曲率流 + 边缘平流），另接 PDC 学习边缘监督。代码 `src/bs/edge.py:GeodesicActiveContourHead`，`head=gac`。
> 3. **EOC**：无参数**边缘方向一致性**损失，约束预测边界法向与原图荧光梯度方向对齐。代码 `src/bs/multilabel.py:EOCLoss`/`edge_orientation_consistency`。
> 均为 clean 验证集 best per-lesion 阈值 sweep macro；对照 conv baseline 0.7799、Wave 1 edge-CPDC-soft 0.7832。

| 方法 | Head / 损失 | best sweep macro | dice_1 | dice_2 | Δ vs conv |
|------|------------|-----------------:|-------:|-------:|----------:|
| conv baseline（同环境） | conv | 0.7799 | 0.7967 | 0.7632 | - |
| edge CPDC-soft（Wave 1） | edge/cpdc+soft | 0.7832 | 0.7994 | 0.7670 | +0.33pp |
| **Oriented-PDC soft（新）** | edge/cpdc+apdc+rpdc+soft | **0.7841** | 0.7926 | **0.7756** | **+0.42pp** |
| GAC（新） | gac + soft edge | 0.7717 | 0.7727 | 0.7707 | −0.82pp |
| EOC on conv（新） | conv + EOC(0.1) | 0.7760 | 0.7943 | 0.7577 | −0.39pp |
| GAC + Oriented-PDC | gac/bank + soft | 0.7759 | 0.7872 | 0.7647 | −0.40pp |

Runs: `runs/edge_orientedpdc_f1`、`runs/edge_gac_f1`、`runs/edge_eoc_f1`、`runs/edge_gac_orientedpdc_f1`；命令用 itksnap 配置 + `--head {edge,gac,conv}` + `--edge-pdc-types cpdc,apdc,rpdc` / `--edge-soft` / `--eoc-weight`。

### Wave 2 结论（诚实）

- **Oriented-PDC 是本轮唯一正向、且为最优边缘方案**：多方向差分卷积组 macro **0.7841**，超 conv baseline **+0.42pp**、超 Wave 1 单 CPDC 软边缘 **+0.09pp**（噪声内），增益集中在**稀有类 dice_2**（超 baseline **+1.24pp**）。参数量≈普通卷积，符合"低参强先验"主线；印证"血管/渗漏边界有取向、径向浓度跃变"的假设。
- **GAC 为诚实负结果**（−0.82pp）：测地主动轮廓演化在 ep10 见顶后 dice_1 明显走低（0.7727）——边缘平流可能把边界吸附到血管等非病灶强边缘、且过度平滑；稀有类 dice_2 反而不低（0.7707），但整体掉点。与"复杂演化模块在小数据上易漂移"一脉相承。
- **EOC 为诚实负结果**（−0.39pp）：方向一致性损失在 weight=0.1 时轻微损伤 dice_2（0.7577），与 FIC 类似——把边界拉向图像强边缘对稀有弱渗漏不利。可留作负消融或降权重复验。
- **下一步（可选）**：对 Oriented-PDC 做 5 折确认其相对 baseline 的稳定增益；并可尝试 **VA-RDH ⊕ Oriented-PDC**（区域各向异性扩散 + 多方向边缘检测两条正交正向线叠加）。

## CSD-DB：源点-轮廓双分支检测头（2026-07-25，f1）

> **目标**：从“单一分割头/单边缘分支”升级为真正的双分支检测结构：
> 1. **Source/Core branch** 显式检测高荧光渗漏核心/起漏源；
> 2. **Contour/Edge branch** 用 Oriented-PDC 检测病灶轮廓；
> 3. 两个检测分支通过零初始化 gate 调制主分割特征，并用 source-inside consistency 约束“源点必须在分割内部”。
> 代码：`src/bs/dual_branch.py`（`CoreContourDualBranchHead` / `DualBranchLoss` / `source_core_target`）；`convnext_seg.py` 新增 `head_type=dual_branch`；训练 CLI 支持 `--source-*` 与 `--edge-pdc-types`。全量测试 **202 passed**。

| 方法 | Head / 分支 | best sweep macro | dice_1 | dice_2 | 结论 |
|------|-------------|-----------------:|-------:|-------:|------|
| conv baseline | conv | 0.7799 | 0.7967 | 0.7632 | 对照 |
| Oriented-PDC | contour-only edge | 0.7841 | 0.7926 | 0.7756 | 边缘分支正向 |
| VA-RDH K16 | anisotropic RDH | **0.7872** | 0.7943 | 0.7802 | f1 macro 仍最高 |
| VA-RDH + CSD-DB fusion | rdh_dual_branch | 0.7786 | 0.7941 | 0.7632 | 融合负结果，未超过任一单独分支 |
| **CSD-DB full** | source + contour + consistency | **0.7854** | 0.7876 | **0.7832** | 双分支最佳，dice_2 最高 |
| CSD-DB source-only | source only | 0.7771 | 0.7900 | 0.7643 | 单独源点分支不足 |
| CSD-DB edge-only | contour only inside dual head | 0.7758 | 0.7846 | 0.7670 | 不如独立 Oriented-PDC，可能 gate/结构差异影响 |
| CSD-DB light consistency | source + contour + weak consistency | 0.7800 | 0.7912 | 0.7688 | consistency=0.05 仍不如 full |

Runs: `runs/dual_branch_full_f1`、`runs/dual_branch_source_only_f1`、`runs/dual_branch_edge_only_f1`、`runs/dual_branch_light_consistency_f1`。

### CSD-DB 结论（诚实）

- **完整双分支 CSD-DB 是正向结果**：macro 0.7854，超 conv baseline **+0.55pp**，也高于 Oriented-PDC 以外多数边缘探索；但仍低于 VA-RDH K16 的 f1 macro 0.7872。
- **最重要的亮点是稀有类 dice_2=0.7832，为当前 f1 训练侧最高**：高于 VA-RDH K16 的 0.7802，也高于 Oriented-PDC 的 0.7756。说明“source/core + contour/edge”双检测分支确实更照顾小而稀有的 lesion_2。
- Source-only / edge-only 均不如 full，说明单独检测源点或轮廓都不足，**源点与轮廓的联合约束**才是有效部分。
- 局限：dice_1 下降到 0.7876，导致 macro 未超过 VA-RDH；若论文主目标是整体 macro，VA-RDH K16+TTA 仍为主方法；若强调稀有病灶 lesion_2，CSD-DB 是很好的创新消融。
- **VA-RDH + CSD-DB 融合已尝试，为诚实负结果**：`rdh_dual_branch_f1` best macro=0.7786（dice_1=0.7941, dice_2=0.7632），低于 VA-RDH K16（0.7872）和 CSD-DB full（0.7854）。可能原因是 source/contour 辅助损失与 RDH 的物理演化优化目标相互干扰，零初始化 residual 虽保证初始安全，但辅助分支梯度仍改变共享 neck/backbone 表征。论文建议将 **VA-RDH（整体最强）** 与 **CSD-DB（稀有类最高）** 分开报告，而不是合并为主方法。



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

## ZAB-LeakNet 新方案（2026-07-20）

ZAB 将 lesion_2 的稀有 presence、conditional area 和空间证据分开建模，
并用弱解剖热图做软定位。clean f1 完整 30 epoch 结果：

| 方法 | Run Name | best epoch | Dice-1 | Dice-2 | macro |
|------|----------|-----------:|--------:|--------:|------:|
| ZAB v1 | `zab_f1` | 13 | 0.7906 | 0.7664 | **0.778518** |

ZAB v1 高于 ConvNeXt reference `0.7769`（+0.16pp），但尚未超过 RDH-PDE
`0.7830`。ZAB v2 的软层级/双向耦合已实现于
`configs/dinov3_convnext_tiny_zab_v2.yaml`，当前只有 v1 权重代理筛查，不能
替代 v2 正式训练；详见 `outputs/reports/zab_screen.md`。

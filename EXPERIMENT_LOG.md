# 分割实验记录

> 本文件只记录病灶分割实验；图像分类实验单独记录在 [classification/EXPERIMENT_LOG.md](classification/EXPERIMENT_LOG.md)。

> 本文件记录每次训练实验的关键信息，用于权重版本管理。权重存储在 `runs/` 下但不纳入 git。

## 稀有病灶采样强度消融：lesion_2 采样权重6→3（2026-10-04启动，2026-10-06阶段性归档）

**当前结论**：在已完成的 f2、f3 两折30轮配对实验中，降低采样权重均改善
lesion_2 和 macro Dice；无TTA的两折平均 macro 从 **0.796555** 增至
**0.800765**（**+0.421个百分点**），lesion_2 平均提升 **0.907个百分点**。
这是单seed、两折验证结果，**五折尚未完成，也未确认超过历史最佳模型**。

### 思路与基准

尝试减少稀有病灶图片的重复采样，同时保留两个病灶通道的联合监督，检验原先
过强的过采样是否影响泛化。`WeightedRandomSampler(replacement=True)` 对含
lesion_2（原标签2或3）的整张图片赋权，其余图片权重为1；权重6→3意味着
每张阳性图相对普通图的抽样机会从6倍降到3倍，不改变像素损失权重。
f2训练集1963张，其中393张含 lesion_2；权重6/3时理论阳性抽样占比分别为
60.03%/42.89%，每轮抽样总量均为1963，训练loader丢弃最后不足8张的批次。

本次matched基准是 **DINOv3 ConvNeXt-Tiny + FPN（decoder attention=none）+
VS-UBR**，不含HMVF。旧HMVF实现和训练checkpoint在当前工作区缺失，因此
从commit `565e466` 的源码建立隔离快照；原有工作区修改和历史结果保留。
`control` 与 `sample3` 唯一训练差异为 `lesion2_sample_weight=6/3`。

共同配置：seed=42，768×768，batch=8，decoder LR=1e-4、backbone LR=1e-5，
cosine调度，BCE正样本权重 `[3,60]`、通道损失权重 `[1,1]`，
VS-UBR weight=0.1、vessel_weight=0.5；完整训练/验证样本，相同增强，
验证集剔除 `_aug` 副本（f2=426张、f3=453张）。每折使用其余四折训练。
完整训练从DINOv3预训练骨干和新解码器重新开始，使用30轮调度，不续接5轮模型。

按验证集 `paper_sweep_ind_macro_dice` 选择best checkpoint，阈值网格为
`[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]`。
以下分项Dice和macro均来自同一个checkpoint、相同评估口径，不拼接各轮分项峰值。
模型和阈值均在验证集选择，不是独立测试集成绩。

### 第一阶段：f2五组5轮筛选（均完成）

| 组 | 相对control的唯一改动 | Best epoch | Dice-1 | Dice-2 | Macro Dice | Δ Macro（个百分点） |
|---|---|---:|---:|---:|---:|---:|
| control | 无 | 5 | 0.807032 | 0.730572 | 0.768802 | — |
| sample3 | lesion_2采样权重6→3 | 4 | 0.808102 | 0.752654 | **0.780378** | **+1.158** |
| bce20 | lesion_2 BCE正样本权重60→20 | 5 | 0.804846 | 0.733619 | 0.769233 | +0.043 |
| slow_backbone | backbone LR 1e-5→1e-6 | 5 | 0.777658 | 0.721232 | 0.749445 | −1.936 |
| joint_l2_weight2 | 通道损失权重[1,1]→[1,2] | 3 | 0.778884 | 0.754655 | 0.766769 | −0.203 |

sample3兼顾两个通道，因此进入完整训练；损失翻倍虽然提升lesion_2，却损伤
lesion_1，其他方案在短训练中未显示足够收益。5轮筛选不用于判断长期收敛上限。
固定0.5阈值、上述同一best checkpoint的macro：control=0.762642，
sample3=0.774142。原始结果在本地 `outputs/training_scheme_screen_20261004/f2/comparison.csv`。

### 第二阶段：30轮配对训练（已完成f2、f3）

| Fold | 组 | Best epoch | 阈值L1/L2 | Dice-1 | Dice-2 | Macro Dice |
|---|---|---:|---|---:|---:|---:|
| f2 | control（采样6） | 20 | 0.7/0.05 | 0.830838 | 0.727693 | 0.779266 |
| f2 | sample3（采样3） | 20 | 0.8/0.05 | 0.829257 | 0.738909 | **0.784083** |
| f3 | control（采样6） | 21 | 0.6/0.9 | 0.814368 | 0.813319 | 0.813844 |
| f3 | sample3（采样3） | 21 | 0.7/0.9 | 0.814645 | 0.820248 | **0.817446** |

sample3相对control：f2的Dice-1/2/Macro分别 **−0.158/+1.122/+0.482个百分点**，
f3分别 **+0.028/+0.693/+0.360个百分点**。两折macro和lesion_2均改善，
平均Dice-1下降0.065个百分点。f2的lesion_2最优阈值落在网格下界0.05，
仅代表该网格内的校准结果，不能据此判断更低阈值的表现。

每组结束后，对同一个无TTA选出的best checkpoint进行原图+h/v/hv翻转TTA，
只扫描预测阈值，不重新选择训练epoch，不启用形态学、FOV或强度后处理：

| Fold | 组 | TTA阈值L1/L2 | TTA Dice-1 | TTA Dice-2 | TTA Macro | Δ Macro（个百分点） |
|---|---|---|---:|---:|---:|---:|
| f2 | control | 0.7/0.05 | 0.833855 | 0.725970 | 0.779913 | — |
| f2 | sample3 | 0.8/0.05 | 0.832808 | 0.744145 | **0.788476** | **+0.856** |
| f3 | control | 0.6/0.9 | 0.817679 | 0.820691 | 0.819185 | — |
| f3 | sample3 | 0.7/0.9 | 0.817297 | 0.821493 | **0.819395** | **+0.021** |

仅f2/f3的均值±总体标准差（**不是五折结果**）：

| 评估 | 组 | Mean Dice-1 | Mean Dice-2 | Mean Macro±population std |
|---|---|---:|---:|---:|
| 无TTA、独立阈值 | control | 0.822603 | 0.770506 | 0.796555±0.017289 |
| 无TTA、独立阈值 | sample3 | 0.821951 | 0.779578 | **0.800765±0.016682** |
| 翻转TTA、独立阈值 | control | 0.825767 | 0.773330 | 0.799549±0.019636 |
| 翻转TTA、独立阈值 | sample3 | 0.825052 | 0.782819 | **0.803935±0.015459** |

相同best checkpoint使用固定0.5阈值时，两折平均macro为control=0.789947、
sample3=0.790722，仅 **+0.077个百分点**，小于独立阈值下的+0.421个百分点。
因此收益随预测阈值变化，应同时报告固定阈值和校准阈值，不能全部归因于
不依赖阈值的分割能力提升。

### 历史参照、运行状态与证据

历史VS-UBR五折独立阈值macro为0.7881±0.0194，f2/f3为0.7835/0.8190；
历史VA-RDH K16+TTA五折macro为0.7890±0.0206，f2/f3为0.7934/0.8164。
本次sample3+TTA在f2低于后者0.492个百分点、f3高0.299个百分点，不能据两折
均值或单折峰值宣称超过历史最佳。历史架构、环境和选模协议不同，采样权重贡献
应以本轮配对control为依据。

lesion_1对比也应固定fold和TTA口径：历史VS-UBR复现实验f2所有epoch的
独立阈值Dice-1峰值为0.829373，本次control峰值0.830838（ep20），sample3
峰值0.830670（ep25），只高约0.13–0.15个百分点。表中sample3的ep20
Dice-1=0.829257来自macro最佳模型，不能与ep25分项峰值混用。

截至2026-10-06核查，10次计划训练中完成4次（f2、f3各两组），且均完成TTA。
f4/control最后日志为第8轮训练结束，但只记录7轮验证；该部分最佳macro
0.737387（ep7）不纳入完整结果。f4/sample3、f5两组、f1两组尚未开始。
训练进程已不存在，GPU空闲；`status.json` 仍写着training，是中断前的旧状态，
不可作为正在运行的证据。现有日志不足以确定中断原因。

已完成四组各跳过9–10个非有限梯度训练批次，验证批次未跳过；checkpoint保存
和加载检查通过。实现检查22项通过，覆盖通道监督、CPU/CUDA阈值统计一致性、
TTA翻转对齐、原子保存失败保留旧权重、完整epoch校验及部分折汇总标识。

本地证据根目录：`outputs/training_scheme_full_20261004/`，包括
`protocol.json`（源码/配置SHA-256）、`comparison.csv/json`、`fold_summary.json`、
各折 `configs/{control,sample3}.yaml`，以及
`<fold>/runs/<scheme>/<fold>/{metrics.csv,train.log,flip_tta.json,checkpoints/best.pt}`。
记录所用 `comparison.csv` SHA-256为
`233d09517e1241ed74ac1821f66e19f6d3c86a434b5bc144b2ada9df509a9f79`。
大体积模型和原始输出按仓库约定保留本地，本节数值已与逐轮CSV和TTA JSON交叉校验。

后续需补完f4/f5/f1，再做五折配对汇总；目前只支持将采样权重3作为待复核候选，
不能替代历史主模型或声称统计显著。

| # | 日期 | Run Name | Backbone | Config | Fold | Epochs | 关键指标 | 权重路径 | 改动摘要 | 备注 |
|---|------|----------|----------|--------|------|--------|----------|----------|----------|------|
| 1 | 2026-06-15 | dinov3_vitb16_1fold_768_20260615_1537 | ViT-B/16 | configs/dinov3_vitb16_multilabel_itksnap.yaml | f1 | 30 | best_macro_dice=0.7337; dice_1=0.7221; dice_2=0.7453 | runs/dinov3_vitb16_1fold_768_20260615_1537/f1/checkpoints/ | Baseline: DINOv3 ViT-B/16 + TokenFPNHead 无创新点; bs=1 grad_accum=4 768x768 | best_epoch=23; val_loss=0.552 |
| 2 | 2026-06-17 | dinov3_wbe_f1 | ViT-B/16 + WBE v1 | configs/dinov3_vitb16_multilabel_wbe.yaml | f1 | 30 | best_macro_dice=0.7248; dice_1=0.7179; dice_2=0.7318 | runs/dinov3_wbe_f1/f1/checkpoints/ | 新增小波边界增强WBE v1模块(4-scale, bottleneck=256, 6.44M参数); bs=8 grad_accum=2 | best_epoch=18; WBE未提点(-0.89%); batch_size不同影响公平对比 |
| 3 | 2026-06-17 | dinov3_wbe_v2_f1 | ViT-B/16 + WBE v2 | configs/dinov3_vitb16_multilabel_wbe_v2.yaml | f1 | 30 | best_macro_dice=0.7253; dice_1=0.7194; dice_2=0.7312; sweep_macro=0.7492 | runs/dinov3_wbe_v2_f1/f1/checkpoints/ | WBE升级v2: 借鉴PFESA加入SNR零参数边缘先验+Structure Attention+snr_gate自适应融合; bs=8 grad_accum=2 | best_epoch=13; 仍未超baseline(-0.84%); 过拟合严重(ep13后无提升) |
| 4 | 2026-06-25 | mae_sam2_f1_20260625_1404 | SAM2 Hiera-Small (MAE预训练) | configs/sam2_mae_multilabel.yaml | f1 | 50 | best_mae_val_loss=0.2365 | runs/mae_sam2_f1_20260625_1404/f1/checkpoints/best.pt | Stage 1 MAE自监督预训练: mask_ratio=0.75, MSE重建损失, lr=1e-4 warmup5+cosine; bs=4 768x768 AMP | best_epoch=50; encoder权重将用于Stage 2微调; train_loss=0.2131 |
| 5 | 2026-06-25 | mae_sam2_ft_f1_20260625_1404 | SAM2 Hiera-Small (分割微调) | configs/sam2_mae_multilabel.yaml | f1 | ~20 (手动停止) | best_macro_dice=0.6236; dice_1=0.6503; dice_2=0.5969 | runs/mae_sam2_ft_f1_20260625_1404/f1/checkpoints/best.pt | Stage 2 分割微调: 加载MAE encoder权重, 0.9 Dice+0.1 BCE损失; bs=4 768x768 AMP | 用户手动停止; 超论文原文0.5593; 对比DINOv3 ConvNeXt 0.7710差距明显 |
| 6 | 2026-06-25 | dinov3_mae_f1_20260625_1718 | DINOv3 ViT-B/16 (荧光MAE预训练) | configs/dinov3_vitb16_mae_multilabel.yaml | f1 | 30 | best_mae_val_loss=0.3984 | runs/dinov3_mae_f1_20260625_1718/f1/checkpoints/best.pt | Stage 1 荧光MAE预训练: mask_mode=fluorescence, mask_ratio=0.75, 亮度加权掩码概率[0.3,0.8]; lr=1e-5 cosine; bs=2 grad_accum=2 768x768 AMP | best_epoch=27; train_loss=0.2922; 全backbone解冻适配FA域 |
| 7 | 2026-06-25 | dinov3_mae_ft_f1_20260625_1718 | DINOv3 ViT-B/16 (荧光MAE+分割微调) | configs/dinov3_vitb16_mae_multilabel.yaml | f1 | 30 | best_macro_dice=0.6818; dice_1=0.6907; dice_2=0.6730 | runs/dinov3_mae_ft_f1_20260625_1718/f1/checkpoints/best.pt | Stage 2 分割微调: 加载MAE-adapted backbone, AsymmetricFocalTverskyBCE损失; lr=1e-4 backbone_lr=1e-5; bs=1 grad_accum=4 768x768 AMP | best_epoch=18; **未超ViT baseline 0.7337 (-5.19pp)**; MAE域适配导致backbone漂移 |
| 8 | 2026-06-26 | dinov3_vitb16_fpn_f1 | ViT-B/16 + SAM2 FPN | configs/dinov3_vitb16_fpn_multilabel.yaml | f1 | 30 | best_macro_dice=0.6939; dice_1=0.7098; dice_2=0.6779; sweep_macro=0.6984 | runs/dinov3_vitb16_fpn_f1/f1/checkpoints/best.pt | SAM2 FPN Neck适配ViT: 虚拟金字塔(48→24→12→6)+top-down融合+深度监督(3个辅助损失); 不改backbone; bs=1 grad_accum=4 768x768 AMP | best_epoch=24; **未超baseline 0.7337 (-3.98pp)**; 虚拟金字塔非真多尺度 |
| 9 | 2026-07-23 | rdh_pde_ref_t29_f1 + rdh_iso_5fold | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_multilabel_itksnap.yaml (--head rdh) | f1-f5 | 30 | 5折 sweep macro=0.7857±0.0178; dice_1=0.8112; dice_2=0.7603 | runs/rdh_pde_ref_t29_f1/f1 + runs/rdh_iso_5fold/{f2..f5} | torch2.9 同环境 RDH-PDE 各向同性锚点 5 折; 复现历史 f1=0.7828 | clean val; best per-lesion 阈值 sweep macro; VA-RDH 对照基线 |
| 10 | 2026-07-23 | va_rdh_iters16_5fold + va_rdh_iters16_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_va_rdh_iters16.yaml | f1-f5 | 30 | 5折 sweep macro=0.7870±0.0188; dice_1=0.8100; dice_2=0.7641 | runs/va_rdh_iters16_f1/f1 + runs/va_rdh_iters16_5fold/{f2..f5} | 新增 VA-RDH: 结构张量相干增强各向异性扩散(Weickert CED, src/bs/rdh.py)+K=16; 仅 +1 可学习标量; 推理零开销 | Δvs iso: macro +0.13pp / dice_2 +0.38pp(4/5折优) / dice_1 持平; 稀有类小而一致涨点; 184 tests passed |
| 11 | 2026-07-23 | va_rdh_f1 / va_rdh_{struct4,across,contrast03}_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_va_rdh.yaml | f1 | 30 | K8=0.7821; K16=0.7872; struct σ4=0.7866; across=0.7843; contrast0.3=0.7776 | runs/va_rdh_*_f1/f1/checkpoints/ | VA-RDH f1 变体消融(择优 K=16); ced_direction/struct尺度/contrast 网格 | K=8 与 baseline 打平, K=16 涨点印证"物理欠表达"诊断; across(外渗)/contrast 次优 |
| 12 | 2026-09-06 | va_rdh_selfreg_f1_30 | ConvNeXt-Tiny + VA-RDH + MSAS(SelfReg-lite) | configs/dinov3_convnext_tiny_va_rdh_selfreg.yaml | f1 | 30 | best fixed macro=0.7641 (ep13); best sweep macro=0.7709 (ep13); final sweep=0.7709; peak VRAM=11.22GB | runs/va_rdh_selfreg_f1_30/f1/checkpoints/best.pt | 训练期对 3 个粗尺度 FPN 特征加入辅助分割头，权重 0.2/0.04/0.008；推理仍为单一 VA-RDH 输出 | 诚实负消融：低于 VA-RDH K16 f1=0.7872（−1.63pp）；训练稳定且无 OOM，但不进入主方案 |
| 13 | 2026-09-06 | va_rdh_cbam_f1_screen | ConvNeXt-Tiny + VA-RDH + CBAM-lite | configs/dinov3_convnext_tiny_va_rdh_iters16.yaml + `--decoder-attention cbam` | f1 | 5 | best fixed macro=0.7275 (ep5); best sweep macro=0.7535 (ep5); peak VRAM=12.67GB | runs/va_rdh_cbam_f1_screen/f1/checkpoints/best.pt | FPN 融合后加入轻量 channel-spatial attention（通道 MLP + 7×7 spatial gate） | 5 epoch 筛选未追平 VA-RDH；首轮 8 个非有限梯度 batch，后续正常；不扩展 30 epoch，保留为负筛选 |
| 14 | 2026-09-06 | va_rdh_wbe_f1_screen | ConvNeXt-Tiny + VA-RDH + WBE/PFESA-lite | runs/va_rdh_mosc_f1_screen2/config.yaml + `--decoder-attention wbe` | f1 | 5 | best fixed macro=0.7351 (ep5); best sweep macro=0.7579 (ep5); peak VRAM=12.42GB | runs/va_rdh_wbe_f1_screen/f1/checkpoints/best.pt | 在 FPN 融合后用 96 通道 bottleneck 做 Haar DWT 高频增强，zero-init residual；参考 WBE/PFESA 频域边界模块 | 低于 VA-RDH K16=0.7872（−2.93pp），不扩展 30 epoch；频域增强在当前小数据/高分辨率 decoder 上易扰动主线 |
| 15 | 2026-09-06 | va_rdh_boundary_f1_screen | ConvNeXt-Tiny + VA-RDH + edge-guided boundary refinement | runs/va_rdh_mosc_f1_screen2/config.yaml + `--decoder-attention boundary-lite` | f1 | 5 | best fixed macro=0.7196 (ep4); best sweep macro=0.7493 (ep4); peak VRAM=13.36GB | runs/va_rdh_boundary_f1_screen/f1/checkpoints/best.pt | 参考 ET-Net/CTO，将 depthwise high-pass 与学习边界门控作为 neck 前零初始化残差，不增加辅助 edge loss | 低于 VA-RDH K16（−3.79pp），不扩展；直接边界重加权与 PDE 演化目标可能冲突 |
| 16 | 2026-09-06 | va_rdh_pdcneck_f1_screen | ConvNeXt-Tiny + VA-RDH + Oriented-PDC neck residual | runs/va_rdh_mosc_f1_screen2/config.yaml + `--decoder-attention pdc-neck` | f1 | 5 | best fixed macro=0.5727 (ep2); best sweep macro=0.7423 (ep5); peak VRAM=13.31GB | runs/va_rdh_pdcneck_f1_screen/f1/checkpoints/best.pt | 参考 PiDiNet/Oriented-PDC，CPDC+APDC+RPDC 仅在 neck 后提取特征并 zero-init 残差，不使用 edge 辅助监督 | 未超过 VA-RDH；出现 1 个 non-finite batch；不扩展 30 epoch |
| 13 | 2026-07-24 | edge_orientedpdc_f1 | ConvNeXt-Tiny | itksnap(--head edge --edge-pdc-types cpdc,apdc,rpdc --edge-soft) | f1 | 30 | sweep macro=0.7841; dice_1=0.7926; dice_2=0.7756 | runs/edge_orientedpdc_f1/f1/checkpoints/ | 新增 Oriented-PDC: APDC角向+RPDC径向差分卷积并联组(PiDiNet全集) src/bs/edge.py | 边缘Wave2唯一正向且最优; 超baseline+0.42pp, 稀有类dice_2+1.24pp; ≈普通卷积参数量 |
| 14 | 2026-07-24 | edge_gac_f1 | ConvNeXt-Tiny | itksnap(--head gac --edge-soft) | f1 | 30 | sweep macro=0.7717; dice_1=0.7727; dice_2=0.7707 | runs/edge_gac_f1/f1/checkpoints/ | 新增 GAC 测地主动轮廓边界演化头(边缘停止曲率流+边缘平流) src/bs/edge.py | 诚实负结果 -0.82pp; ep10见顶后dice_1走低; 边界或被吸附到血管等非病灶强边缘 |
| 15 | 2026-07-24 | edge_eoc_f1 | ConvNeXt-Tiny | itksnap(--head conv --eoc-weight 0.1) | f1 | 30 | sweep macro=0.7760; dice_1=0.7943; dice_2=0.7577 | runs/edge_eoc_f1/f1/checkpoints/ | 新增 EOC 边缘方向一致性损失(无参数, src/bs/multilabel.py) | 诚实负结果 -0.39pp; 轻损稀有类dice_2; 可降权重复验 |
| 16 | 2026-07-25 | dual_branch_full_f1 | ConvNeXt-Tiny | itksnap(--head dual_branch --edge-pdc-types cpdc,apdc,rpdc) | f1 | 30 | sweep macro=0.7854; dice_1=0.7876; dice_2=0.7832 | runs/dual_branch_full_f1/f1/checkpoints/ | 新增 CSD-DB 源点-轮廓双分支检测头: source/core分支 + Oriented-PDC contour分支 + source-inside一致性; src/bs/dual_branch.py | 正向结果: 超conv+0.55pp; f1训练侧dice_2最高0.7832; 但macro低于VA-RDH K16 0.7872; 202 tests passed |
| 17 | 2026-07-25 | dual_branch_source_only_f1 / dual_branch_edge_only_f1 / dual_branch_light_consistency_f1 | ConvNeXt-Tiny | itksnap(--head dual_branch variants) | f1 | 30 | source-only=0.7771; edge-only=0.7758; light-consistency=0.7800 | runs/dual_branch_*_f1/f1/checkpoints/ | CSD-DB 消融: 只开source / 只开contour / 降低consistency=0.05 | 单分支均不如full; 说明source+contour联合有效; light consistency仍不如full |
| 18 | 2026-07-25 | rdh_dual_branch_f1 | ConvNeXt-Tiny | itksnap(--head rdh_dual_branch; VA-RDH K16 + CSD-DB aux) | f1 | 30 | sweep macro=0.7786; dice_1=0.7941; dice_2=0.7632 | runs/rdh_dual_branch_f1/f1/checkpoints/ | 尝试融合 VA-RDH 主物理头 + CSD-DB source/contour辅助检测 + zero-init residual | 诚实负结果: 低于 VA-RDH K16 0.7872 与 CSD-DB 0.7854; 辅助分支可能干扰 RDH 共享表征优化 |
| 19 | 2026-07-31 | dmt_dice2_stable_noamp_f1_from_best_e12 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_dmt_dice2_stable.yaml | f1 | 30 | best_macro_dice=0.7459; dice_1=0.7593; dice_2=0.7325 | outputs/runs/dmt_dice2_stable_noamp_f1_from_best_e12/f1/checkpoints/ | 新增并稳定 DMT-Dice2: 多方向多尺度差分矩阵边界 Dice2 + 轻量闭合项; 空边界通道跳过与 non-finite 防护; no AMP 低LR继续训练 | best_epoch=25; 从污染run安全best epoch12初始化; best/latest无NaN; 原AMP版本epoch17 NaN |
| 20 | 2026-08-01 | vsubr_w010_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr.yaml | f1 | 30 | fixed ep25 macro=0.7769 dice_1=0.7868 dice_2=0.7671; sweep ep24 macro=0.7861 dice_1=0.7968 dice_2=0.7754 | runs/vsubr_w010_f1/f1/checkpoints/ | 新增 VS-UBR: 血管抑制的不确定性边界增强 loss; 图像梯度 + 结构张量 vesselness 抑制 + 预测不确定性聚焦边界 | 正向结果: sweep macro 略高于 edge_soft_dou03; lesion_2 明显高于 baseline; 适合作为“FA血管干扰边界”故事主线 |
| 21 | 2026-08-01 | va_rdh_vsubr_w010_f1 | ConvNeXt-Tiny + VA-RDH | configs/dinov3_convnext_tiny_va_rdh_vsubr.yaml | f1 | 30 | fixed ep19 macro=0.7680 dice_1=0.7759 dice_2=0.7600; sweep ep19 macro=0.7803 dice_1=0.7892 dice_2=0.7714 | runs/va_rdh_vsubr_w010_f1/f1/checkpoints/ | 新方案探索: 将 VA-RDH 血管锚定各向异性扩散头与 VS-UBR 血管抑制不确定边界 loss 结合 | 负结果: 低于单独 VS-UBR sweep 0.7861 与 VA-RDH K16 0.7872; 两种血管结构先验可能目标冲突; 训练早期/ep10/ep26 有少量 non-finite grad skip |
| 22 | 2026-08-01 | vsubr_ucacl_w003_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_ucacl.yaml | f1 | 30 | fixed ep17 macro=0.7748 dice_1=0.7894 dice_2=0.7602; sweep ep14 macro=0.7842 dice_1=0.7967 dice_2=0.7717 | runs/vsubr_ucacl_w003_f1/f1/checkpoints/ | 新增 UCACL: 无额外 head 的临床面积负担校准 loss; 在 VS-UBR 基础上用 log-scaled per-image lesion area fraction 约束概率质量 | 轻微负结果: 低于单独 VS-UBR sweep 0.7861; 面积校准过强压低 lesion_2; metrics CSV 缺列问题已修复，本 run 字段正常 |
| 23 | 2026-08-02 | vsubr_ucacl_w001_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_ucacl_lite.yaml | f1 | 30 | fixed ep25 macro=0.7620 dice_1=0.7857 dice_2=0.7383; sweep ep16 macro=0.7797 dice_1=0.7915 dice_2=0.7680 | runs/vsubr_ucacl_w001_f1/f1/checkpoints/ | UCACL-lite: 保留 VS-UBR, 将 clinical_area weight 从 0.03 降至 0.01 且 channel_weights 从 [1,2] 降至 [0.5,1.0] | 负结果: 仍低于 VS-UBR sweep 0.7861 且 lesion_2 明显下降; 面积校准线暂时放弃, 转向新架构/局部细化方案 |
| 24 | 2026-08-02 | vsubr_hfcrop_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_hfcrop.yaml | f1 | 30 | fixed ep18 macro=0.7716 dice_1=0.7864 dice_2=0.7568; shared sweep ep26 macro=0.7798 dice_1=0.7947 dice_2=0.7649; ind sweep ep20 macro=0.7815 dice_1=0.7952 dice_2=0.7678 | runs/vsubr_hfcrop_f1/f1/checkpoints/ | 新增 HFCrop: 高荧光候选区域引导的 ROI resized crop 训练增强; 与 VS-UBR 组合模拟粗到细局部细化 | 负结果: 低于单独 VS-UBR sweep 0.7861; 高荧光 ROI crop 未改善小病灶, 可能打乱全局FA上下文; 转向 CSD-DB + VS-UBR 组合 |
| 25 | 2026-08-02 | vsubr_dual_branch_f1 | ConvNeXt-Tiny + CSD-DB | configs/dinov3_convnext_tiny_dual_branch_vsubr.yaml | f1 | 30 | fixed ep29 macro=0.7654 dice_1=0.7748 dice_2=0.7559; shared sweep ep29 macro=0.7794 dice_1=0.7948 dice_2=0.7639; ind sweep ep7 macro=0.7825 dice_1=0.7896 dice_2=0.7754 | runs/vsubr_dual_branch_f1/f1/checkpoints/ | 新组合: 将 CSD-DB 源点/轮廓双分支 head 与 VS-UBR 血管抑制不确定边界 loss 结合 | 负结果: 低于单独 VS-UBR sweep 0.7861 与原 CSD-DB 0.7854; aux 分支可能过强干扰主分割, ep1/2/5 有少量 non-finite grad skip; 下一步降辅助权重 |
| 26 | 2026-08-02 | vsubr_dual_branch_light_f1 | ConvNeXt-Tiny + CSD-DB-lite | configs/dinov3_convnext_tiny_dual_branch_vsubr_light.yaml | f1 | 30 | fixed ep12 macro=0.7639 dice_1=0.7821 dice_2=0.7458; shared sweep ep14 macro=0.7797 dice_1=0.7881 dice_2=0.7714; ind sweep ep14 macro=0.7822 dice_1=0.7881 dice_2=0.7763 | runs/vsubr_dual_branch_light_f1/f1/checkpoints/ | CSD-DB + VS-UBR 轻量辅助版: edge/source 权重降至 0.2/0.1, dice 权重降至 0.25, consistency 降至 0.03 | 负结果: 仍低于单独 VS-UBR sweep 0.7861 与原 CSD-DB 0.7854; 降权未解决双分支与 VS-UBR 主目标冲突; 转向更简单 Oriented-PDC + VS-UBR |
| 27 | 2026-08-02 | vsubr_edge_orientedpdc_f1 | ConvNeXt-Tiny + Oriented-PDC | configs/dinov3_convnext_tiny_edge_vsubr.yaml | f1 | 30 | fixed ep4 macro=0.7685 dice_1=0.7630 dice_2=0.7740; shared sweep ep23 macro=0.7810 dice_1=0.7938 dice_2=0.7682; ind sweep ep23 macro=0.7818 dice_1=0.7938 dice_2=0.7699 | runs/vsubr_edge_orientedpdc_f1/f1/checkpoints/ | 将 Oriented-PDC 边缘 head 与 VS-UBR 组合: APDC/RPDC/CPDC 显式方向边缘检测 + 血管抑制不确定边界主损失 | 负结果: 低于单独 VS-UBR sweep 0.7861 与原 Oriented-PDC 0.7841; 显式边缘分支与 VS-UBR 边界正则冗余; 转向 VS-UBR 自身权重消融 |
| 28 | 2026-08-02 | vsubr_w005_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_w005.yaml | f1 | 30 | fixed ep24 macro=0.7687 dice_1=0.7822 dice_2=0.7553; shared sweep ep22 macro=0.7789 dice_1=0.7926 dice_2=0.7651; ind sweep ep22 macro=0.7789 dice_1=0.7926 dice_2=0.7651 | runs/vsubr_w005_f1/f1/checkpoints/ | VS-UBR 权重消融: 将 vsubr weight 从 0.1 降至 0.05, 其余保持 vsubr_w010 一致 | 负结果: 低于单独 VS-UBR w0.10 sweep 0.7861; 0.05 边界约束偏弱, VS-UBR 增益下降; ep1/9/13 有少量 non-finite grad skip |
| 29 | 2026-08-02 | vsubr_w015_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_w015.yaml | f1 | 30 | fixed ep8 macro=0.7703 dice_1=0.7817 dice_2=0.7589; shared sweep ep24 macro=0.7769 dice_1=0.7943 dice_2=0.7595; ind sweep ep24 macro=0.7788 dice_1=0.7948 dice_2=0.7628 | runs/vsubr_w015_f1/f1/checkpoints/ | VS-UBR 权重消融: 将 vsubr weight 从 0.1 提高至 0.15, 其余保持 vsubr_w010 一致 | 负结果: 低于 VS-UBR w0.10 sweep 0.7861, 也未超过 w0.05; 0.15 约束偏强且 lesion_2 下降; ep1/3 有 non-finite grad skip; 0.10 暂定甜点权重 |
| 30 | 2026-08-02 | vsubr_vw05_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f1 | 30 | fixed ep24 macro=0.7815 dice_1=0.7814 dice_2=0.7815; shared sweep ep24 macro=0.7902 dice_1=0.7976 dice_2=0.7828; ind sweep ep24 macro=0.7902 dice_1=0.7976 dice_2=0.7828 | runs/vsubr_vw05_f1/f1/checkpoints/ | VS-UBR vessel suppression 消融: 保持 vsubr weight=0.1, 将 vessel_weight 从 1.0 降至 0.5 | 正向结果: 超过 VS-UBR w0.10 sweep 0.7861, lesion_2 提升到 0.7828; 说明原始强血管抑制可能过度压制邻血管渗漏边界; 下一步测试更弱抑制 0.25 |
| 31 | 2026-08-02 | vsubr_vw025_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw025.yaml | f1 | 30 | fixed ep24 macro=0.7711 dice_1=0.7770 dice_2=0.7652; shared sweep ep22 macro=0.7834 dice_1=0.7911 dice_2=0.7757; ind sweep ep22 macro=0.7842 dice_1=0.7911 dice_2=0.7773 | runs/vsubr_vw025_f1/f1/checkpoints/ | VS-UBR vessel suppression 消融: 保持 vsubr weight=0.1, 将 vessel_weight 从 0.5 继续降至 0.25 | 负结果: 低于 vessel_weight=0.5 的 sweep 0.7902; 抑制过弱会重新放大血管边缘干扰, 0.5 目前是较优平衡点; 下一步测试中间值 0.75 |
| 32 | 2026-08-02 | vsubr_vw075_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw075.yaml | f1 | 30 | fixed ep17 macro=0.7746 dice_1=0.7813 dice_2=0.7679; shared sweep ep22 macro=0.7876 dice_1=0.7907 dice_2=0.7846; ind sweep ep22 macro=0.7886 dice_1=0.7907 dice_2=0.7866 | runs/vsubr_vw075_f1/f1/checkpoints/ | VS-UBR vessel suppression 消融: 保持 vsubr weight=0.1, 测试 vessel_weight=0.75 | 次优正向: 高于原 VS-UBR w0.10 sweep 0.7861, 但低于 vessel_weight=0.5 的 0.7902; 消融呈 0.25<1.0<0.75<0.5 的倒U趋势, 0.5 暂定主结果 |
| 33 | 2026-08-02 | vsubr_vw05_f2 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f2 | 30 | fixed ep9 macro=0.7827 dice_1=0.8221 dice_2=0.7433; shared sweep ep9 macro=0.7827 dice_1=0.8221 dice_2=0.7433; ind sweep ep9 macro=0.7835 dice_1=0.8221 dice_2=0.7448 | runs/vsubr_vw05_f2/f2/checkpoints/ | 当前最佳 VS-UBR 设置跨折验证: vsubr weight=0.1, vessel_weight=0.5 | f2 结果稳定且较强, 支持继续 f3-f5 汇总 5 折; ep1/5/10 有少量 non-finite grad skip |
| 34 | 2026-08-02 | vsubr_vw05_f3 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f3 | 30 | fixed ep17 macro=0.8119 dice_1=0.8100 dice_2=0.8138; shared sweep ep17 macro=0.8164 dice_1=0.8134 dice_2=0.8193; ind sweep ep17 macro=0.8190 dice_1=0.8134 dice_2=0.8246 | runs/vsubr_vw05_f3/f3/checkpoints/ | 当前最佳 VS-UBR 设置跨折验证: vsubr weight=0.1, vessel_weight=0.5 | f3 表现很强, 明显支持继续 f4-f5 汇总 5 折; ep1/3/16 有少量 non-finite grad skip |
| 35 | 2026-08-02 | vsubr_vw05_f4 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f4 | 30 | fixed ep20 macro=0.7484 dice_1=0.7877 dice_2=0.7092; shared sweep ep20 macro=0.7557 dice_1=0.7980 dice_2=0.7134; ind sweep ep20 macro=0.7581 dice_1=0.7980 dice_2=0.7183 | runs/vsubr_vw05_f4/f4/checkpoints/ | 当前最佳 VS-UBR 设置跨折验证: vsubr weight=0.1, vessel_weight=0.5 | f4 低于 f1-f3 但仍给出可用跨折结果, lesion_2 是主要瓶颈; ep1/2/27/29 有少量 non-finite grad skip |
| 36 | 2026-08-02 | vsubr_vw05_f5 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f5 | 30 | fixed ep24 macro=0.7855 dice_1=0.8127 dice_2=0.7582; shared sweep ep24 macro=0.7895 dice_1=0.8209 dice_2=0.7580; ind sweep ep24 macro=0.7896 dice_1=0.8209 dice_2=0.7582 | runs/vsubr_vw05_f5/f5/checkpoints/ | 当前最佳 VS-UBR 设置跨折验证: vsubr weight=0.1, vessel_weight=0.5 | f5 稳定正向, ep1/5/10 有少量 non-finite grad skip; 可与 f1-f4 汇总形成 5 折主结果 |
| 37 | 2026-08-02 | vsubr_vw05_5fold_summary | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_vsubr_vw05.yaml | f1-f5 | 30 | fixed macro=0.7820±0.0202 dice_1=0.8028 dice_2=0.7612; shared sweep macro=0.7869±0.0194 dice_1=0.8104 dice_2=0.7634; ind sweep macro=0.7881±0.0194 dice_1=0.8104 dice_2=0.7657 | runs/vsubr_vw05_f{1..5}/ | VS-UBR 主结果: vsubr weight=0.1 且 vessel_weight=0.5, 基于 f1 消融最优后完成 f1-f5 跨折验证 | 5 折结果支持 moderate vessel suppression 的主叙事; f3 最强, f4 lesion_2 偏低拉低均值; 可作为论文主实验与倒U消融一起呈现 |
| 38 | 2026-08-10 | dino_sam_refiner_vsubr_vw05_mvp | ConvNeXt-Tiny VS-UBR + DINO-SAM Refiner | configs/dino_sam_refiner.yaml | f1-f5 | 20 | click0 macro=0.7809±0.0193 dice_1=0.7982 dice_2=0.7636; click1 macro=0.7853±0.0195 dice_1=0.8012 dice_2=0.7694; click3 macro=0.7918±0.0200 dice_1=0.8072 dice_2=0.7764; click5 macro=0.7965±0.0201 dice_1=0.8106 dice_2=0.7824 | outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_mvp/f{1..5}/checkpoints/ | 新增交互式 SAM-DINO 细化系统: 缓存 VS-UBR/DINO 粗预测与 SAM prompt 先验, 用 13通道 refiner 融合 image/coarse masks/click maps/residual maps 做多轮点击细化 | 5折已完成; click 数从0到5稳定提升 +1.56pp macro; f1/f2/f3/f4/f5 click5 macro=0.7887/0.8005/0.8272/0.7653/0.8010; 支持“医生交互驱动粗到细精修系统”叙事 |
| 39 | 2026-08-10 | convnext_tiny_baseline_tta_f1 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_baseline_tta_f1.yaml | f1 | 30 | fixed ep24 macro=0.7624 dice_1=0.7736 dice_2=0.7511; shared sweep ep24 macro=0.7827 dice_1=0.8006 dice_2=0.7648; ind sweep ep24 macro=0.7827 dice_1=0.8006 dice_2=0.7648 | runs/convnext_tiny_baseline_tta_f1/f1/checkpoints/ | TTA 消融: 在 ConvNeXt-Tiny baseline 上仅启用验证/推理 flip TTA(h/v/hv, scale=1.0), 不启用 morphology/FOV/intensity 后处理; 训练增强与非TTA baseline 保持一致 | 已完成; latest ep30 sweep=0.7750; 对比非TTA baseline 未观察到稳定涨点 |
| 40 | 2026-08-10 | rdh_pde_tta_f1 | ConvNeXt-Tiny + RDH-PDE | configs/dinov3_convnext_tiny_rdh_pde_tta_f1.yaml | f1 | 30 | fixed ep24 macro=0.7639 dice_1=0.7700 dice_2=0.7578; shared sweep ep16 macro=0.7865 dice_1=0.7968 dice_2=0.7762; ind sweep ep16 macro=0.7865 dice_1=0.7968 dice_2=0.7762 | runs/rdh_pde_tta_f1/f1/checkpoints/ | TTA 消融: 在 RDH-PDE isotropic 头上仅启用验证/推理 flip TTA(h/v/hv, scale=1.0), 不启用额外后处理 | 已完成; latest ep30 sweep=0.7793; TTA 未明显超过历史 RDH-PDE/VA-RDH f1 最优结果 |
| 41 | 2026-08-10 | edge_pdc_tta_f1 | ConvNeXt-Tiny + Edge Oriented-PDC | configs/dinov3_convnext_tiny_edge_pdc_tta_f1.yaml | f1 | 30 | fixed ep26 macro=0.7667 dice_1=0.7701 dice_2=0.7633; shared sweep ep23 macro=0.7867 dice_1=0.7976 dice_2=0.7759; ind sweep ep23 macro=0.7867 dice_1=0.7976 dice_2=0.7759 | runs/edge_pdc_tta_f1/f1/checkpoints/ | TTA 消融: 在 Edge-PDC(cpdc/apdc/rpdc) 头上仅启用验证/推理 flip TTA(h/v/hv, scale=1.0), 不启用额外后处理 | 已完成; latest ep30 sweep=0.7849; 与非TTA Edge-PDC f1 结果接近, 暂无明确涨点 |
| 42 | 2026-08-11 | convnext_tiny_baseline_clean_f2_f5 + diffleak_f1_baseline_clean | ConvNeXt-Tiny + FPN baseline | configs/dinov3_convnext_tiny_multilabel_itksnap.yaml | f1-f5 | 30 | fixed f2-f5 macro=0.7765/0.8116/0.7403/0.7822; shared sweep 5折 macro=0.7849±0.0219 dice_1=0.8080 dice_2=0.7618; f1/f2/f3/f4/f5 macro=0.7769/0.7845/0.8215/0.7536/0.7881 | runs/diffleak_f1_baseline_clean/f1/checkpoints/ + runs/convnext_tiny_baseline_clean_f2_f5/{f2..f5}/checkpoints/ | 无代码改动; 按当前 clean 验证集口径补训基础 ConvNeXt-Tiny + FPN baseline 的 f2-f5, head=conv, bs=8, 768x768 | 所有 fold 最优 shared sweep 阈值均为 0.9; 可替代历史 f1-f4 均值0.7710作为正式五折 baseline |
| 43 | 2026-08-13 | edge_cpdc_soft_5fold_20260813 | ConvNeXt-Tiny + CPDC edge head | configs/dinov3_convnext_tiny_edge.yaml | f1-f5 | 30 | fixed macro=0.7747±0.0206 dice_1=0.7962 dice_2=0.7533; shared sweep macro=0.7853±0.0195 dice_1=0.8136 dice_2=0.7570; f1/f2/f3/f4/f5 sweep macro=0.7751/0.7864/0.8159/0.7569/0.7920 | runs/edge_cpdc_soft_5fold_20260813/{f1..f5}/checkpoints/ | 无代码改动; 复跑 CPDC 单一中心像素差分边缘头五折, head=edge, edge_pdc_types=cpdc, edge_soft=true, bs=8, 768x768 | 与正式 ConvNeXt baseline 0.7849±0.0219 基本打平, macro +0.04pp; dice_1 +0.56pp 但 dice_2 -0.48pp; f5 最优阈值0.8/0.8, 其余0.9/0.9 |
| 44 | 2026-09-21 | nnunet2d_clean_f1_30 + resume | official nnU-Net v2 PlainConvUNet 2D | configs/nnunet_2d_multilabel_itksnap.yaml | f1 | 30 | fixed ep28 macro=0.7298 dice_1=0.7304 dice_2=0.7293; shared/ind sweep ep28 macro=0.7571 dice_1=0.7685 dice_2=0.7457; peak VRAM=6.84GB | runs/nnunet2d_clean_f1_resume_final_30/f1/checkpoints/ | official dynamic-network-architectures PlainConvUNet; 6 stages [32,64,128,256,320,320], InstanceNorm + LeakyReLU; 与后续 AAAG/U-KAN 共用 clean multilabel 协议 | 负对照: 比 ConvNeXt clean f1 sweep=0.7769 低1.98pp; 从头训练 encoder 在当前小样本 FA 数据上弱于 DINOv3 预训练 encoder；epoch11 后按完整 optimizer/scaler/scheduler 状态续训，最终 best checkpoint 可复核 |
| 45 | 2026-09-21 | hmvf_mambavision_f1_30 | DINOv3 ConvNeXt-Tiny + Hybrid MambaVision Feature Fusion | configs/dinov3_convnext_tiny_hmvf.yaml | f1 | 30 | best ep21 fixed macro=0.7685 dice_1=0.7854 dice_2=0.7516; shared/ind sweep ep21 macro=0.7802 dice_1=0.7979 dice_2=0.7626, threshold=0.9/0.9; peak VRAM=12.21GB | runs/hmvf_mambavision_f1_30/f1/checkpoints/ | 在 FPN fused decoder feature 上并联 depthwise local branch 与低分辨率四向 selective-SSM global branch，以学习门控融合；输出投影 zero-init，初始等价于 baseline | 轻微正向: 相对 clean ConvNeXt f1 sweep=0.7769 提升0.34pp，超过旧 S3RD Mamba=0.7807 约0.01pp，但低于 VS-UBR f1=0.7902；需要跨折验证确认稳定性 |
| 46 | 2026-09-21 | aaag_nnunet2d_clean_f1_30 | AAAG-UNet 2D (official nnU-Net + axial attention + attention-gated skips) | configs/aaag_nnunet_2d_multilabel_itksnap.yaml | f1 | 30 | best ep18 fixed macro=0.7390 dice_1=0.7621 dice_2=0.7158; shared sweep ep18 macro=0.7443 dice_1=0.7684 dice_2=0.7202 at threshold=0.8; independent sweep macro=0.7456 dice_1=0.7684 dice_2=0.7228 at threshold=0.8/0.9; peak VRAM=11.19GB | runs/aaag_nnunet2d_clean_f1_30/f1/checkpoints/ | Medical Physics 2026 AAAG-UNet 的 2D 适配: pooled axial attention、spatial/channel calibrated attention-gated skip；与 official nnU-Net 共用 clean multilabel 协议 | 负结果: independent sweep 比 official nnU-Net baseline=0.7571 低1.15pp，主要损失来自 lesion_2（-2.29pp），虽 fixed macro 比 nnU-Net fixed=0.7298 高0.91pp；比 DINOv3 ConvNeXt clean sweep=0.7769 低3.13pp，完整30轮 best checkpoint 可复核 |
| 47 | 2026-09-23 | upen_pelit_{baseline,pe,pehmvf}_f1_screen5 + hmvf_mambavision_f1_screen5 | DINOv3 ConvNeXt-Tiny + U-PEN Mamba PEL-3 启发的 PE-lite 消融 | configs/dinov3_convnext_tiny_hmvf.yaml + `--decoder-attention {none,pe-lite,mambavision,pe-hmvf}` | f1 | 5 | matched baseline ep4: d1=0.7610 d2=0.7661 macro=0.7636; PE-lite ep5: 0.7614/0.7258/0.7436; HMVF ep5: 0.7599/0.7377/0.7488; PE-lite+HMVF ep4: 0.7525/0.7446/0.7485; peak VRAM=10.54/13.34/12.21/15.01GB | runs/upen_pelit_baseline_f1_screen5/; runs/upen_pelit_pe_f1_screen5/; runs/hmvf_mambavision_f1_screen5/; runs/upen_pelit_pehmvf_f1_screen5/ | 对 fused FPN feature 做归一化、限幅后拼接一至三阶项，zero-init residual；并与已有 HMVF 做 2×2 组合筛选；附加参数 0/178081/117553/295634 | 5 epoch clean f1 负筛选：PE-lite 比 matched baseline 低2.00pp，主要损伤 lesion_2；PE+HMVF 比 baseline 低1.50pp，且与 HMVF macro 基本相同但资源更高；不扩展30 epoch/5折，不进入默认配置 |

### 2026-08-10 TTA 有/无对比总结

本组实验只在验证/推理阶段启用 flip TTA（h/v/hv, scale=1.0），不改变训练增强，不叠加 morphology、FOV mask 或 intensity refine 等后处理。对比口径优先使用 f1 的 best threshold sweep macro Dice。

| 模型 | 无 TTA f1 sweep macro | TTA f1 sweep macro | Δ macro | 对比结论 |
|------|----------------------:|-------------------:|--------:|----------|
| ConvNeXt-Tiny baseline | 0.7799 | 0.7827 | +0.28pp | 仅轻微提升，latest ep30 回落到 0.7750，收益不稳定 |
| RDH-PDE isotropic | 0.7828 | 0.7865 | +0.37pp | 有小幅提升，但未超过 VA-RDH K16 f1=0.7872，不能作为主要涨点证据 |
| Edge Oriented-PDC | 0.7841 | 0.7867 | +0.26pp | 与非 TTA 接近，增益很小，基本属于阈值/训练波动范围 |

> 结论: 单独 flip TTA 在三类模型上均只有 0.26-0.37pp 的小幅提升，且 latest checkpoint 普遍回落，说明 TTA 可作为推理阶段轻量校准或补充消融，但不适合作为论文主创新点。后续若要使用 TTA，建议只在最终模型评估中作为可选 inference trick 报告，并与 morphology/FOV 等后处理分开消融。

---

## 待验证实验：ConvNeXt + TTA + 形态学后处理

### 2026-07-07 实现记录

**目标**: 在当前最强的 DINOv3 ConvNeXt-Tiny 多标签 baseline 上，不改 backbone、不重训或少重训，通过验证/推理阶段校准提升 Dice，并形成可写入论文的小创新点。

**方案名**: FA-Calibrated Morphological FOV-TTA (FCM-FOV-TTA)

**代码与配置**

| 文件 | 作用 |
|------|------|
| `src/bs/tta.py` | 验证/推理阶段 h/v/hv flip/scale/appearance TTA，自动反变换 logits 后平均；支持 tuple 输出模型；可选不确定性惩罚 UATTA |
| `src/bs/adaptive_threshold.py` | Adaptive Probability-Quantile Thresholding (APQT)：按单张图/单通道概率分位数自适应调整阈值，并支持与固定阈值 blend/clamp |
| `src/bs/postprocess.py` | 阈值后形态学先验：closing、opening、小连通域过滤、小孔洞填充；支持高低阈值滞后连通域、置信度引导的小连通域保留、形状先验过滤和 top-component 过滤 |
| `src/bs/fov.py` | 从 FA 原图估计 retinal field-of-view，有效视野外预测清零；可选 FECS 视野边缘组件过滤，减少贴边伪影假阳性 |
| `src/bs/intensity_refine.py` | FA intensity-guided refinement (FAIGR)：用原始荧光亮度的组件级统计过滤低亮度假阳性，并允许高置信度组件救援 |
| `src/bs/preprocess.py` | FA Local Contrast Enhancement (FA-LCE)：训练/验证读图阶段按局部背景残差增强高荧光渗漏候选 |
| `src/bs/convnext_seg.py` | ConvNeXt decoder 可选 CBAM-style channel-spatial attention gate 和多尺度辅助监督，用于训练侧模块融合消融 |
| `src/bs/multilabel.py` | 可选 Boundary-weighted BCE：在 Asymmetric Focal Tversky + BCE 中提高病灶边界带权重 |
| `scripts/train_dinov3_multilabel.py` | 验证指标、threshold sweep、样例保存接入 TTA + morphology + FOV mask；训练损失不受影响 |
| `scripts/evaluate_dinov3_postprocess.py` | 对已有 checkpoint 一次性离线消融 default / threshold / TTA / morphology / FOV，无需重训 |
| `scripts/evaluate_ensemble_postprocess.py` | 对 ConvNeXt / ViT 等多个 checkpoint 做概率或 logit 集成评估，支持 FOV mask |
| `scripts/run_fcm_tta_5fold_ablation.sh` | 对已有 5 折 ConvNeXt checkpoint 批量跑 FCM-TTA 消融 |
| `scripts/summarize_postprocess_eval.py` | 汇总每折 postprocess JSON，输出 5 折均值 CSV/Markdown/JSON |
| `scripts/search_morphology_postprocess.py` | 在已有 checkpoint 上联合搜索 lesion-specific threshold + 形态学参数 + FOV，输出 top-k 和可复制 YAML 片段 |
| `scripts/log_eval_result.py` | 从 summary/ensemble/search JSON 生成 EXPERIMENT_LOG 表格行，显式 `--append` 才写入 |
| `configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml` | ConvNeXt-Tiny + lesion-specific threshold `[0.50, 0.90]` + TTA + morphology + FOV mask 实验配置 |

**当前默认后处理参数**

```yaml
metric:
  threshold: [0.50, 0.90]
  threshold_sweep:
    enabled: true
    postprocess: false  # sweep 默认只做原始概率阈值校准，避免每个阈值都跑连通域
    fov_mask: false
  adaptive_threshold:
    enabled: false
    method: quantile
    quantile: [0.0, 0.0]
    blend: [1.0, 1.0]
    min_threshold: [0.0, 0.0]
    max_threshold: [1.0, 1.0]
  tta:
    enabled: true
    flips: [h, v, hv]
    scales: [1.0]
    size_multiple: 32
    uncertainty_penalty: [0.0, 0.0]
    appearance_preprocess:
      enabled: false
      mode: fa_lce
      channel_reduce: max
      kernel_size: 31
      strength: 0.25
      quantile: 0.99
      reference_threshold: 0.03
  postprocess:
    enabled: true
    close_kernel: [3, 3]
    open_kernel: [0, 0]
    hysteresis_seed_threshold: [0.0, 0.0]
    hysteresis_min_seed_pixels: [1, 1]
    min_component_area: [64, 16]
    small_component_min_mean_prob: [0.0, 0.0]
    small_component_min_max_prob: [0.0, 0.0]
    min_component_mean_prob: [0.0, 0.0]
    min_component_prob_mass: [0.0, 0.0]
    max_component_aspect_ratio: [0.0, 0.0]
    min_component_extent: [0.0, 0.0]
    lesion2_support_dilation_kernel: 0
    lesion2_min_support_pixels: 0
    lesion2_min_support_fraction: 0.0
    lesion2_support_threshold: 0.0
    max_components: [0, 0]
    component_score: area
    fill_holes_max_area: [128, 64]
    connectivity: 8
  intensity_refine:
    enabled: false
    input_mode: imagenet
    channel_reduce: max
    reference_threshold: 0.03
    min_component_mean_quantile: [0.0, 0.0]
    min_component_max_quantile: [0.0, 0.0]
    contrast_kernel: 0
    min_component_mean_contrast_quantile: [0.0, 0.0]
    min_component_max_contrast_quantile: [0.0, 0.0]
    rescue_min_max_prob: [0.0, 0.0]
  fov_mask:
    enabled: true
    input_mode: imagenet
    threshold: 0.03
    close_kernel: 15
    min_component_area: 4096
    fill_holes_max_area: 262144
    keep_largest: true
    fallback_full_if_empty: true
    border_erode_kernel: 0
    border_min_inner_pixels: [0, 0]
    border_min_inner_fraction: [0.0, 0.0]
    border_rescue_min_mean_prob: [0.0, 0.0]
    border_rescue_min_max_prob: [0.0, 0.0]
    connectivity: 8
preprocess:
  enabled: false
  mode: fa_lce
  channel_reduce: max
  kernel_size: 31
  strength: 0.35
  quantile: 0.99
  reference_threshold: 0.03
```

**可选训练侧消融：Boundary-weighted BCE (BAW-BCE)**

在 `AsymmetricFocalTverskyBCE` 的 BCE 分支上加入边界带权重，目标是减少渗漏边界断裂和边缘模糊导致的 Dice 损失。默认 `boundary_weight=0.0`，旧实验不受影响。建议先跑单折：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_baw_morph_tta_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --boundary-weight 2.0 --boundary-kernel 5
```

如果边界加权训练本身提升不明显，也可以仅保留 FCM-TTA 作为后处理创新点；两者在论文里可以拆成独立消融。

**可选训练侧消融：Boundary Dice Auxiliary Loss (BDL)**

BAW-BCE 是在 BCE 分支里提高边界带权重；BDL 则单独从 `sigmoid(logits)` 和 GT mask 提取软边界带，并对预测边界和真实边界计算 Dice loss。它直接约束病灶轮廓一致性，目标是减少 FA 渗漏边界模糊、孔洞和断裂对 Dice 的影响。默认 `boundary_dice_weight=0.0`，旧实验不受影响。

单折试验命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_bdl_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --boundary-dice-weight 0.2 --boundary-dice-kernel 5
```

建议先不要和 BAW-BCE 同时打开，避免边界项过强导致模型只学轮廓、内部概率变弱；如果单独有效，再尝试 `--boundary-weight 1.0 --boundary-dice-weight 0.1` 的轻组合。

**可选训练侧消融：Hard-Negative BCE Mining (HNB-BCE)**

FA 泄漏分割的背景像素占绝大多数，普通 BCE 会被大量容易背景像素稀释；而当前 Dice 更容易受小面积假阳性影响。HNB-BCE 在 BCE 分支中保留所有正样本像素，只从每个 batch/通道的背景像素里选 BCE 最大的一部分 hard negatives 参与平均，使训练更关注模型最容易误报的背景区域。默认 `hard_negative_ratio=0.0`，旧 loss 数值完全不变；Tversky 分支仍使用全图，避免只看局部 hard negative 导致召回崩掉。

单折试验命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_hnb_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --hard-negative-ratio 0.25 --hard-negative-min-pixels 2048
```

如果训练早期召回下降明显，可把 `--hard-negative-ratio` 调到 `0.50`；如果假阳性仍多，再尝试 `0.10`。建议与 BAW-BCE 分开单折消融，确认 HNB-BCE 本身是否降低 `paper_pred_pixels_2` 且不损伤 lesion_2 Dice。

**可选训练侧消融：ConvNeXt Decoder CBAM Attention (CSA)**

在 ConvNeXt FPN decoder 的多尺度特征拼接后、原有 `fuse` 分类头前加入轻量 channel-spatial attention gate。该模块通过全局平均/最大池化生成通道注意力，再用平均/最大空间响应生成空间注意力，目标是让 decoder 更关注高响应渗漏区域并压低大面积背景噪声。默认 `decoder_attention: none`，旧 checkpoint 的 `state_dict` 键不变；启用时属于训练侧模块融合消融，需要重新训练。

单折试验命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_cbam_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --decoder-attention cbam --decoder-attention-reduction 16
```

建议先比较 `cbam` 训练得到的 best checkpoint 在同一套 FCM-FOV-TTA/CGMF 后处理下的 final eval；如果单折有效，再跑 5 折。若提升不稳定，论文中仍可把它作为“模块融合尝试未优于后处理先验”的负结果记录。

**可选训练侧消融：ConvNeXt Multi-Scale Auxiliary Supervision (MSAS)**

ConvNeXt backbone 天然输出 4 个尺度的特征，FPN decoder 在 top-down 融合后仍保留多尺度 pyramid。MSAS 在训练阶段对 3 个较粗尺度各接一个 `1x1` 辅助分割头，并把辅助 logits 上采样到原图尺寸参与同一个 multilabel loss；推理阶段只输出主 head，不增加推理开销。目标是让低分辨率语义层更早获得病灶监督，缓解 lesion_2 极小样本下主 head 梯度稀疏的问题。默认 `decoder_deep_supervision: false`，旧 checkpoint 不受影响。

单折试验命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_msas_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --decoder-deep-supervision --aux-loss-weight 0.4
```

也可以和 CBAM 组合成一个模块融合实验，但建议先分别跑单折，确认每个模块的独立贡献：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_cbam_msas_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --decoder-attention cbam \
  --decoder-deep-supervision --aux-loss-weight 0.4
```

**可选训练侧消融：EMA Weight Averaging (EMA-Val)**

EMA 不改变网络结构，而是在训练过程中维护一份参数指数滑动平均副本；每个 epoch 额外用 EMA 权重跑一次验证，并保存 `best_ema.pt` / `latest_ema.pt`。这类权重平均通常能降低小数据集训练抖动，尤其适合比较 raw best 与 EMA best 在同一套 FCM-FOV-TTA/CGMF 后处理下的稳定性。默认 `ema_enabled: false`，旧训练和 checkpoint 选择逻辑不变。

单折试验命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_ema_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --ema --ema-decay 0.999 --ema-start-epoch 1
```

若 `best_ema.pt` 优于 `best.pt`，后续离线评估只需把 checkpoint 路径换成 `runs/<run-name>/f1/checkpoints/best_ema.pt`；评估脚本仍读取 checkpoint 内的 `model` 权重，不需要额外参数。论文叙事中可把它定位为训练稳定化策略，与 CBAM/MSAS 的结构模块消融分开报告。

**可选预处理消融：FA Local Contrast Enhancement (FA-LCE)**

FA 渗漏通常表现为相对局部背景更亮的高荧光区域，而不同眼底图像的整体曝光、黑边和背景亮度差异很大。FA-LCE 在读图阶段先估计局部平均背景，再取 `intensity - local_mean` 的正残差，用图内分位数归一化后温和提升局部高亮区域。它不改 mask、不引入外部模型，属于“成像先验驱动的预处理”。默认 `preprocess.enabled=false`，旧 checkpoint 和旧训练不受影响。

单折训练命令：

```bash
python scripts/train_dinov3_multilabel.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --run-name dinov3_convnext_tiny_fa_lce_f1 \
  --fold f1 --batch-size 8 --grad-accum-steps 1 \
  --preprocess-mode fa_lce \
  --preprocess-kernel 31 \
  --preprocess-strength 0.35 \
  --preprocess-quantile 0.99
```

离线评估同一个 checkpoint 时必须使用相同预处理参数，否则输入分布不一致：

```bash
python scripts/evaluate_dinov3_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_fa_lce_f1/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 --ablation-suite \
  --preprocess-mode fa_lce \
  --preprocess-kernel 31 \
  --preprocess-strength 0.35 \
  --preprocess-quantile 0.99 \
  --output-json runs/dinov3_convnext_tiny_fa_lce_f1/f1/postprocess_eval.json
```

若 FA-LCE 单折有效，再跑 5 折并继续接同一套 FCM-FOV-TTA/CGMF/HSCF 搜索；若提升不稳定，可把它作为“预处理成像先验尝试”的消融，不并入最终主方法。

FA-LCE checkpoint 做 5 折 CGMF/HSCF 搜索时，需要把预处理参数透传给搜索脚本：

```bash
PREPROCESS_MODE=fa_lce \
PREPROCESS_KERNEL=31 \
PREPROCESS_STRENGTH=0.35 \
PREPROCESS_QUANTILE=0.99 \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_fa_lce_5fold
```

**可选推理侧消融：FA-LCE Appearance TTA (A-TTA)**

训练前 FA-LCE 会改变输入分布，需要重训；A-TTA 则不改 checkpoint，只在推理时把“原图 view”和“FA-LCE 增强 view”一起送入同一个模型并平均 logits。它相当于一种面向 FA 成像的 appearance test-time augmentation：如果模型对局部高荧光增强后的渗漏区域更敏感，平均投票可能提高弱渗漏召回；如果增强导致伪阳性，则可以通过 UATTA/CGMF/HSCF 抑制。默认关闭。

单折离线评估命令：

```bash
python scripts/evaluate_dinov3_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 --ablation-suite \
  --tta-appearance-mode fa_lce \
  --tta-appearance-kernel 31 \
  --tta-appearance-strength 0.25 \
  --tta-appearance-quantile 0.99 \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/postprocess_eval_atta.json
```

5 折 CGMF/HSCF 搜索时可直接透传：

```bash
TTA_APPEARANCE_MODE=fa_lce \
TTA_APPEARANCE_KERNEL=31 \
TTA_APPEARANCE_STRENGTH=0.25 \
TTA_APPEARANCE_QUANTILE=0.99 \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

**可选推理侧消融：Confidence-Guided Morphological Filtering (CGMF)**

普通小连通域过滤只看面积，容易在 lesion_2 极小真阳性和小面积假阳性之间误删。CGMF 在 `min_component_area` 的基础上增加置信度救援规则：小于面积阈值的连通域，如果组件内最大/平均概率足够高，则保留；否则删除。默认配置中该功能关闭，建议在 morphology search 的第二阶段围绕最佳阈值和面积参数搜索。

**可选推理侧消融：Confidence-Mass Component Filtering (CMCF)**

面积过滤只看像素数，HSCF 只要求局部高置信种子，仍可能保留一大片平均置信度偏低的雾状假阳性。CMCF 对每个阈值后二值连通域计算模型概率的均值 `min_component_mean_prob` 和概率质量 `min_component_prob_mass=sum(prob)`；均值门控压低整体不可靠的组件，概率质量门控要求组件同时具备一定面积和置信度。默认 `[0.0,0.0]` 不启用，建议与 CGMF/HSCF 分开小网格搜索，防止误删弱渗漏。

**可选推理侧消融：Hysteresis-Seeded Component Filtering (HSCF)**

单一阈值会在 FA 渗漏边缘和低置信伪阳性之间二选一：阈值低能保留边界和弱渗漏，但容易产生噪声；阈值高能压假阳性，但会切掉连续渗漏的低置信边缘。HSCF 把 `metric.threshold` 当作低阈值生成候选连通域，再要求每个连通域内部至少有 `hysteresis_seed_threshold` 以上的高置信种子像素。这样可保留与高置信核心相连的弱渗漏，同时删除没有高置信支持的低置信孤岛。默认 `hysteresis_seed_threshold=[0.0,0.0]`，不改变旧结果。

单折或 5 折搜索建议把低阈值网格适当下调，再搜索高置信 seed 阈值：

```bash
THRESHOLD_1="0.30 0.35 0.40 0.45 0.50" \
THRESHOLD_2="0.50 0.60 0.70 0.80 0.90" \
HYSTERESIS_SEED_THRESHOLD_1="0.0 0.50 0.60" \
HYSTERESIS_SEED_THRESHOLD_2="0.0 0.90 0.95" \
HYSTERESIS_MIN_SEED_PIXELS_1="1" \
HYSTERESIS_MIN_SEED_PIXELS_2="1" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若 HSCF 有效，最终 YAML 里会同时出现低阈值 `metric.threshold` 和高阈值 `metric.postprocess.hysteresis_seed_threshold`。论文中可解释为“高置信种子约束的滞后阈值连通域过滤”，属于概率-几何联合后处理，不需要重训。

**可选推理侧消融：Cross-Lesion Consistency Filtering (CLCF)**

当前多标签定义中 label 3 会同时进入 lesion_1 与 lesion_2，且 lesion_2 极罕见，模型容易在无 lesion_1 支撑的位置产生小块 lesion_2 假阳性。CLCF 在完成各通道 morphology 后，把 lesion_1 预测或高置信 lesion_1 概率图作为支持区域，经过可选膨胀后，要求每个 lesion_2 连通域内部至少有一定数量/比例的 lesion_1 支持像素。默认 `lesion2_min_support_pixels=0` 且 `lesion2_min_support_fraction=0.0`，不启用。该方法应小网格搜索，防止误删独立存在的 lesion_2 真阳性。

**可选推理侧消融：Adaptive Probability-Quantile Thresholding (APQT)**

固定阈值假设每张 FA 图的概率校准一致，但实际图像曝光、渗漏面积和模型置信度分布会明显波动。APQT 在每张图、每个病灶通道上计算概率分位数阈值，再与原固定阈值按 `blend` 融合并用 `min_threshold/max_threshold` 限幅。它适合优先约束 lesion_2 这类极罕见类别的预测负担，例如只让 lesion_2 在概率分布最高的极少数区域通过阈值，同时保留固定阈值作为下限。默认 `enabled=false`，旧评估、可视化和导出配置不受影响。

**可选推理侧消融：FOV-Edge Component Suppression (FECS)**

单纯 FOV mask 只清掉视野外预测，但 FA 图像在视网膜视野边缘常有亮边、睫毛/眼睑遮挡、曝光不均或插值伪影，容易产生贴边小连通域。FECS 在估计 FOV 后再向内腐蚀一圈，把原 FOV 与腐蚀后 inner-FOV 之间的环形区域视为边界风险带；每个预测连通域需要有一定像素数/比例落在 inner-FOV 内，否则删除。为了避免误删真实贴边渗漏，可用 `border_rescue_min_max_prob` 保留模型极高置信度组件。默认 `border_erode_kernel=0`，不启用。

**可选推理侧消融：Top-Component Morphology (TCM)**

如果模型在 lesion_2 上产生多个离散小假阳性，可以在所有形态学处理后仅保留 top-N 连通域。排序依据可选 `area`、`mean_prob`、`max_prob`。默认 `max_components=[0,0]` 不启用；建议只在 lesion_2 上小范围尝试，例如 `max_components=[0,1]` 或 `[0,2]`，并通过 5 折 summary 判断是否稳定。

**可选推理侧消融：Shape-Guided Component Filtering (SGCF)**

FA 血管或边缘伪影常呈细长条状，部分噪声连通域在外接框内也很稀疏；真正的渗漏灶通常更接近片状或团块状。SGCF 在连通域级别计算外接框长宽比 `max_component_aspect_ratio` 和填充度 `min_component_extent = area / bbox_area`，可删除过细长或过稀疏的组件。默认 `[0.0,0.0]` 不启用，建议只作为离线后处理搜索项，避免用单折视觉印象手动定参。

**可选推理侧消融：FA Intensity-Guided Refinement (FAIGR)**

FA 渗漏在图像上通常表现为高荧光区域；模型在暗背景、视野边缘或低荧光结构上产生的小连通域更可能是假阳性。FAIGR 在 morphology 后按组件计算原图亮度的 mean/max，并与每张图自身的亮度分位数比较；低亮度组件会被过滤，但可用 `rescue_min_max_prob` 保留模型极高置信度的小组件。增强版 FAIGR 还可以计算 `intensity - local_mean` 的局部对比度，对全局曝光偏暗但局部突出的渗漏更友好。默认关闭，建议只作为离线推理消融搜索，不改变训练集和 checkpoint。

**可选推理侧消融：Uncertainty-Aware TTA (UATTA)**

普通 TTA 只平均 h/v/hv 翻转预测。UATTA 额外计算各翻转预测概率的标准差，并用 `mean_prob - penalty * std_prob` 得到更保守的概率，目标是压低翻转不稳定的伪阳性区域。默认 `uncertainty_penalty=[0.0,0.0]`，旧结果不受影响；建议先在 f1 上试 `0.0,0.15` 或 `0.0,0.25`，优先惩罚更容易假阳性的 lesion_2。

**可选推理侧消融：Scale-Aware TTA (SATTA)**

在 flip TTA 基础上增加输入尺度投票，例如 `[0.875,1.0,1.125]`，再把各尺度 logits 插值回原始验证尺寸后平均。FA 渗漏区域大小差异明显，SATTA 可以减少单一输入尺度下漏检/过检的偶然性。默认 `scales=[1.0]` 不改变旧结果；ConvNeXt 768 输入建议优先试 `0.875,1.0,1.125`，这三个尺寸分别是 672/768/864，均可被 32 整除。

**动机**

1. 现有 ViT-FPN 日志显示同一 epoch 下独立阈值 sweep 可从 default macro Dice 0.6939 提到 0.7030，说明固定 0.5 阈值不是最优。
2. lesion_2 极罕见且 pos_weight=60，模型容易产生小面积假阳性；小连通域过滤和较高 lesion_2 阈值有明确医学图像后处理动机。
3. FA 渗漏区域通常呈连续片状/边界模糊，closing 与小孔洞填充能修补局部断裂，比继续堆 ViT 解码头更贴合当前失败分析。
4. FA 图像常存在黑边、圆形视野外区域或无效背景；基于原图亮度估计 FOV 后裁掉视野外预测，可降低无医学意义的假阳性。
5. CGMF 把模型概率置信度和形态学面积先验结合，可以避免 lesion_2 小病灶被硬面积阈值误删，适合作为 “概率-形态联合后处理” 小创新。
6. SATTA 用多尺度投票缓解病灶大小变化带来的阈值敏感性，尤其适合渗漏面积跨度大的 FA 图像。
7. UATTA 把几何/尺度增强一致性作为不确定性估计，能惩罚对 TTA 变换敏感的局部假阳性，和 CGMF 的连通域级过滤互补。
8. TCM 用组件级 top-N 先验进一步压制离散伪阳性，尤其适合作为 lesion_2 极罕见类别的后处理消融。
9. FAIGR 把模型概率、形态学组件和原始荧光强度/局部对比度联系起来，相当于引入 FA 成像先验；若 lesion_2 假阳性主要来自低亮度或低对比度小区域，理论上比单纯提高阈值更不容易漏掉局部突出的真病灶。
10. CSA/CBAM decoder attention 属于训练侧模块融合，不改 DINOv3 backbone，只在轻量 decoder 中重标定通道和空间响应，适合作为“结构模块创新”消融。
11. MSAS 利用 ConvNeXt 的天然多尺度 pyramid 做训练期深度监督，推理期零额外开销，适合与 FCM/CGMF 后处理组成“训练侧 + 推理侧”完整创新链路。
12. 边界加权训练把 BCE 梯度集中到病灶内外过渡带，和推理阶段 morphology 的结构修补方向一致，适合作为训练侧小创新消融。
13. EMA weight averaging 不引入推理结构变化，但提供一个低风险的训练稳定化对照；如果 EMA best 在多折上更稳，可作为小样本医学图像训练策略写入消融。
14. HSCF 用低阈值候选区域 + 高阈值种子约束模拟医学图像常用的滞后阈值思想，适合 FA 渗漏“中心亮、边缘弱”的形态，可与 CGMF/FAIGR 叠加搜索。
15. FA-LCE 把局部背景扣除和残差增强前移到输入阶段，利用 FA 渗漏局部高荧光先验，可能改善弱边界渗漏的可分性；但它改变输入分布，必须训练和评估一致。
16. A-TTA 把 FA-LCE 作为推理时 appearance view，而不是训练预处理；它无需重训，适合作为“成像先验测试时增强”离线消融，并可与 SATTA/UATTA/CGMF/HSCF 组合。
17. HNB-BCE 用 hard negative mining 把 BCE 梯度集中到高损失背景区域，针对假阳性导致 Dice 下降的问题；它和后处理 CGMF/HSCF 的目标一致，但作用在训练阶段。
18. BDL 用预测和 GT 的软边界带做 Dice 约束，比单纯边界加权 BCE 更直接地优化轮廓一致性，适合解释为几何边界辅助监督。
19. SGCF 把连通域形态从“面积大小”扩展到“外接框长宽比 + 填充度”，针对 FA 血管样细长假阳性和稀疏噪声组件；它与 CGMF/HSCF/FAIGR 互补，属于概率-形态-成像先验中的形态几何分支。
20. CMCF 把连通域从硬面积统计扩展到概率质量统计，既能过滤低置信大块雾状假阳性，也比单点 max-prob 规则更稳，适合作为“组件级置信质量先验”消融。
21. CLCF 利用两病灶通道之间的标签耦合关系，用 lesion_1 空间支撑约束 lesion_2 候选组件，针对 lesion_2 极罕见导致的孤立假阳性；它属于跨通道几何一致性后处理。
22. APQT 把阈值从全局常数扩展到单图概率分布自适应阈值，尤其适合控制 lesion_2 的预测像素负担；它与 CGMF/FAIGR/SGCF/CMCF/CLCF 可叠加，属于概率校准分支。
23. FECS 把 FOV 从“视野外清零”扩展为“视野边缘风险带组件过滤”，针对 FA 边缘亮边/遮挡/插值伪影导致的贴边假阳性；它属于 FOV 几何先验分支，可用高置信 rescue 降低误删风险。

**推荐先跑的离线评估命令**

```bash
python scripts/evaluate_dinov3_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 --ablation-suite \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/postprocess_eval.json
```

若要单独测试 SATTA/UATTA，把 `--tta-scales` / `--uncertainty-penalty` 加到评估命令中；此时 ablation 里的 TTA/FOV/morphology variants 都会使用对应 TTA logits：

```bash
python scripts/evaluate_dinov3_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 --ablation-suite \
  --tta-scales 0.875,1.0,1.125 \
  --uncertainty-penalty 0.0,0.15 \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/postprocess_eval_satta_uatta.json
```

**推荐消融顺序**

`--ablation-suite` 会在同一次模型前向中输出基础 5 个 variant；若配置启用 `metric.fov_mask.enabled`，额外输出 2 个 FOV variant：

1. `default_0_5`: 原始 logits + 0.5 阈值，无后处理。
2. `calibrated_threshold`: 原始 logits + `[0.50, 0.90]` 双阈值。
3. `calibrated_threshold_tta`: TTA logits + 双阈值。
4. `calibrated_threshold_morph`: 原始 logits + 双阈值 + morphology。
5. `fcm_tta_full`: TTA logits + 双阈值 + morphology。
6. `calibrated_threshold_fov`: 原始 logits + 双阈值 + FOV mask。
7. `fcm_tta_fov_full`: TTA logits + 双阈值 + morphology + FOV mask。

若 f1 的 `fcm_tta_full`、`fcm_tta_fov_full` 或某个中间 variant 有提升，再跑 f2-f5，最后把 5 折均值补进主表。

**5折批量评估与汇总**

```bash
# 默认读取 runs/dinov3_convnext_tiny_5fold_20260615_0250/{f1..f5}/checkpoints/best.pt
bash scripts/run_fcm_tta_5fold_ablation.sh

# 或指定 run-name
bash scripts/run_fcm_tta_5fold_ablation.sh dinov3_convnext_tiny_5fold_20260615_0250
```

输出：

```
runs/<run-name>/postprocess_eval/
  f1.json ... f5.json
  summary.csv
  summary.md
  summary.json
```

`summary.md` 会包含两张表：

1. 各消融 variant 的 5 折 Dice 均值、标准差、相对 `default_0_5` 的提升，并单独标出是否使用 FOV mask。
2. 从每折 `raw_threshold_sweep_base/tta` 汇总出的推荐全局双阈值，例如 `tta -> [0.50, 0.90]`。

`summary.json` 顶层会写入 `best_variant` 和 `recommended_threshold`，方便后续自动追加实验表。

生成实验日志表格行：

```bash
python scripts/log_eval_result.py \
  --json runs/dinov3_convnext_tiny_5fold_20260615_0250/postprocess_eval/summary.json \
  --run-name dinov3_convnext_tiny_fcm_tta_5fold \
  --backbone "ConvNeXt-Tiny + FCM-TTA" \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --epochs "-" \
  --weights-path runs/dinov3_convnext_tiny_5fold_20260615_0250/postprocess_eval/ \
  --change-summary "离线后处理: per-lesion阈值 + TTA + morphology + FOV mask; 无重训" \
  --notes "由summary.json自动生成; 确认后加 --append 写入"
```

**联合阈值 + 形态学参数搜索**

先用 `summary.json` 的 `recommended_threshold` 确定是否采用 TTA，再对单折 checkpoint 联合搜索 lesion-specific threshold 和 morphology 参数。这个步骤比单独 threshold sweep 更贴近最终推理流程，因为它把阈值、连通域过滤、孔洞填充、FOV mask 放在同一套验证指标里比较。

```bash
python scripts/search_morphology_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 \
  --logits tta \
  --threshold-1 0.40 0.45 0.50 0.55 0.60 \
  --threshold-2 0.70 0.80 0.90 0.95 \
  --close-kernels 0 3 \
  --min-area-1 0 32 64 128 \
  --min-area-2 0 8 16 32 \
  --fill-holes-1 0 64 128 \
  --fill-holes-2 0 32 64 \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/morph_search.json \
  --output-md runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/morph_search.md
```

该脚本只前向一次模型，并在同一轮验证里比较多个 threshold + morphology 候选；输出 top-k Dice 和最佳参数 YAML 片段。若需要隔离 FOV 的贡献，可加 `--disable-fov-mask` 重跑一遍。

5 折稳定性选择建议使用批处理脚本，避免只拿 f1 最优参数写论文：

```bash
bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 SATTA/UATTA 一起纳入 5 折 CGMF 搜索，可用环境变量传入 scales 和 penalty：

```bash
TTA_SCALES=0.875,1.0,1.125 UNCERTAINTY_PENALTY=0.0,0.15 \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 FAIGR 一起纳入搜索，建议先用很小网格，只在 lesion_2 上尝试较温和的亮度分位过滤，并用高置信度救援避免误删极小真阳性：

```bash
INTENSITY_MEAN_Q_1=0.0 \
INTENSITY_MEAN_Q_2="0.0 0.25 0.50" \
INTENSITY_RESCUE_MAX_PROB_2="0.0 0.95" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若全局亮度过滤不稳定，可改用局部对比度门控。`INTENSITY_CONTRAST_KERNELS=31` 表示用 31x31 邻域估计局部背景，再按 lesion_2 组件平均局部对比度分位数筛选：

```bash
INTENSITY_MEAN_Q_2=0.0 \
INTENSITY_CONTRAST_KERNELS=31 \
INTENSITY_MEAN_CONTRAST_Q_2="0.0 0.50 0.75" \
INTENSITY_RESCUE_MAX_PROB_2="0.0 0.95" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 SGCF 形状先验一起纳入搜索，建议从温和网格开始，尤其避免过强过滤误删长条状真实渗漏：

```bash
MAX_ASPECT_RATIO_1="0.0 8.0 12.0" \
MAX_ASPECT_RATIO_2="0.0 4.0 6.0" \
MIN_EXTENT_1="0.0 0.15 0.25" \
MIN_EXTENT_2="0.0 0.20 0.30" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 CMCF 概率质量先验一起纳入搜索，建议先只对假阳性更敏感的 lesion_2 尝试温和阈值：

```bash
COMPONENT_MEAN_PROB_1="0.0" \
COMPONENT_MEAN_PROB_2="0.0 0.60 0.70" \
COMPONENT_PROB_MASS_1="0.0" \
COMPONENT_PROB_MASS_2="0.0 4.0 8.0 16.0" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 CLCF 跨病灶一致性纳入搜索，建议先用较小支持要求，只过滤完全孤立的 lesion_2 组件：

```bash
LESION2_SUPPORT_DILATION_KERNELS="0 9 17" \
LESION2_MIN_SUPPORT_PIXELS="0 1 2" \
LESION2_MIN_SUPPORT_FRACTION="0.0" \
LESION2_SUPPORT_THRESHOLDS="0.0 0.50" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 APQT 自适应分位阈值纳入搜索，建议先只打开 lesion_2，避免 lesion_1 大面积渗漏被过强分位阈值截断：

```bash
ADAPTIVE_QUANTILE_1="0.0" \
ADAPTIVE_QUANTILE_2="0.0 0.995 0.997 0.999" \
ADAPTIVE_BLEND_1="1.0" \
ADAPTIVE_BLEND_2="0.5 1.0" \
ADAPTIVE_MIN_THRESHOLD_1="0.0" \
ADAPTIVE_MIN_THRESHOLD_2="0.50 0.60" \
ADAPTIVE_MAX_THRESHOLD_1="1.0" \
ADAPTIVE_MAX_THRESHOLD_2="0.90 0.95" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

若要把 FECS 视野边缘组件过滤纳入搜索，建议只在 lesion_2 或明显贴边假阳性较多的通道上先试温和参数，并开启高置信救援：

```bash
FOV_BORDER_ERODE_KERNELS="0 15 31" \
FOV_BORDER_MIN_INNER_PIXELS_1="0" \
FOV_BORDER_MIN_INNER_PIXELS_2="0 1" \
FOV_BORDER_MIN_INNER_FRACTION_1="0.0" \
FOV_BORDER_MIN_INNER_FRACTION_2="0.0 0.25" \
FOV_BORDER_RESCUE_MAX_PROB_1="0.0" \
FOV_BORDER_RESCUE_MAX_PROB_2="0.0 0.95" \
  bash scripts/run_cgmf_search_5fold.sh dinov3_convnext_tiny_5fold_20260615_0250
```

输出：

```
runs/<run-name>/cgmf_search/
  f1.json ... f5.json
  summary.csv
  summary.md
  summary.json
```

其中 `summary.md` 默认按“同一组阈值 + morphology + FOV + CGMF/FAIGR/SGCF/CMCF/CLCF/APQT/FECS 参数在 5 折上的稳定性惩罚分数”排序，并给出推荐 YAML 片段。若只想汇总已有搜索结果，可直接运行：

```bash
python scripts/summarize_morphology_search.py \
  runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/f*.json \
  --require-all-folds \
  --rank-by robust \
  --output-csv runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/summary.csv \
  --output-md runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/summary.md \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/summary.json
```

`run_cgmf_search_5fold.sh` 默认用稳定性惩罚分数选择推荐配置：`rank_score = mean_macro - 0.5 * std_macro - 0.25 * (mean_macro - min_macro)`。这样可以避免只选择某一折涨得很高、但跨折波动很大的参数。若论文主表想按纯均值排序，可设置 `SUMMARY_RANK_BY=mean` 重跑汇总；若想更保守，可调大 `ROBUST_STD_WEIGHT` 或 `ROBUST_MIN_GAP_WEIGHT`。

将 5 折最优稳定配置导出成可复现实验 YAML：

```bash
python scripts/export_best_morphology_config.py \
  --base-config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --summary-json runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/summary.json \
  --output-config runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/final_cgmf_config.yaml \
  --project-name uveitis_dinov3_convnext_tiny_fcm_fov_cgmf \
  --disable-threshold-sweep
```

随后可用该 `final_cgmf_config.yaml` 做最终 5 折评估/可视化，保证论文中报告的后处理参数与 `summary.json` 一致。

用导出的 final config 做最终 5 折评估：

```bash
bash scripts/run_final_config_5fold_eval.sh \
  dinov3_convnext_tiny_5fold_20260615_0250 \
  runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/final_cgmf_config.yaml
```

输出：

```
runs/<run-name>/final_eval/
  f1.json ... f5.json
  summary.csv
  summary.md
  summary.json
```

`final_eval/summary.json` 是最终建议写入主表的结果；`cgmf_search/summary.json` 用于选择参数，`final_eval/summary.json` 用于报告最终 5 折性能。

导出最终配置的定性可视化（红/黄为预测填充，绿/蓝为 GT 边界；默认每折 12 例，可用 `--sample-ids` 指定病例）：

```bash
python scripts/export_final_visualizations.py \
  --config runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/final_cgmf_config.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 2 --num-workers 4 \
  --max-samples 12 \
  --output-dir runs/dinov3_convnext_tiny_5fold_20260615_0250/final_eval/visualizations/f1
```

将 CGMF 5 折最优稳定配置生成主表行：

```bash
python scripts/log_eval_result.py \
  --json runs/dinov3_convnext_tiny_5fold_20260615_0250/final_eval/summary.json \
  --run-name dinov3_convnext_tiny_fcm_fov_cgmf_5fold \
  --backbone "ConvNeXt-Tiny + FCM-FOV-TTA + CGMF" \
  --config runs/dinov3_convnext_tiny_5fold_20260615_0250/cgmf_search/final_cgmf_config.yaml \
  --epochs "-" \
  --weights-path runs/dinov3_convnext_tiny_5fold_20260615_0250/final_eval/ \
  --change-summary "离线后处理: per-lesion阈值 + TTA + FOV + confidence-guided morphology; 无重训" \
  --notes "由final_eval/summary.json自动生成; 确认后加 --append 写入"
```

若第一阶段找到较稳定的阈值和面积参数，再围绕最佳参数做小范围 CGMF 置信度救援搜索。例如假设最佳参数接近 `[0.50,0.90] + close=3 + min_area=[64,16]`：

```bash
python scripts/search_morphology_postprocess.py \
  --config configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml \
  --checkpoint runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --fold f1 --batch-size 4 --num-workers 4 \
  --logits tta --threshold 0.50,0.90 \
  --close-kernels 3 \
  --min-area-1 64 \
  --min-area-2 16 \
  --rescue-max-prob-1 0.0 0.95 0.98 \
  --rescue-max-prob-2 0.0 0.95 0.98 \
  --max-components-1 0 \
  --max-components-2 0 1 2 \
  --component-score mean_prob \
  --fill-holes-1 128 \
  --fill-holes-2 64 \
  --output-json runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/cgmf_search.json \
  --output-md runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/cgmf_search.md
```

**异构 backbone 集成：ConvNeXt + ViT**

如果本地同时保留了 ConvNeXt-Tiny 和 ViT-B/16 的同折 checkpoint，可以离线评估概率集成。该方法不改训练过程，论文里可作为“多预训练结构互补集成 + FA 后处理校准”的消融。

```bash
python scripts/evaluate_ensemble_postprocess.py \
  --member configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml:runs/dinov3_convnext_tiny_5fold_20260615_0250/f1/checkpoints/best.pt \
  --member configs/dinov3_vitb16_multilabel_itksnap.yaml:runs/dinov3_vitb16_1fold_768_20260615_1537/f1/checkpoints/best.pt \
  --fold f1 --batch-size 2 --num-workers 4 \
  --weights 0.7,0.3 --average prob \
  --threshold 0.50,0.90 \
  --output-json runs/ensemble_convnext_vit_f1/postprocess_eval.json
```

建议先试 `--weights 0.7,0.3` 或 `0.8,0.2`，因为 ConvNeXt 单模型明显强于 ViT；如果集成后 macro Dice 不升，保留 FCM-TTA 和 morphology search 作为主创新即可。

**当前状态**

当前工作区没有 `weights/`、`runs/` checkpoint，也没有解压后的 `dataset/dataset/split_dataorigin`，因此本次只完成代码闭环和轻量验证，尚未产生可填入主结果表的 Dice。

---

## SAM 相关实验汇总

| 实验 | Run Name | 范式 | Fold | 关键结果 | 结论 |
|------|----------|------|------|----------|------|
| SAM2 MAE 预训练 | mae_sam2_f1_20260625_1404 | SAM2 Hiera-Small encoder + MAE 自监督重建 | f1 | best_mae_val_loss=0.2365; best_epoch=50 | 可完成 FA 域自监督适配, 但仅是分割微调前置阶段 |
| MAE-SAM2 分割微调 | mae_sam2_ft_f1_20260625_1404 | SAM2 Hiera-Small + MAE encoder + 2通道多标签分割头 | f1 | best_macro_dice=0.6236; dice_1=0.6503; dice_2=0.5969 | 明显弱于 DINOv3 ViT baseline 0.7337 和 ConvNeXt 主线, 说明 SAM2 直接迁移 FA 渗漏分割存在 domain gap |
| DINOv3 ViT + SAM2 FPN | dinov3_vitb16_fpn_f1 | DINOv3 ViT-B/16 backbone + SAM2-style FPN neck + 深度监督 | f1 | best_macro_dice=0.6939; dice_1=0.7098; dice_2=0.6779; sweep_macro=0.6984 | SAM2 FPN 结构迁移仍未超 ViT baseline, 虚拟金字塔不等于真实多尺度医学特征 |
| SAM-DINO 交互式细化 | dino_sam_refiner_vsubr_vw05_mvp | VS-UBR/DINO 粗预测 + SAM prompt 先验 + 13通道交互式 refiner | f1-f5 | click0 macro=0.7809±0.0193; click3 macro=0.7918±0.0200; click5 macro=0.7965±0.0201 | 将 SAM 定位为交互粗提示而非最终分割器后, 多轮医生点击能稳定提升细分割结果, 支持“粗分割+交互精修系统”作为毕业设计第二部分 |

> 总结: 直接使用 SAM2/MAE-SAM2 做最终 FA 病灶分割效果不足, 但将 SAM 的 promptable 能力转化为医生交互先验, 再交给 DINO/VS-UBR 细化网络进行医学语义分割, 能在 5 折上获得随点击次数稳定提升的结果。

---

## SAM2 实验详情

### 模型版本信息

| 项目 | 版本/路径 |
|------|----------|
| **SAM2 模型** | `sam2-hiera-small` (AI-ModelScope/sam2-hiera-small) |
| **权重文件** | `sam2_hiera_small.pt` (184MB, 46M params) |
| **下载来源** | ModelScope (`modelscope snapshot_download`) |
| **本地路径** | `/root/.cache/modelscope/hub/models/AI-ModelScope/sam2-hiera-small/sam2_hiera_small.pt` |
| **配置文件** | `sam2_hiera_s.yaml` (同目录下) |
| **备用变体** | `sam2-hiera-tiny` (AI-ModelScope/sam2-hiera-tiny), 155MB |
| **PyTorch** | 2.5.1+cu124 |
| **CUDA** | 12.4 |
| **transformers** | 5.12.1 (未用于SAM2, 从零实现Hiera backbone) |
| **modelscope** | 1.37.1 |

### 架构参数

| 组件 | 配置 |
|------|------|
| Backbone | Hiera-Small: embed_dim=96, stages=[1,2,11,2], global_att_blocks=[7,10,13] |
| 总参数量 | 37.1M (encoder 34.3M, seg_head 2.8M) |
| FPN Neck | 4×Conv1x1 [768,384,192,96]→256, top_down_levels=[2,3] |
| MAE Decoder | 4×(Conv3x3+BN+ReLU+Upsample2x), 仅Stage 1使用 |
| Seg Head | Conv3x3(256→128)+Conv3x3(128→128)+Conv1x1(128→2) |
| 窗口注意力 | window_size=8, 非global block使用窗口注意力节省显存 |
| 权重加载 | SAM2 image_encoder权重100%匹配 (missing=0, unexpected=0) |

### 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 768×768 |
| Batch size | 4 |
| Stage 1 epochs | 50 (MAE预训练) |
| Stage 2 epochs | 50 (分割微调) |
| 学习率 | 1e-4, warmup 5 epoch + cosine annealing |
| 优化器 | AdamW (weight_decay=0.01) |
| AMP | 开启 (fp16) |
| 梯度裁剪 | max_norm=1.0 |
| MAE mask ratio | 0.75 |
| 损失函数 | Stage1: MSE(masked patches); Stage2: 0.9×Dice + 0.1×BCE |
| 数据增强 | hflip, vflip |
| 5折交叉验证 | f1-f5 依次运行 |

### 参考文献

- 论文: "MAE-SAM2: Mask Autoencoder-Enhanced SAM2 for Clinical Retinal Vascular Leakage Segmentation" (arXiv:2509.10554)
- 论文报告 Dice: 0.5593 (MAE-SAM2) vs 0.5288 (SAM2 baseline)
- 本项目实现: 从零实现Hiera backbone精确匹配SAM2 checkpoint, 因GitHub被墙无法安装官方sam2包

### 与DINOv3对比

| 模型 | Backbone | 预训练数据 | f1 Macro Dice | 4/5折均值 |
|------|----------|-----------|--------------|----------|
| DINOv3 ConvNeXt-Tiny | ConvNeXt-Tiny | LVD-1689M | - | 0.7710 |
| DINOv3 ViT-B/16 | ViT-B/16 | LVD-1689M | 0.7337 | - |
| MAE-SAM2 (本文) | Hiera-Small | SA-1B + MAE | 0.6236 | - |

> 结论: SAM2在FA泄漏分割上远不如DINOv3, 验证了"通用foundation model直接迁移不如领域预训练"的判断。SAM2实验作为对比基线, 为面向FA特性的专用创新方案提供动机。

---

## DINOv3 ViT + 荧光MAE 实验详情

### 创新点：荧光先验引导的MAE掩码策略

标准MAE使用均匀随机遮盖patch，本方案根据FA图像的荧光亮度调整遮盖概率：

```
patch亮度 → 归一化[0,1] → mask_prob = 0.3 + 0.5 × brightness
```

高荧光区域（血管渗漏区）有更高概率被遮盖（最高80%），迫使模型从周围上下文重建渗漏区域，学习更强的结构理解能力。

### 模型版本信息

| 项目 | 版本/路径 |
|------|----------|
| **Backbone** | DINOv3 ViT-B/16 (Meta AI, LVD-1689M预训练) |
| **权重文件** | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| **代码来源** | `backbone/dinov3/` (Meta官方实现) |
| **模型定义** | `src/bs/dinov3_mae.py` |
| **训练脚本** | `scripts/train_dinov3_mae.py` |
| **配置文件** | `configs/dinov3_vitb16_mae_multilabel.yaml` |
| **Pipeline** | `scripts/run_dinov3_mae.sh` |

### 架构参数

| 组件 | 配置 |
|------|------|
| Backbone | ViT-B/16: embed_dim=768, 12 blocks, patch_size=16 |
| 总参数量 | 110.7M (backbone 85.7M, MAE decoder 14.5M, seg head 10.5M) |
| MAE Decoder | 4×(Conv3x3+BN+ReLU+Upsample2x): 48×48→768×768 |
| Seg Head | TokenFPNHead: 4×Conv1x1(768→256) → fuse → Conv1x1(256→2) |
| 中间层特征 | layers [2, 5, 8, 11] |
| 荧光掩码 | avg_pool2d(image.mean, 16, 16) → brightness_norm → mask_prob ∈ [0.3, 0.8] |
| 掩码比例 | 0.75 (75% patches被遮盖) |

### 两阶段训练配置

| 参数 | Stage 1 (MAE) | Stage 2 (Fine-tune) |
|------|---------------|---------------------|
| 输入尺寸 | 768×768 | 768×768 |
| Batch size | 2 (grad_accum=2) | 1 (grad_accum=4) |
| Epochs | 30 | 30 |
| 学习率 | 1e-5 (cosine) | 1e-4 / backbone 1e-5 (cosine) |
| 优化器 | AdamW (wd=1e-4) | AdamW (wd=1e-4) |
| 损失函数 | MSE (masked patches only) | AsymmetricFocalTverskyBCE |
| Backbone | 全解冻 (12 blocks) | 全解冻 |
| AMP | fp16 | fp16 |
| 梯度裁剪 | max_norm=1.0 | max_norm=1.0 |
| 数据增强 | hflip, vflip, affine, foreground_crop, brightness, noise, blur | 同左 |

### 训练过程

**Stage 1 - MAE预训练 (30 epochs, ~5.2小时)**

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 1 | 0.7195 | 0.5678 | 起始 |
| 5 | 0.3823 | 0.4711 | 快速下降 |
| 10 | 0.3289 | 0.4304 | 持续下降 |
| 20 | 0.3011 | 0.4033 | 趋于收敛 |
| 27 | 0.2922 | **0.3984** | best (最终保存) |
| 30 | 0.2944 | 0.4002 | 训练结束 |

**Stage 2 - 分割微调 (30 epochs, ~2.1小时)**

| Epoch | Val Macro Dice | Val Dice_1 | Val Dice_2 | 备注 |
|-------|---------------|-----------|-----------|------|
| 1 | 0.4935 | 0.5914 | 0.3955 | 起始 |
| 4 | 0.6392 | 0.6739 | 0.6044 | 快速上升 |
| 10 | 0.6195 | 0.7112 | 0.5278 | dice_1最高 |
| 18 | **0.6818** | 0.6907 | 0.6730 | **best** (最终保存) |
| 30 | 0.6678 | 0.6734 | 0.6621 | 训练结束 |

### 结果对比

| 模型 | Backbone | 预训练 | f1 Macro Dice | 与ViT baseline差 |
|------|----------|--------|--------------|-----------------|
| DINOv3 ConvNeXt-Tiny | ConvNeXt-Tiny | LVD-1689M | - (4折均值0.7710) | - |
| **DINOv3 ViT-B/16 (baseline)** | ViT-B/16 | LVD-1689M | **0.7337** | - |
| DINOv3 ViT-B/16 + WBE v1 | ViT-B/16 | LVD-1689M | 0.7248 | -0.89pp |
| DINOv3 ViT-B/16 + WBE v2 | ViT-B/16 | LVD-1689M | 0.7253 | -0.84pp |
| **DINOv3 ViT-B/16 + 荧光MAE** | ViT-B/16 | LVD-1689M + FA-MAE | **0.6818** | **-5.19pp** |
| MAE-SAM2 | Hiera-Small | SA-1B + MAE | 0.6236 | -11.01pp |

### 失败分析

荧光MAE预训练未提升反而降低了分割性能（0.6818 vs 0.7337, -5.19pp），原因分析：

1. **Backbone漂移**: DINOv3 ViT-B/16已在LVD-1689M上充分预训练，MAE微调以1e-5学习率全解冻30 epochs，导致backbone权重从通用特征向FA域偏移，丢失了部分通用表征能力
2. **特征级掩码 vs token级掩码**: 本实现采用feature-level masking（在backbone输出后遮盖），而非原版MAE的token-level masking（在backbone输入前遮盖visible tokens），学习信号较弱
3. **重建目标不匹配**: MAE重建的是原始像素(RGB)，而分割任务关注的是语义边界，两者目标不完全对齐
4. **过拟合**: train_loss持续下降(0.72→0.29)而val_loss在ep10后停滞在0.40附近，MAE decoder可能过拟合训练集

### 启示

- DINOv3 ViT-B/16的LVD-1689M预训练已足够强，直接MAE域适配反而有害
- 未来方向应考虑：(1) 降低MAE学习率或冻结前几层; (2) 采用token-level masking; (3) 将MAE作为辅助损失而非独立阶段; (4) 创新点应放在分割头/损失函数而非backbone适配

---

## DINOv3 ViT + SAM2 FPN 实验详情

### 创新点：SAM2 FPN Neck 适配 ViT + 深度监督

将 SAM2 的 FPN Neck（lateral connection + top-down pathway）适配到 DINOv3 ViT-B/16 上，通过渐进下采样创建虚拟多尺度金字塔，并加入深度监督辅助训练。

```
ViT layers [2, 5, 8, 11] → 全部 48×48, 768-dim
         ↓ 虚拟金字塔下采样
Level 0 (layer 2):  48×48  (最细，低级特征)
Level 1 (layer 5):  24×24
Level 2 (layer 8):  12×12
Level 3 (layer 11):  6×6   (最粗，高级语义)
         ↓ SAM2 FPN top-down 融合
Level 3 → 上采样+add → Level 2 → 上采样+add → Level 1 → 上采样+add → Level 0
         ↓
主输出(48×48) + 3个辅助输出(训练时) → 深度监督 loss
```

### 模型版本信息

| 项目 | 版本/路径 |
|------|----------|
| **Backbone** | DINOv3 ViT-B/16 (Meta AI, LVD-1689M预训练) |
| **权重文件** | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| **模型定义** | `src/bs/model.py` → `ViTFpnHead` + `DinoV3FpnSegmentationModel` |
| **训练脚本** | `scripts/train_dinov3_multilabel.py` (复用，新增 `--head vit_fpn` 支持) |
| **配置文件** | `configs/dinov3_vitb16_fpn_multilabel.yaml` |

### 架构参数

| 组件 | 配置 |
|------|------|
| Backbone | ViT-B/16: embed_dim=768, 12 blocks, patch_size=16 (不修改权重) |
| 总参数量 | 88.8M (backbone 85.7M, FPN head 3.2M) |
| Lateral convs | 4×Conv1×1(768→256) |
| Output convs | 4×(Conv3×3+BN+GELU) after top-down fusion |
| 虚拟金字塔 | avg_pool2d: 48→24→12→6 (2× stride per level) |
| Top-down融合 | nearest上采样 + 逐元素加 (SAM2 FPN style) |
| 主分割头 | Dropout2d(0.1) + Conv1×1(256→2) on finest level |
| 辅助头 (训练) | 3×Conv1×1(256→2) on coarser levels, 权重 0.4/0.16/0.064 |
| 推理额外开销 | 零 (eval模式只返回主输出) |

### 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 768×768 |
| Batch size | 1 (grad_accum=4) |
| Epochs | 30 |
| 学习率 | 1e-4 (head) / 1e-5 (backbone), cosine annealing |
| 优化器 | AdamW (weight_decay=1e-4) |
| 损失函数 | AsymmetricFocalTverskyBCE + 深度监督辅助损失 |
| Backbone | 全解冻 |
| AMP | fp16 |
| 梯度裁剪 | max_norm=1.0 |
| 数据增强 | hflip, vflip, affine, foreground_crop, brightness, noise, blur |

### 训练过程 (30 epochs, ~2.5小时)

| Epoch | Val Macro Dice | Val Dice_1 | Val Dice_2 | 备注 |
|-------|---------------|-----------|-----------|------|
| 1 | 0.5157 | 0.5879 | 0.4436 | 起步低（深度监督拖慢收敛） |
| 3 | 0.6388 | 0.6732 | 0.6043 | 快速上升 |
| 10 | 0.6787 | 0.6927 | 0.6647 | 接近最佳平台 |
| 24 | **0.6939** | 0.7098 | 0.6779 | **best** (最终保存) |
| 30 | 0.6777 | 0.6871 | 0.6683 | 训练结束 |

### 结果对比（全实验汇总）

| # | 模型 | Backbone | 预训练 | f1 Macro Dice | 与baseline差 |
|---|------|----------|--------|--------------|-------------|
| 1 | **DINOv3 ViT-B/16 (baseline)** | ViT-B/16 | LVD-1689M | **0.7337** | - |
| 2 | DINOv3 ViT-B/16 + WBE v1 | ViT-B/16 | LVD-1689M | 0.7248 | -0.89pp |
| 3 | DINOv3 ViT-B/16 + WBE v2 | ViT-B/16 | LVD-1689M | 0.7253 | -0.84pp |
| 4 | DINOv3 ViT-B/16 + 荧光MAE | ViT-B/16 | LVD-1689M + FA-MAE | 0.6818 | -5.19pp |
| 5 | **DINOv3 ViT-B/16 + SAM2 FPN** | ViT-B/16 | LVD-1689M | **0.6939** | **-3.98pp** |
| 6 | MAE-SAM2 | Hiera-Small | SA-1B + MAE | 0.6236 | -11.01pp |
| - | DINOv3 ConvNeXt-Tiny | ConvNeXt-Tiny | LVD-1689M | - (4折均值0.7710) | - |

### 失败分析

SAM2 FPN Neck 适配 ViT 未提升反而降低性能（0.6939 vs 0.7337, -3.98pp），原因分析：

1. **虚拟金字塔是"假"多尺度**: ViT 所有层都是同一分辨率（48×48），下采样只是模糊化同一组特征，未引入新的尺度信息。对比 SAM2 Hiera 天然有 4 个不同感受野的 stage（192→96→48→24），FPN 的 top-down 融合在真多尺度上才有效
2. **下采样丢失空间细节**: TokenFPNHead 保留 4 层特征在 48×48 全分辨率拼接，FPN 把 layer 5/8/11 下采样到 24/12/6，对病灶边界定位丢失了精细空间信息
3. **深度监督在粗尺度上可能有害**: 6×6/12×12 分辨率下病灶仅 1-2 像素，Dice 梯度噪声大，干扰主头训练
4. **top-down 融合不如拼接有效**: 对同分辨率特征，`upsample+add` 本质是加权平均，而 TokenFPNHead 的 `concat+Conv3×3` 让网络自学习各层权重，表达力更强

### 启示

- **SAM2 FPN 是为层级 backbone（Hiera/CNN）设计的，强行适配到非层级 ViT 上不成立**
- ViT 的优势在于全局注意力建模，不是多尺度；创新方向不应强行补"多尺度"
- 综合所有实验（#1-#8），DINOv3 ViT-B/16 baseline 仍为最优；创新应转向损失函数设计、后处理或多模型集成

---

## DiffLeak：扩散物理引导框架 实现记录（2026-07-08）

**核心故事**：把 FA 荧光渗漏建模为一个 **扩散物理过程**（荧光素自病灶源向外扩散，形成中心亮、边界弥散、浓度衰减的高荧光区）。同一物理模型统一导出三个轻量组件，全部不动 backbone（规避 WBE/MAE/SAM2-FPN 的失败教训），在最强的 ConvNeXt-Tiny baseline 上叠加，单张 3090(24GB) 可训。

**三组件与代码**

| 组件 | 作用 | 主要文件 | 开关 |
|------|------|----------|------|
| DALS 扩散外观渗漏合成 | 数据层：物理引导地合成极稀有 lesion_2，热核扩散生成软 alpha 与向外衰减外观 | `src/bs/leakage_synthesis.py`；接入 `augmentations.py`/`dataset.py` | config `augmentations` 中 `leakage_copy_paste` 块 `enabled` |
| DSB 扩散软边界监督 | 损失层：热核扩散把硬标签在边界带软化，缓解边界模糊+标注主观 | `src/bs/multilabel.py`（`AsymmetricFocalTverskyBCE`） | `loss.soft_boundary_sigma`（0=关）/CLI `--soft-boundary-sigma` |
| UGI 不确定性引导推理 | 推理层：TTA 一致性不确定性图 + 可选 ADR 各向异性扩散细化 + 三联可视化 | `src/bs/uncertainty.py`、`src/bs/tta.py`、`scripts/export_uncertainty_visualizations.py` | `metric.tta` + 脚本 `--adr` |
| 补充指标 | 边界 NSD/HD95、面积 MAE/Pearson、ECE 校准（论文报告用） | `src/bs/clinical_metrics.py` | 离线评估调用 |

主配置：`configs/dinov3_convnext_tiny_diffleak.yaml`（DALS+DSB+TTA 全开）。

**前置准备**

```bash
pip install -r requirements-dev.txt        # torch/numpy/PIL/matplotlib/pyyaml/nibabel/pytest
unzip split_dataorigin.zip -d dataset/dataset/     # -> dataset/dataset/split_dataorigin/{img,mask_only_itksnap}
# 放置 backbone 权重: weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
# 单元测试(venv, 复用系统 torch):
python -m venv .venv --system-site-packages && .venv/bin/pip install pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q      # 已验证 111 passed (含 22 个 DiffLeak 新测试)
```

**消融命令（先 f1 判涨点，过则铺 5 折）**

```bash
# baseline (无 DALS 无 DSB)
python scripts/train_dinov3_multilabel.py --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
  --run-name convnext_tiny_baseline_f1 --fold f1 --batch-size 8

# +DSB only (baseline 配置无 DALS, CLI 开软边界)
python scripts/train_dinov3_multilabel.py --config configs/dinov3_convnext_tiny_multilabel_itksnap.yaml \
  --run-name diffleak_dsb_f1 --fold f1 --batch-size 8 \
  --soft-boundary-sigma 2.0 --soft-boundary-band 7 --soft-boundary-weight 1.0

# +DALS only (diffleak 配置 DALS 开, CLI 关 DSB)
python scripts/train_dinov3_multilabel.py --config configs/dinov3_convnext_tiny_diffleak.yaml \
  --run-name diffleak_dals_f1 --fold f1 --batch-size 8 --soft-boundary-sigma 0

# +DALS +DSB (完整训练侧)
python scripts/train_dinov3_multilabel.py --config configs/dinov3_convnext_tiny_diffleak.yaml \
  --run-name diffleak_full_f1 --fold f1 --batch-size 8

# UGI: 不确定性三联可视化 (+可选 ADR)
python scripts/export_uncertainty_visualizations.py --config configs/dinov3_convnext_tiny_diffleak.yaml \
  --checkpoint runs/diffleak_full_f1/f1/checkpoints/best.pt --fold f1 --max-samples 12 --adr \
  --output-dir runs/diffleak_full_f1/f1/uncertainty_vis
```

> A100 迭代期可加大 `--batch-size`；最终论文配置固定 bs=8@768（3090 24GB 可训）。DALS 只增训练期 dataloader 开销、DSB/UGI 近零额外开销，推理与 3090 显存无压力。

**消融主表（5 折均值±std，待训练填充）**

| 方法 | dice_1 | dice_2 | macro_dice | NSD@2 | HD95 | area_MAE | ECE |
|------|--------|--------|-----------|-------|------|----------|-----|
| ConvNeXt-Tiny baseline | | | (对照 ~0.7710) | | | | |
| + DALS | | | | | | | |
| + DSB | | | | | | | |
| + DALS + DSB | | | | | | | |
| + UGI (完整 DiffLeak) | | | | | | | |
| + ADR (可选) | | | | | | | |

**当前状态**：代码闭环，`pytest` 全绿（111 passed，DiffLeak 新增 22 例覆盖 DALS 合成/软边界/ADR/不确定性/临床指标/数据集集成）。尚未训练——待解压数据集与放置 backbone 权重后按上表跑 f1 单折验证涨点，再铺 5 折。默认开关保证旧实验数值不变（`soft_boundary_sigma=0`、`leakage_copy_paste.enabled=false` 时完全回退到既有 baseline）。

---

## 结构头对比实验：RDH-PDE (Perona-Malik) vs S3RD (Mamba)（2026-07-09, f1 单折）

数据/权重已就位：`dataset/dataset/split_dataorigin`(5 折) + `weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`(由 ModelScope `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` HF safetensors 经 `scripts/convert_dinov3_convnext_weights.py` 键名转换)。指标为 f1 验证集 best threshold-sweep(per-lesion 最优阈值) macro dice。

### 结果总表

| # | 方法 | head | DALS | DSB | best_epoch | dice_1 | dice_2 | macro |
|---|------|------|------|-----|-----------|--------|--------|-------|
| 1 | conv baseline | conv | - | - | 24 | 0.7967 | 0.7538 | 0.7752 |
| 2 | + DSB | conv | - | Y | 16 | 0.7941 | 0.7688 | 0.7814 |
| 3 | + DALS | conv(diffleak) | Y | - | 26 | 0.7986 | 0.7793 | 0.7889 |
| 4 | + DALS+DSB (full) | conv(diffleak) | Y | Y | 27 | 0.7972 | 0.7771 | 0.7872 |
| 5 | **RDH-PDE (rdh_only)** | rdh/pde | - | - | 24 | 0.7955 | **0.7872** | **0.7913** |
| 6 | S3RD (Mamba) | rdh/ssm | - | - | 26 | 0.7958 | 0.7614 | 0.7786 |

> 最强 = RDH-PDE 0.7913；S3RD(Mamba) 0.7786，低 1.27pp，主要差在稀有类 dice_2；5/6 均在 itksnap 配置(无 DALS/DSB)下与 conv baseline 纯净对比。
>
> ⚠️ **上表为修正前（脏验证集）数字**，f1 验证集混入 `_aug` 副本导致评估有偏；**无偏结论以本节末尾『数据卫生修正』为准**。

### 实验一：RDH-PDE — 反应-扩散分割头（固定 Perona-Malik）

**动机**：把 FA 渗漏建模为扩散过程，分割="从漏点向外扩散生长"，内在可解释(interpretable-by-design)，参数极少、受物理正则、抗小数据过拟合。

**结构**(`src/bs/rdh.py` dynamics=pde；接入 `convnext_seg.py` head_type=rdh)：
1. 种子 `s = sigmoid(Conv1x1(fused))`：渗漏起漏点。
2. 传导 `c = sigmoid(Conv1x1(feat)) * exp(-(gradI/kappa)^2)`：由原图高荧光梯度(Perona-Malik)决定，血管/边界处 c→0 停止扩散；kappa 可学习。
3. 反应-扩散演化 K=8 步：`u_0=s`，`u_{t+1}=clamp(u_t+dt*[div(c*grad u_t)+rho*s*u_t*(1-u_t)-lam*u_t], 0,1)`，dt/rho/lam 每类可学习。
4. 残差输出 `logit(u_K)`：iters=0/dt→0 退化为普通 1x1 头，保证不劣于 conv。

**命令**：`--config .../itksnap.yaml --run-name diffleak_f1_rdh_only --fold f1 --batch-size 8 --head rdh`。
**结果**：macro 0.7913(最强), dice_2 0.7872，显存 ~12.9G。可视化诊断：种子精准落在高荧光渗漏区(可解释成立)，但传导 c 偏均匀、演化 u 变化细微(固定 Perona-Malik 表达力受限)。

### 实验二：S3RD — 选择性状态空间反应-扩散（Mamba 驱动）

**动机**：RDH 迭代 `u_{t+1}=f(u_t)` 本质是状态空间递推；用选择性 SSM(Mamba)沿空间扫描学习数据驱动的各向异性传播，替代固定 Perona-Malik，博 novelty 且保留可解释框架。

**实现**(`src/bs/ssm.py` 纯 PyTorch, 无需编译 mamba-ssm)：`selective_scan_1d`(离散化 `Abar=exp(dt*A)`,A<0; `h=Abar*h+dt*B*x`; `y=C*h+D*x`, for 循环) + `SelectiveSSM2D`(VMamba 式 4 方向扫描, 序列长=H/W 其余并行; dt/B/C 由 feat+高荧光 guide 投影; avg_pool 降分辨率 stride=4 控显存; out_proj 零初始化→退化为 seed; d_state=16,d_inner=64)。接入 `rdh.py` dynamics=ssm：`logits=seed_logits+SSM传播`。

**命令**：`--config .../itksnap.yaml --run-name diffleak_f1_s3rd --fold f1 --batch-size 8 --head rdh --rdh-dynamics ssm`。
**结果**：macro 0.7786, dice_2 0.7614。无 NaN，显存 12.7G，纯 torch selective scan ~2 it/s(比 conv/pde 慢约 2x)。

### 对比与结论（修正前 · 脏验证集，评估有偏）

- **S3RD 0.7786 < RDH-PDE 0.7913 (-1.27pp)**，差距几乎全在稀有类 **dice_2 (0.7614 vs 0.7872, -2.58pp)**，lesion_1 持平；S3RD 仍略高于 conv baseline 0.7752。
- **原因**：小数据(~2k, lesion_2 仅 21% 图含)下 Mamba 参数远多于近乎无参的 Perona-Malik，稀有类过拟合；固定物理先验零过拟合、更稳。
- 与此前 WBE(-0.9pp)/荧光MAE(-5.2pp)/SAM2-FPN(-4pp) 缝复杂模块均掉点**一脉相承**——小样本医学分割中强物理先验 > 数据驱动复杂模块。
- **决策**：论文主线锁定 **RDH-PDE(可解释+最强)**；S3RD/Mamba 作为"探索 SSM 但物理先验更优"的诚实对比消融写入；下一步铺 RDH-PDE 的 5 折确认稳定性。

---

### 数据卫生修正（干净验证集，2026-07-10）

**问题发现**：badcase 分析(`scripts/compare_rdh_s3rd.py`)显示 top badcase 全是同一病例 **7474 及其 9 个 `_aug` 离线增强副本**，均落在 f1 验证集且被 S3RD 全漏检——同一难病例被重复计入验证集约 10 次，**人为放大了 S3RD 的 dice_2 劣势**（评估偏差 + 潜在泄漏）。

**修正**：`dataset.py:discover_samples` 新增 `exclude_augmented`；`train_dinov3_multilabel.py` 验证集默认剔除 `_aug`（config `data.exclude_val_augmented`，f1 494→444，训练集保留增强）。用干净验证集重训 baseline/RDH-PDE/S3RD（126 tests 全绿无回归）。

**干净 f1 结果（best sweep macro）**：

| 方法 | dice_1 | dice_2 | macro | best_epoch |
|------|--------|--------|-------|-----------|
| baseline | 0.7906 | 0.7632 | 0.7769 | 16 |
| **RDH-PDE** | 0.7958 | **0.7702** | **0.7830** | 24 |
| S3RD(Mamba) | **0.7970** | 0.7643 | 0.7807 | 17 |

**脏 vs 干净对照**：

| 方法 | 脏(有偏) | 干净(无偏) |
|------|---------|-----------|
| RDH-PDE | 0.7913 | 0.7830 |
| S3RD | 0.7786 | 0.7807 |
| **差距** | **1.27pp** | **0.23pp** |

**修正结论（以此为准）**：
- RDH-PDE 仍最强，但仅比 S3RD 高 **0.23pp（几乎打平）**，两者均 > baseline 0.7769；
- 之前"S3RD 明显差 1.27pp / 物理先验明显更优"**主要是脏验证集(7474 重复+漏检)造成的假象**，已被推翻；
- 论文叙事更新为"**物理扩散头 RDH 与数据驱动的 Mamba 版性能相当，但 RDH 参数更少、更可解释、更适合小样本部署**"；
- 仍需 **5 折**确认（两者 best_epoch 差异大：RDH ep24 / S3RD ep17，S3RD 更早达峰）。

---

## ViT-B/16 backbone 上的 RDH-PDE / S3RD 对比（2026-07-15）

**动机**：验证 RDH/S3RD 的增益是否随 backbone 迁移（ConvNeXt→ViT），以定主线 backbone。

**准备**：ModelScope 下载 `facebook/dinov3-vitb16-pretrain-lvd1689m`(HF safetensors)，经 `scripts/convert_dinov3_vit_weights.py` 转为 dinov3 hub state_dict(188 键 strict OK，qkv 合并/bias_mask与rope继承初始化) → `weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。

**接入**：`model.py:TokenFPNHead` 新增 `head_type`(conv/rdh)，rdh 时 neck 融合 4 层 ViT token(48×48) → `ReactionDiffusionHead`，`DinoV3SegmentationModel.forward` 透传原图作 guide；train build_model ViT 分支支持 `--head rdh` + 全套 rdh 参数 + 新增 `--rdh-stride`(ViT S3RD 用 stride=2, 因 token 仅 48×48)。126 tests 全绿无回归。

**结果（f1 干净验证集 444, best sweep macro）**：

| 方法 | Run Name | head | best_epoch | dice_1 | dice_2 | macro |
|------|----------|------|-----------|--------|--------|-------|
| ViT baseline | `diffleak_f1_vitb16_clean` | conv | 24 | 0.7194 | 0.7507 | 0.7350 |
| ViT + RDH-PDE | `diffleak_f1_vitb16_rdh_clean` | rdh/pde | 25* | 0.7325 | 0.6877 | 0.7101 |
| ViT + S3RD | `diffleak_f1_vitb16_s3rd_clean` | rdh/ssm | 3* | 0.7017 | 0.5779 | 0.6398 |

\* RDH/S3RD 手动提前终止(跑到 ep25/ep24, 未满 30)；趋势已明确(RDH 缓慢爬升但追不上自身 baseline，S3RD ep3 见顶后一路过拟合)。

**对比 ConvNeXt(主线)**：conv 0.7769 / RDH-PDE 0.7830 / S3RD 0.7807。

**结论**：
- ViT-B/16 整体弱于 ConvNeXt-Tiny(baseline 0.7350 vs 0.7769, 低 ~4.2pp)，与历史一致。
- **RDH/S3RD 在 ViT 上不复现增益、反而掉点**(RDH 0.7101 < ViT baseline 0.7350；S3RD 0.6398 且 ep3 见顶)，与 ConvNeXt 上“接 RDH 涨点”相反。
- **归因**：RDH 物理扩散演化依赖高分辨率多尺度特征(ConvNeXt 192×192)；ViT 48×48 粗 token 上扩散空间精度不足、上采样边界糊化，稀有类 lesion_2 掉最多。
- **决策**：主线锁定 **ConvNeXt-Tiny**；本对比作为“RDH 增益与特征分辨率强相关”的消融证据，支撑论文选 ConvNeXt 而非 ViT。

---

## 边缘检测创新 Wave 1：Boundary DoU 损失 + Edge-PDC 边缘引导头（2026-07-21, f1 单折）

**动机**：从边缘检测方向提升 ConvNeXt-Tiny 主线。文献调研（CTO IPMI2023 / PiDiNet ICCV2021 / Boundary DoU MICCAI2023 / 软边缘）后，选“低参数 + 强先验 + 训练期监督”的三条路线，规避 WBE(6.44M) 在 ~2k 小数据上的过拟合教训。

**新增代码**（172 tests 全绿；默认开关全 0，旧实验数值零回归）：
- `src/bs/edge.py`：中心像素差分卷积 CPDC（参数量=普通卷积、对常数区响应恒 0、天然边缘敏感）、`EdgeGuidedHead`（PDC 边缘分支 + 边缘门控增强主特征 + 主分割头；gate 零初始化 → 初始等价普通 conv 头）、`lesion_edge_target`（硬/软边界带监督目标）。
- `src/bs/multilabel.py`：`AsymmetricFocalTverskyBCE` 新增 Boundary DoU 项（`boundary_dou_weight`，自适应 α）；`EdgeGuidedLoss` 包装类（分割损失 + 边界带 BCE/Dice 辅助监督，可选软边缘）。
- `src/bs/convnext_seg.py`：`head_type=edge` 接入；`scripts/train_dinov3_multilabel.py`：新增 `--boundary-dou-weight/--edge-*/--seed` CLI 与 build_model/build_loss 接线；`configs/dinov3_convnext_tiny_edge.yaml`；看板 `scripts/monitor_edge_experiments.py`（自动刷新 HTML 仪表盘）。

> ⚠️ 本环境为 torch 2.9（历史为 2.5.1），故**重训同环境 baseline**（0.7800）作公平对照，不与历史 0.7769/0.7710 直接比。

**结果（f1 干净验证集 444, best per-lesion 阈值 sweep macro）**：

| # | 方法 | Run Name | head | 损失附加 | best_epoch | dice_1 | dice_2 | macro | Δ vs baseline |
|---|------|----------|------|---------|-----------|--------|--------|-------|--------------|
| 1 | baseline(同环境) | `edge_baseline_f1` | conv | - | 24 | 0.7967 | 0.7633 | 0.7800 | - |
| 2 | Boundary DoU w=1.0 | `edge_dou_f1` | conv | DoU 1.0 | 7 | 0.7819 | 0.7777 | 0.7798 | -0.02pp |
| 3 | **Edge-PDC 软边缘** | `edge_pdc_soft_f1` | edge | edge 0.5(soft) | 16 | 0.7994 | 0.7670 | **0.7832** | **+0.32pp** |
| 4 | Edge-PDC 硬边缘 | `edge_pdc_hard_f1` | edge | edge 0.5(hard) | 16 | 0.7958 | 0.7675 | 0.7817 | +0.17pp |

权重路径：`runs/<run-name>/f1/checkpoints/best.pt`。

**分析**：
- 排名 **edge-soft 0.7832(+0.32pp) > edge-hard 0.7817(+0.17pp) > baseline 0.7800 > DoU@1.0 0.7798**；涨幅小、处于单折噪声区间，且未收敛前领先亦反映**收敛更快**（edge 峰 ep16 vs baseline ep24）。
- ✅ **软边缘 > 硬边缘**：印证“FA 渗漏边界弥散、硬边缘 GT 噪声大、宜用软边缘”的假设，可作论文消融结论。
- ✅ **边界/边缘监督早期抬升稀有类 lesion_2**：DoU 在 ep7 时 dice_2=0.7777（全场最高）；两 edge 头 dice_2≈0.767 均 > baseline 0.7633。
- ⚠️ **DoU@1.0 调过头**（非方法失败）：val_loss 1.33（约他者 2.4×），ep7 见顶后单调退化——边界项权重过大压过区域项；Wave 2 降到 0.3 复验。

**决策**：edge-soft 暂列最优但涨幅在噪声内 → Wave 2（f1, 2026-07-21）用 seed=43 对照测**噪声地板 + 跨种子稳定性**，并复验 **DoU@0.3** 与 **edge-soft+DoU@0.3** 组合；跨种子仍稳定领先者再铺 **5 折**出 mean±std 定论。

## UAG-SAM v1（2026-08-28，五折训练完成；2026-08-29 现场恢复）

针对现有 DINO-SAM interactive 的 oracle 点击偏差、硬提示编码、无自动找错和点击噪声脆弱性，新增 UAG-SAM v1：不确定性驱动的无 GT 点击策略、17 通道 signed/reliability prompt、uncertainty-gated residual refiner、点击 jitter/dropout 和 click-consistency loss。实现与实验门槛见 `docs/dino_sam_refiner_uag.md`；配置为 `configs/dino_sam_refiner_uag.yaml`，无 GT policy 评估为 `scripts/evaluate_interactive_policy.py`。当前只通过静态检查和合成张量回归，**没有预先声称 Dice 提升**。

正式五折训练使用 `outputs/dino_refiner_cache/vsubr_vw05_compact` 的紧凑缓存，并由 `scripts/run_dino_sam_uag_background.sh` 在 detached screen 会话 `uag_train` 中运行。由于迭代式点击会保留多轮高分辨率激活，实际配置采用 batch=1、base_channels=16、训练展开至 3-click，验证仍评估 0/1/3/5-click；首个 f1 epoch 已完成且未再出现显存溢出。缓存约 49 GiB、训练输出约数十 MiB，脚本在进入各阶段前保留至少 20 GiB 可用空间。

新增 `scripts/monitor_uag_storage.py` 作为持续磁盘 watchdog：每 30 秒检查项目所在文件系统，剩余空间低于 25 GiB 时告警，低于 20 GiB 时向 UAG 进程组发送 SIGTERM（必要时 SIGKILL），随后只清理临时文件和未被 manifest 引用的缓存样本，绝不删除已完成 checkpoint/metrics。当前运行同时由 detached screen 会话 `uag_storage_guard` 监控，日志为 `outputs/interactive_refiner_runs/dino_sam_refiner_uag_v1/storage_guard.log`。监控发现训练异常退出或触发磁盘保护时，会写入 `failure_event.json` / `storage_guard.stop`，并通过 `codex queue` 将带有日志路径的处理请求排入当前 Codex 线程（线程标识保存在 `codex_thread_id`）；若 Codex app-server 不可用，事件文件和日志仍会保留供下一次进入任务时处理。

**现场恢复记录（2026-08-29）**：失败事件记录的剩余空间为 41.664 GiB、使用率 58.34%，不是磁盘保护触发；日志显示 f1–f5 均已完成 epoch 20，随后旧监督脚本在退出阶段报 `line 63: syntax error near unexpected token ')'`。保留原 `failure_event.json` 作为审计证据，未删除 49 GiB 紧凑缓存或任何 checkpoint/metrics。修复内容包括：后台脚本在被 `sh` 错误调用时自举到 Bash；新增 `training.complete` 正常完成标记并让 watchdog 识别该标记；训练器按 `metrics.csv`/`latest.pt` 跳过完整折、从断点续训未完成折，并在折间释放 DataLoader worker 与 CUDA cache。修复后的后台复跑于 07:29:37 正常完成，f1–f5 各 20 个 epoch 均被完整复用。

五折 UAG-SAM 结果（`click3_paper_macro_dice` 最佳值）：f1 **0.766291**（ep10）、f2 **0.789687**（ep13）、f3 **0.805373**（ep12）、f4 **0.742596**（ep4）、f5 **0.785360**（ep13）；五折均值 **0.777862**，样本标准差 **0.024138**（总体标准差 **0.021589**）。各折完整指标见 `outputs/interactive_refiner_runs/dino_sam_refiner_uag_v1/f{1..5}/metrics.csv`，汇总见 `fold_summary.csv`。

## UAG-SAM 组件消融 smoke（2026-08-29，f1 小样本）

为定位正式 UAG 五折结果偏弱的原因，复用同一份 49 GiB DINO 紧凑 cache，在 f1 上做 4 组 5 epoch smoke：训练最多 128 张、验证最多 64 张，四组均使用相同 seed 和数据子集。结果不是最终论文数字，只用于判断组件和训练闭环的方向。

| 组 | 变体 | click0 best | click1 best | click3 best | click5 best | click3 最佳 epoch |
|---|---|---:|---:|---:|---:|---:|
| A | 原始 MVP（13ch、one-shot） | 0.8091 | 0.8096 | 0.8105 | 0.8114 | 5 |
| B | 17ch soft prompt（无 UAG gate） | 0.8141 | 0.8149 | 0.8160 | **0.8171** | 3 |
| C | 17ch soft prompt + UAG gate（one-shot） | **0.8143** | **0.8149** | **0.8160** | 0.8171 | 4 |
| D | 完整 iterative UAG（累计点击、迭代展开） | 0.8111 | 0.8126 | 0.8129 | 0.8121 | 2 |

**观察**：

- B 相对 A 在 click3 提升 **+0.54pp**、click5 提升 **+0.57pp**；提示编码是当前最明确的正向因素。
- C 与 B 几乎持平（click3 仅 **+0.002pp**，click5 **-0.006pp**），当前 smoke 不支持单独保留 UAG gate 作为性能增益来源，但它没有造成明显退化。
- D 在 click3 达到 **0.8129** 后，click5 降到 **0.8121**；其最佳 epoch 只有 2，且最终 epoch 的 click5 比峰值低 **0.41pp**，说明问题集中在 iterative rollout 的稳定性/训练目标，而不是 soft prompt 本身。
- 需要注明一个控制变量限制：A 使用 batch=4，B/C 使用 batch=1，D 为控制迭代显存把 `base_channels` 降到 16（A–C 为 32）。因此 B/C 的优势是“组件+当前训练设置”的方向性证据，不能当作严格因果增益；下一轮应统一 batch/base，或做参数量匹配复验。
- 因此下一轮应优先做“soft prompt + 稳定迭代”：限制每轮 residual、使用 stop-gradient/teacher-forcing 或 scheduled rollout，并单独扫 click-consistency 权重；暂不直接扩大完整 iterative UAG 的训练规模。

汇总表、JSON 和曲线图由 `scripts/summarize_uag_ablation.py` 生成，位于 `outputs/interactive_refiner_ablation_smoke/summary.csv`、`summary.json` 和 `uag_ablation_curves.png`。本轮输出约 233 MiB，实验结束后项目文件系统剩余约 42 GiB。

## UAG-SAM 统一配置稳定性复验（2026-08-29，f1 小样本）

针对上轮 iterative UAG 在 click5 退化的问题，保留 17 通道 soft prompt 和 uncertainty-gated refiner，并将四组实验统一为 batch=1、`base_channels=16`、seed=42、f1 训练 128 张/验证 64 张、5 epoch；仅改变迭代稳定策略。验证阶段 `eval_teacher_forcing_ratio=0`，因此 teacher 只用于训练中间状态，不读取验证标注。

| 组 | 变体 | click0 best | click1 best | click3 best | click5 best | click3 最佳 epoch |
|---|---|---:|---:|---:|---:|---:|
| S0 | soft prompt one-shot 参考 | 0.8117 | 0.8118 | 0.8119 | 0.8119 | 5 |
| S1 | iterative + residual step limit=0.5 | 0.8110 | 0.8134 | 0.8162 | 0.8171 | 5 |
| S2 | S1 + stop-gradient | 0.8112 | 0.8131 | 0.8149 | 0.8156 | 2 |
| S3 | S1 + teacher-forcing ratio=0.5 | 0.8115 | 0.8137 | 0.8162 | **0.8175** | 5 |

**结果解读**：

- residual 步长限制单独就恢复了稳定的点击收益：S1 从 click0 到 click5 增长 **+0.61pp**，且 click3/click5 不再出现旧 D 组的反向下降。
- stop-gradient 进一步降低了跨轮反传和显存图复杂度，曲线保持单调，但性能比 S1 低约 0.15pp（click5）；它更像稳定性/工程选项，而不是当前的性能增益来源。
- teacher-forcing 在训练中间状态后，S3 达到本轮最高 click5 **0.8175**，比 S1 高约 **0.04pp**，同时 lesion_2 best Dice 达到 **0.8143**；这是值得进入下一轮较大样本验证的候选。
- S0 在同一窄模型配置下几乎不随点击变化，说明收益来自真正的多轮 residual 更新，而不是仅仅增加 prompt 通道。

这仍是单折、64 张验证子集、5 epoch 的筛选实验，不能直接替代五折结论；尤其 S3 的 teacher-forcing 仍需做 ratio/teacher logit scale 扫描，并在无 GT policy 和扰动点击下复验。结果文件位于 `outputs/interactive_refiner_stability_smoke/summary.csv`、`summary.json` 和 `uag_ablation_curves.png`。

## S3 参数扫描（2026-08-29，f1 小样本）

在 S3 的统一设置上扫描 `residual_step_limit ∈ {0.25, 0.5}` 与 `teacher_forcing_ratio ∈ {0.25, 0.5, 0.75}`，其余配置保持不变。

| residual limit | teacher 0.25 | teacher 0.50 | teacher 0.75 |
|---:|---:|---:|---:|
| 0.25 | 0.8172 | 0.8159 | 0.8165 |
| 0.50 | 0.8170 | 0.8175 | **0.8210** |

表中为 click5 的最佳 macro Dice；click3 的最佳值分别为 0.8154、0.8146、0.8150、0.8161、0.8162、**0.8185**。最优组合为 `limit=0.5 + teacher_forcing_ratio=0.75`：click0/1/3/5 = **0.8126/0.8154/0.8185/0.8210**，最佳 epoch=5，lesion_2 Dice=**0.8194**。相较 limit=0.5、teacher=0.5，click5 提升约 **0.35pp**。

结果显示两个参数存在协同关系：teacher ratio 提高后，需要更大的单步 residual 上限才能把教师状态的方向转化为有效迭代；在 limit=0.25 时 ratio=0.5/0.75 反而没有提升。该最优点仍只来自单折 64 张验证子集和 5 epoch，下一步应固定 `limit=0.5`，围绕 ratio=0.75 做 0.6/0.75/0.9 复验，并加入无 GT policy、jitter/dropout 鲁棒性和五折确认。

扫描结果位于 `outputs/interactive_refiner_s3_scan/summary.csv`、`summary.json` 和 `uag_ablation_curves.png`，总输出约数百 MiB，项目文件系统剩余约 42 GiB。

## MVP residual uncertainty-click gate（2026-08-31，已有 checkpoint 无重训复评）

> **口径更正（2026-09-03）**：本节原表把 `iterative=false` 的 MVP checkpoint
> 错误地按 iterative rollout 评估，因此表中的绝对值和 `+1.29pp/+2.67pp`
> 不可作为正式结果。该历史输出仍保留用于审计；修正后的一次性 MVP 主结果和
> 严格 iterative 对照见下方“口径校正与严格 iterative 稳定性”小节。

针对 MVP 在多轮点击中会对高置信正确区域做全图 residual 更新的问题，新增参数无关的 `uncertainty_click` gate：以 DINO uncertainty 为基础门控，点击热图附近恢复完整 residual；默认 `floor=0.25`、`click_gain=0.75`。旧配置 `residual_gate=none` 的行为和 checkpoint 不变。实现位于 `src/bs/interactive_refiner.py`，配对评估入口为 `scripts/evaluate_interactive_thresholds.py`，汇总入口为 `scripts/summarize_interactive_gate.py`。

使用同一批 `dino_sam_refiner_vsubr_vw05_compact_full` MVP checkpoint、同一折、同一随机种子（42）和同一点击模拟，只改变 gate；下面先给出原始训练口径阈值 0.5 的五折配对结果：

| clicks | MVP（无 gate） | uncertainty-click gate | Δ macro Dice | 提升折数 |
|---:|---:|---:|---:|---:|
| 0 | 0.7874 | 0.7871 | −0.02pp | 2/5 |
| 3 | 0.7645 | **0.7774** | **+1.29pp** | **5/5** |
| 5 | 0.7506 | **0.7773** | **+2.67pp** | **5/5** |

0-click 基本不变，提升集中在交互后的多轮更新；逐折 click5 增益为 f1 +0.69pp、f2 +2.15pp、f3 +2.20pp、f4 +1.19pp、f5 +7.11pp。另在 0.9/0.9 验证校准阈值下，click3/click5 的五折增益分别为 +1.19pp/+2.38pp。结果文件为 `outputs/interactive_eval/mvp_gate_fixed05_summary.csv` 及对应逐折 JSON。

该结果是**固定 seed 的离线 oracle-click 配对复评**，用于证明 gate 能抑制多轮全图漂移；absolute Dice 会随随机 oracle 点位置变化，不能直接替换历史 MVP 主表。无 GT policy 在 f1 上仍随点击数下降（gate click5 macro=0.7584），说明下一步应继续改进点击推荐策略或采用“医生确认极性”的交互模式，不能把 gate 结果包装成已解决的自动闭环。

## MVP 提示半径校准与 MVP-R12（2026-09-03）

针对 UAG-SAM 在完整五折上低于 MVP 的结果，先固定已有 `dino_sam_refiner_vsubr_vw05_compact_full` checkpoint，仅做推理侧配对复评。输出阈值在候选网格中选择，五折均稳定落在 lesion_1=0.9，lesion_2 多数为 0.8–0.95；因此论文应把 0.9 作为校准阈值单独报告，不能与训练指标混淆。

在同一 checkpoint、seed=42、random oracle-click 口径下，将点击热图半径由 8 调为 12（模型参数和前向结构不变），固定输出阈值 0.9 的已有复评结果如下：

| fold | click-3 macro | click-5 macro |
|------|--------------:|--------------:|
| f1 | 0.7935 | 0.7974 |
| f2 | 0.8046 | 0.8092 |
| f3 | 0.8259 | 0.8307 |
| f4 | 0.7623 | 0.7654 |
| f5 | 0.8116 | 0.8187 |
| **mean±std** | **0.8004±0.0218** | **0.8037±0.0220** |

该结果是轻量的提示足迹校准，不应宣称为新增网络参数带来的提升。为验证训练/推理一致性，新增 `configs/dino_sam_refiner_mvp_r12.yaml`，使用同一 radius=12 训练 MVP refiner；f1 训练在第 3 个 epoch 后停止（batch=2，峰值显存约 15 GiB），未形成可替代原 MVP 的新 checkpoint。代码同时支持 `radius: [12, 8]` 的每病灶半径消融，后者用于检验 lesion_1 弥散、lesion_2 细小的形态先验。

在 R12 基础上把 lesion_2 residual 通道缩放到 1.2 的五折复评（无重训）得到
click3 **0.8008±0.0215**、click5 **0.8042±0.0221**，相对 R12 的变化为
**+0.03pp/+0.05pp**；逐折并非一致提升，因此只保留为通道校准消融，不纳入主配置。对应参数无关的评估入口为
`--residual-channel-scale 1.0,1.2`，结果位于
`outputs/interactive_eval/mvp_oneshot_chscale10_12_f{1..5}_r12.json`。

补充测试了按单图概率分位数调整 lesion_2 阈值的 APQT 变体（`quantile=0.999`、阈值下限 0.9）。f1 的 click3/click5 仅为 **0.7864/0.7897**，明显低于固定阈值 R12，因此不纳入部署路径；该结果保留在 `outputs/interactive_eval/mvp_oneshot_apqt_l2_f1_r12.json`。

补充的 f1 配对消融（均为已有 MVP checkpoint、输出阈值网格校准）记录如下：residual scale=0.5 明显下降，scale=1.5 与原值接近；radius=[12,8] 低于统一 radius=12；点击极性投影 strength=8 只带来局部安全约束，没有超过统一阈值的 MVP；四向 flip-TTA 也未超过单向推理。因此主线暂采用 **统一 radius=12 + 固定校准阈值**，不叠加这些未稳定的选项。soft-signed 17 通道 refiner 的完整 f1 checkpoint 在同一阈值扫描下为 click3/click5 **0.7904/0.7944**，仍低于 legacy MVP-R12，故不扩展五折。

无 GT policy 的独立输出阈值复评（`mvp_policy_r12_thr09_f1.json`）显示 click-0=0.7840、click-3=0.7657、click-5=0.7512；把最终二值化阈值从 0.5 校准到 0.9 并不能消除自动点选误差。由此当前部署建议仍是“模型自动给出粗分割，医生确认/点击后再细化”，自动 policy 作为后续工作而不是论文主结果。

## MVP 一次性口径校正与严格 iterative 稳定性（2026-09-03）

此前 residual-gate 小节中的 `+1.29pp/+2.67pp` 是由评估器把原始 MVP checkpoint 误按 iterative 路径展开造成的，不能作为正式 MVP 增益。修正后，原始 MVP 的“一次采样 K 点→一次 residual 更新”五折、seed=42、随机 oracle-click 结果为：click0 **0.7879±0.0196**、click3 **0.7992±0.0215**、click5 **0.8047±0.0229**（population std；逐折 JSON 位于 `outputs/interactive_eval/mvp_oneshot_random_sweep_f{1..5}.json`）。这组数字替代旧的错误评估口径，历史实验文件仍保留以便审计。

为验证真实 SAM 风格闭环，在 f1 使用同一 MVP checkpoint、逐轮根据最新预测采样误差点，得到严格 iterative 基线：click3 **0.7770**、click5 **0.7743**（最佳 lesion-specific 阈值均为 0.95/0.95），相对于一次性口径分别下降约 1.5pp 和 2.2pp。加入每步 residual `tanh` 限制 0.5 并把相对初始 DINO 的累计漂移限制为 1.0 后，click3 **0.7839**、click5 **0.7854**；把 residual scale 再降到 0.5 几乎不变（0.7841/0.7854），因此累计锚定比简单缩放更有效，但仍未达到一次性 MVP。相关结果为 `mvp_iterative_f1_none.json`、`mvp_iterative_f1_step05_total10.json` 和 `mvp_iterative_f1_scale05_step05_total10.json`。

从 MVP 初始化的 iterative fine-tune（f1，learning rate=3e-5，step limit=0.5，teacher forcing=0.75）在第 1/2 epoch 的固定 0.5 阈值 click3 macro Dice 为 **0.7586/0.7622**，虽逐 epoch 上升但仍明显低于 MVP；该 run 在 epoch2 后停止，作为“直接闭环训练未恢复一次性性能”的负消融，不扩展五折。代码新增的 `--total-residual-limit`、累计锚定和对应单元测试保留在主分支，部署时可作为安全稳定策略。

无 GT 自动点击 policy 也做了固定 seed=42 的配对复评：在输出阈值 0.9 下，未加累计锚定的 click3/click5 macro Dice 为 **0.7793/0.7737**；加入 step=0.5、total=1.0 后为 **0.7815/0.7816**，对应 +0.22pp/+0.79pp。该结果来自 `mvp_policy_f1_thr09_seed42.json` 与 `mvp_policy_f1_step05_total10_seed42.json`，并不使用 GT 选点；它支持将“累计残差锚定”作为部署安全策略，但不能替代 oracle-click 主表。

在同一稳定化 rollout 下加入 click jitter=2 像素、click dropout=0.1 的 f1 鲁棒性测试，click3/click5 为 **0.7835/0.7851**，与无扰动的 0.7839/0.7854 接近；说明该约束策略对轻微医生点位误差和偶发漏点不敏感。结果文件为 `outputs/interactive_eval/mvp_iterative_f1_total10_jitter2_dropout10.json`。

将 stable rollout 扩展到 f2–f5 后，五折均值为 click3 **0.7868±0.0199**、click5
**0.7884±0.0201**；相同 checkpoint 的 one-shot 均值为 0.7992/0.8047，因此
稳定化主要解决“多轮不稳定”，尚未超过一次性 MVP。逐折结果文件为
`outputs/interactive_eval/mvp_iterative_stable_f2_v2.json` 以及
`mvp_iterative_stable_f{3,4,5}.json`（f1 使用
`mvp_iterative_f1_step05_total10.json`；f2 的 `_v2` 文件是在修复 evaluator
读取 YAML 默认限制后重跑的有效版本）。论文中应把 stable rollout 作为闭环
安全消融，而把 one-shot 医生点击路径作为当前主性能结果。

## MVP→soft prompt 增量迁移（2026-09-03）

为避免从随机初始化训练 17 通道提示编码器造成遗忘，新增了严格的 MVP
初始化路径：将原 13 通道 stem 权重复制到扩展模型，新增的 signed/reliability
四个输入通道置零，因此初始前向与 MVP 完全一致。`train.init_checkpoint` 支持
该形状受限的 13→17 通道扩展；`train.prompt_adapter_only=true` 时冻结原有
参数，并用梯度 mask 只更新新增 stem 通道和（默认可选的）residual output
projection。配置与实现分别见
`configs/dino_sam_refiner_soft_prompt_adapter_f1.yaml`、
`scripts/train_interactive_refiner.py`。

使用 seed=42、random one-shot oracle-click、radius=8、统一阈值网格
`{0.5,0.6,0.7,0.8,0.9,0.95}` 对五折 checkpoint 做配对扫描。f1 使用
`dino_sam_refiner_soft_prompt_adapter_f1`，f2–f5 使用各自两 epoch 筛选
checkpoint（文件位于 `outputs/interactive_refiner_runs/`）。结果如下，均为
每折独立选择 lesion-specific 最优阈值后的 macro Dice：

| clicks | MVP 5 折 mean±std | prompt adapter mean±std | Δ | 正向折数 |
|---:|---:|---:|---:|---:|
| 0 | 0.78788±0.01957 | 0.78764±0.01960 | −0.024pp | 0/5 |
| 3 | 0.79920±0.02154 | **0.80006±0.02157** | **+0.086pp** | 5/5 |
| 5 | 0.80470±0.02292 | **0.80551±0.02289** | **+0.080pp** | 4/5 |

增益幅度较小但 click3 五折同向，适合作为“参数极少的提示增量适配”消融；
不能把它表述为显著替代 MVP 的主涨点。逐折 JSON 为
`outputs/interactive_eval/soft_prompt_adapter_f{1..5}_thrscan.json`，MVP 对照
为 `mvp_oneshot_random_sweep_f{1..5}.json`。

另实现并短训了更严格的 stem-only 版本
`configs/dino_sam_refiner_soft_prompt_stem_adapter_f1.yaml`：仅训练新增
stem 通道，冻结 residual output projection。该版本在 f1 的 0-click
**0.78555** 与 MVP 完全一致，click3/click5 为 **0.79275/0.79663**；它验证了
“不牺牲粗分割、只让提示通道改变交互结果”的安全设计，但单折增益低于可训练
projection 的 adapter，因此暂不扩展五折。

### Adapter + radius-12 推理校准

在同一五折 adapter checkpoint 上仅把参数无关的点击热图半径由训练时的 8 调为
12，并重新进行 lesion-specific 阈值扫描。完整 `{0.5,0.6,0.7,0.8,0.9,0.95}`
网格得到 click3 **0.80229±0.02229**、click5 **0.80678±0.02294**；相对
radius=8 MVP 的 `0.79920/0.80470` 分别为 **+0.309pp/+0.208pp**，click3 为
5/5 折正向、click5 为 4/5 折正向。逐折文件为
`outputs/interactive_eval/soft_prompt_adapter_f{1..5}_r12_fullthrscan.json`。
该组合可作为当前推荐的“轻量提示适配 + 弥散足迹校准”配置，但增益仍然有限，
不能替代 MVP 的主模型结论。

## Lesion-2 单任务专家与双模型路由消融（2026-09-23，f1）

为验证极稀有 `lesion_2` 是否适合从共享多标签模型中拆出，新增可配置的
`loss.channel_weights`。专家模型使用 `[0, 1]`，因此只有 lesion_2 会产生训练
梯度；结构仍保持两个输出通道，以便从共享主模型 checkpoint 稳定初始化。

专家从 `runs/hmvf_vsubr_f1_30/f1/checkpoints/best.pt` 初始化，使用 HMVF +
VS-UBR、learning rate `3e-5`、5 epoch，在 clean f1 验证集按 lesion_2 的独立
阈值 Dice 选模。最佳为 epoch 3：lesion_2 Dice **0.771877**（threshold=0.8），
相同验证口径下主模型 lesion_2 为 **0.762987**（threshold=0.9），提升
**+0.889pp**。训练初期出现 7 个 non-finite batch，epoch 4 出现 1 个，均由
现有 skip 保护处理；峰值显存约 12.21 GiB。

最终推理不是平均两个模型的全部输出，而是逐通道路由：lesion_1 固定使用主模型，
lesion_2 在主模型/专家之间改变概率权重，并对两个通道独立搜索阈值。所有结果均
关闭 TTA、形态学后处理、强度修正和 FOV mask，避免把后处理收益混入模型消融。

| lesion_2 主模型权重 | lesion_2 专家权重 | 最优阈值 (L1/L2) | Dice-1 | Dice-2 | Macro |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 0.8 / 0.9 | 0.795973 | 0.762987 | 0.779480 |
| 0.75 | 0.25 | 0.8 / 0.9 | 0.795973 | 0.768858 | 0.782416 |
| 0.50 | 0.50 | 0.8 / 0.9 | 0.795973 | 0.770506 | 0.783240 |
| 0.25 | 0.75 | 0.8 / 0.9 | 0.795973 | 0.771045 | 0.783509 |
| **0.00** | **1.00** | **0.8 / 0.8** | **0.795973** | **0.771877** | **0.783925** |

硬路由相对单主模型使 lesion_2 提升 **+0.889pp**，Macro 提升 **+0.445pp**；
且专家权重从 25% 增至 100% 时性能单调提高，当前不需要概率平均。这个结论仍然
只是 f1 单折验证集上的适配/选模结果，不能当作独立测试集或五折结论；下一步应在
f2-f5 分别训练各自专家并做交叉验证汇总，再决定是否作为最终部署路径。

配置与权重位于
`configs/dinov3_convnext_tiny_hmvf_vsubr_lesion2_expert.yaml` 和
`runs/lesion2_expert_hmvf_vsubr_f1_screen5/f1/checkpoints/best.pt`；完整联合评估
JSON 位于 `runs/ablation_lesion2_expert_f1/`。评估器同时增加无后处理情况下的
等价快速阈值扫描，相关回归测试已通过。

## Experiment Archive Records

| Date | Archive ZIP | Backbone | Method | Results | Included runs |
|---|---|---|---|---|---|
| 20260823 | `20260823_dinov3-convnext-tiny_dino-sam-interactive.zip` | DINOv3-ConvNeXt-Tiny | DINO-SAM-Interactive | outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_dice_1_best=0.775192; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_dice_2_best=0.765483; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_macro_dice_best=0.770338; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_dice_1_best=0.780071; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_dice_2_best=0.768226; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_macro_dice_best=0.774148; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_dice_1_best=0.786511; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_dice_2_best=0.773842; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_macro_dice_best=0.779011; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_dice_1_best=0.791791; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_dice_2_best=0.777675; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_macro_dice_best=0.782579; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_dice_1_best=0.816481; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_dice_2_best=0.752624; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_macro_dice_best=0.783349; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_dice_1_best=0.819859; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_dice_2_best=0.762182; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_macro_dice_best=0.788824; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_dice_1_best=0.825441; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_dice_2_best=0.774748; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_macro_dice_best=0.796096; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_dice_1_best=0.829355; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_dice_2_best=0.782388; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_macro_dice_best=0.802470; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_dice_1_best=0.804196; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_dice_2_best=0.808572; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_macro_dice_best=0.805402; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_dice_1_best=0.808972; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_dice_2_best=0.812170; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_macro_dice_best=0.809080; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_dice_1_best=0.816522; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_dice_2_best=0.817266; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_macro_dice_best=0.815856; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_dice_1_best=0.822932; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_dice_2_best=0.823318; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_macro_dice_best=0.822039; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_dice_1_best=0.790636; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_dice_2_best=0.699803; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_macro_dice_best=0.745214; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_dice_1_best=0.795567; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_dice_2_best=0.703094; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_macro_dice_best=0.749330; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_dice_1_best=0.800215; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_dice_2_best=0.707258; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_macro_dice_best=0.753183; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_dice_1_best=0.803709; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_dice_2_best=0.712207; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_macro_dice_best=0.757731; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_dice_1_best=0.804032; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_dice_2_best=0.771027; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_macro_dice_best=0.786770; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_dice_1_best=0.807872; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_dice_2_best=0.782718; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_macro_dice_best=0.792973; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_dice_1_best=0.814250; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_dice_2_best=0.805238; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_macro_dice_best=0.807095; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_dice_1_best=0.819209; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_dice_2_best=0.818223; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_macro_dice_best=0.816557; fivefold_0click_macro_dice=0.7782; fivefold_1click_macro_dice=0.7829; fivefold_3click_macro_dice=0.7902; fivefold_5click_macro_dice=0.7963 | `outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full` |
| 20260829 | `20260829_dinov3-convnext-tiny_uag-sam-s3.zip` | DINOv3-ConvNeXt-Tiny | UAG-SAM-S3 | outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click0_paper_dice_1_best=0.770708; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click0_paper_dice_2_best=0.764090; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click0_paper_macro_dice_best=0.767276; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click1_paper_dice_1_best=0.765634; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click1_paper_dice_2_best=0.764866; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click1_paper_macro_dice_best=0.765250; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click3_paper_dice_1_best=0.759528; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click3_paper_dice_2_best=0.765814; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click3_paper_macro_dice_best=0.762671; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click5_paper_dice_1_best=0.756126; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click5_paper_dice_2_best=0.766415; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f1_metrics_click5_paper_macro_dice_best=0.761270; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click0_paper_dice_1_best=0.817523; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click0_paper_dice_2_best=0.753547; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click0_paper_macro_dice_best=0.784965; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click1_paper_dice_1_best=0.812056; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click1_paper_dice_2_best=0.754473; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click1_paper_macro_dice_best=0.782356; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click3_paper_dice_1_best=0.796728; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click3_paper_dice_2_best=0.756264; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click3_paper_macro_dice_best=0.774817; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click5_paper_dice_1_best=0.782764; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click5_paper_dice_2_best=0.757230; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f2_metrics_click5_paper_macro_dice_best=0.768633; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click0_paper_dice_1_best=0.807614; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click0_paper_dice_2_best=0.807197; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click0_paper_macro_dice_best=0.807101; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click1_paper_dice_1_best=0.802327; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click1_paper_dice_2_best=0.807816; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click1_paper_macro_dice_best=0.804444; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click3_paper_dice_1_best=0.795921; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click3_paper_dice_2_best=0.808242; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click3_paper_macro_dice_best=0.801046; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click5_paper_dice_1_best=0.792825; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click5_paper_dice_2_best=0.808367; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f3_metrics_click5_paper_macro_dice_best=0.799274; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click0_paper_dice_1_best=0.790157; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click0_paper_dice_2_best=0.699676; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click0_paper_macro_dice_best=0.744708; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click1_paper_dice_1_best=0.783294; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click1_paper_dice_2_best=0.699395; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click1_paper_macro_dice_best=0.740817; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click3_paper_dice_1_best=0.776825; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click3_paper_dice_2_best=0.698681; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click3_paper_macro_dice_best=0.736945; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click5_paper_dice_1_best=0.774530; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click5_paper_dice_2_best=0.697950; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f4_metrics_click5_paper_macro_dice_best=0.735129; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click0_paper_dice_1_best=0.809112; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click0_paper_dice_2_best=0.766506; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click0_paper_macro_dice_best=0.787557; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click1_paper_dice_1_best=0.805518; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click1_paper_dice_2_best=0.766083; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click1_paper_macro_dice_best=0.785072; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click3_paper_dice_1_best=0.792370; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click3_paper_dice_2_best=0.765924; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click3_paper_macro_dice_best=0.779147; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click5_paper_dice_1_best=0.785849; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click5_paper_dice_2_best=0.766543; outputs_interactive_refiner_runs_dino_sam_refiner_uag_s3_best_v1_f5_metrics_click5_paper_macro_dice_best=0.776196; fivefold_best_click3_macro_dice_mean=0.770925; fivefold_best_click5_macro_dice_mean=0.768100; fold_f1_best_click3_macro_dice=0.762671; fold_f2_best_click3_macro_dice=0.774817; fold_f3_best_click3_macro_dice=0.801046; fold_f4_best_click3_macro_dice=0.736945; fold_f5_best_click3_macro_dice=0.779147 | `outputs/interactive_refiner_runs/dino_sam_refiner_uag_s3_best_v1` |
| 20260830 | `20260830_dinov3-convnext-tiny_fa-ubir.zip` | DINOv3-ConvNeXt-Tiny | FA-UBIR | outputs_interactive_boundary_runs_fa_ubir_v1_f1_metrics_oracle_click3_paper_dice_1_best=0.758444; outputs_interactive_boundary_runs_fa_ubir_v1_f1_metrics_oracle_click3_paper_dice_2_best=0.750387; outputs_interactive_boundary_runs_fa_ubir_v1_f1_metrics_oracle_click3_paper_macro_dice_best=0.748740; outputs_interactive_boundary_runs_fa_ubir_v1_f2_metrics_oracle_click3_paper_dice_1_best=0.783258; outputs_interactive_boundary_runs_fa_ubir_v1_f2_metrics_oracle_click3_paper_dice_2_best=0.724693; outputs_interactive_boundary_runs_fa_ubir_v1_f2_metrics_oracle_click3_paper_macro_dice_best=0.749813; outputs_interactive_boundary_runs_fa_ubir_v1_f3_metrics_oracle_click3_paper_dice_1_best=0.778449; outputs_interactive_boundary_runs_fa_ubir_v1_f3_metrics_oracle_click3_paper_dice_2_best=0.798642; outputs_interactive_boundary_runs_fa_ubir_v1_f3_metrics_oracle_click3_paper_macro_dice_best=0.787065; status=early_stopped_manual; f1_best_oracle_click3_macro_dice=0.748740; f2_best_oracle_click3_macro_dice=0.749813; f3_best_oracle_click3_macro_dice=0.787065; complete_folds=2; partial_fold=f3_epoch11 | `outputs/interactive_boundary_runs/fa_ubir_v1` |
| 20260903 | `20260903_dinov3-convnext-tiny_mvp-r12-interactive.zip` | DINOv3-ConvNeXt-Tiny | MVP-R12-Interactive | outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_dice_1_best=0.775192; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_dice_2_best=0.765483; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click0_paper_macro_dice_best=0.770338; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_dice_1_best=0.780071; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_dice_2_best=0.768226; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click1_paper_macro_dice_best=0.774148; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_dice_1_best=0.786511; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_dice_2_best=0.773842; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click3_paper_macro_dice_best=0.779011; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_dice_1_best=0.791791; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_dice_2_best=0.777675; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f1_metrics_click5_paper_macro_dice_best=0.782579; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_dice_1_best=0.816481; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_dice_2_best=0.752624; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click0_paper_macro_dice_best=0.783349; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_dice_1_best=0.819859; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_dice_2_best=0.762182; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click1_paper_macro_dice_best=0.788824; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_dice_1_best=0.825441; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_dice_2_best=0.774748; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click3_paper_macro_dice_best=0.796096; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_dice_1_best=0.829355; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_dice_2_best=0.782388; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f2_metrics_click5_paper_macro_dice_best=0.802470; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_dice_1_best=0.804196; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_dice_2_best=0.808572; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click0_paper_macro_dice_best=0.805402; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_dice_1_best=0.808972; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_dice_2_best=0.812170; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click1_paper_macro_dice_best=0.809080; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_dice_1_best=0.816522; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_dice_2_best=0.817266; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click3_paper_macro_dice_best=0.815856; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_dice_1_best=0.822932; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_dice_2_best=0.823318; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f3_metrics_click5_paper_macro_dice_best=0.822039; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_dice_1_best=0.790636; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_dice_2_best=0.699803; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click0_paper_macro_dice_best=0.745214; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_dice_1_best=0.795567; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_dice_2_best=0.703094; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click1_paper_macro_dice_best=0.749330; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_dice_1_best=0.800215; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_dice_2_best=0.707258; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click3_paper_macro_dice_best=0.753183; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_dice_1_best=0.803709; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_dice_2_best=0.712207; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f4_metrics_click5_paper_macro_dice_best=0.757731; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_dice_1_best=0.804032; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_dice_2_best=0.771027; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click0_paper_macro_dice_best=0.786770; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_dice_1_best=0.807872; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_dice_2_best=0.782718; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click1_paper_macro_dice_best=0.792973; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_dice_1_best=0.814250; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_dice_2_best=0.805238; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click3_paper_macro_dice_best=0.807095; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_dice_1_best=0.819209; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_dice_2_best=0.818223; outputs_interactive_refiner_runs_dino_sam_refiner_vsubr_vw05_compact_full_f5_metrics_click5_paper_macro_dice_best=0.816557; fivefold_r12_click0_macro=0.787355±0.019682; fivefold_r12_click1_macro=0.793279±0.020686; fivefold_r12_click3_macro=0.800413±0.021850; fivefold_r12_click5_macro=0.803711±0.021954; r12_vs_r8_click3_delta_pp=+0.105; r12_vs_r8_click5_delta_pp=-0.023; fivefold_r12_click3_gain_vs_click0_pp=+1.306; fivefold_r12_click5_gain_vs_click0_pp=+1.636 | `outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full` |
| 20260906 | `20260906_dinov3-convnext-tiny_va-rdh-oriented-pdc-neck-screen.zip` | DINOv3-ConvNeXt-Tiny | VA-RDH-Oriented-PDC-neck-screen | runs_va_rdh_pdcneck_f1_screen_f1_metrics_train_paper_dice_1_best=0.755021; runs_va_rdh_pdcneck_f1_screen_f1_metrics_train_paper_dice_2_best=0.798546; runs_va_rdh_pdcneck_f1_screen_f1_metrics_train_paper_macro_dice_best=0.776784; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_dice_1_best=0.746329; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_dice_2_best=0.704107; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_macro_dice_best=0.720888; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_best_dice_1_best=0.755899; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_best_dice_2_best=0.728636; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_best_macro_dice_best=0.742267; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_ind_dice_1_best=0.755899; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_ind_dice_2_best=0.728636; runs_va_rdh_pdcneck_f1_screen_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.742267; runs_va_rdh_pdcneck_f1_screen_fold_summary_best_paper_macro_dice_best=0.720888; runs_va_rdh_pdcneck_f1_screen_fold_summary_best_paper_dice_1_best=0.746329; runs_va_rdh_pdcneck_f1_screen_fold_summary_best_paper_dice_2_best=0.695446; runs_va_rdh_pdcneck_f1_screen_mean_paper_dice_1=0.746329; runs_va_rdh_pdcneck_f1_screen_mean_paper_dice_2=0.695446; runs_va_rdh_pdcneck_f1_screen_mean_paper_macro_dice=0.720888; best_sweep_macro_dice=0.7423; best_fixed_macro_dice=0.5727; baseline_va_rdh_k16_macro_dice=0.7872; peak_vram_gb=13.31 | `runs/va_rdh_pdcneck_f1_screen` |
| 20260921 | `20260921_dinov3-convnext-tiny_vsubr-vw05-repro.zip` | DINOv3-ConvNeXt-Tiny | VSUBR-VW05-Repro | runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_train_paper_dice_1_best=0.838295; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_train_paper_dice_2_best=0.900012; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_train_paper_macro_dice_best=0.867842; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_dice_1_best=0.786711; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_dice_2_best=0.762778; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_macro_dice_best=0.770947; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_best_dice_1_best=0.794288; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_best_dice_2_best=0.771076; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_best_macro_dice_best=0.781235; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_ind_dice_1_best=0.794288; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_ind_dice_2_best=0.775393; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.782263; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_train_paper_dice_1_best=0.835231; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_train_paper_dice_2_best=0.907253; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_train_paper_macro_dice_best=0.871242; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_dice_1_best=0.823893; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_dice_2_best=0.752427; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_macro_dice_best=0.786044; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_best_dice_1_best=0.828558; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_best_dice_2_best=0.752594; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_best_macro_dice_best=0.787477; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_ind_dice_1_best=0.829373; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_ind_dice_2_best=0.752594; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f2_metrics_val_paper_sweep_ind_macro_dice_best=0.787477; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_train_paper_dice_1_best=0.836464; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_train_paper_dice_2_best=0.899719; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_train_paper_macro_dice_best=0.868092; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_dice_1_best=0.813041; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_dice_2_best=0.806523; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_macro_dice_best=0.809782; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_best_dice_1_best=0.815766; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_best_dice_2_best=0.811636; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_best_macro_dice_best=0.811449; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_ind_dice_1_best=0.815766; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_ind_dice_2_best=0.819109; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f3_metrics_val_paper_sweep_ind_macro_dice_best=0.812819; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_train_paper_dice_1_best=0.841747; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_train_paper_dice_2_best=0.904137; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_train_paper_macro_dice_best=0.872476; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_dice_1_best=0.799130; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_dice_2_best=0.703347; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_macro_dice_best=0.748929; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_best_dice_1_best=0.802108; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_best_dice_2_best=0.706558; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_best_macro_dice_best=0.752691; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_ind_dice_1_best=0.802108; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_ind_dice_2_best=0.707747; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f4_metrics_val_paper_sweep_ind_macro_dice_best=0.753286; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_train_paper_dice_1_best=0.839474; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_train_paper_dice_2_best=0.906891; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_train_paper_macro_dice_best=0.872983; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_dice_1_best=0.821586; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_dice_2_best=0.766968; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_macro_dice_best=0.787460; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_best_dice_1_best=0.823235; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_best_dice_2_best=0.766968; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_best_macro_dice_best=0.791198; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_ind_dice_1_best=0.823235; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_ind_dice_2_best=0.771431; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_f5_metrics_val_paper_sweep_ind_macro_dice_best=0.791525; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_fold_summary_best_paper_macro_dice_best=0.809782; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_fold_summary_best_paper_dice_1_best=0.819661; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_fold_summary_best_paper_dice_2_best=0.806523; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_mean_paper_dice_1=0.803482; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_mean_paper_dice_2=0.757783; runs_dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514_mean_paper_macro_dice=0.780633 | `runs/dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514` |
| 20260921 | `20260921_dinov3-vit-b16_hmvf.zip` | DINOv3-ViT-B16 | HMVF | runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_train_paper_dice_1_best=0.734663; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_train_paper_dice_2_best=0.729509; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_train_paper_macro_dice_best=0.732086; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_dice_1_best=0.686949; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_dice_2_best=0.658181; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_macro_dice_best=0.670957; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_best_dice_1_best=0.733188; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_best_dice_2_best=0.668572; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_best_macro_dice_best=0.693566; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_dice_1_best=0.733188; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_dice_2_best=0.669527; runs_vitb16_hmvf_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.694043 | `runs/vitb16_hmvf_f1_20260921_2010` |
| 20260921 | `20260921_dinov3-vit-b16_vsubr.zip` | DINOv3-ViT-B16 | VSUBR | runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_train_paper_dice_1_best=0.785748; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_train_paper_dice_2_best=0.816803; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_train_paper_macro_dice_best=0.800143; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_dice_1_best=0.727758; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_dice_2_best=0.715840; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_macro_dice_best=0.708692; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_best_dice_1_best=0.738360; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_best_dice_2_best=0.735600; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_best_macro_dice_best=0.734755; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_dice_1_best=0.738360; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_dice_2_best=0.735600; runs_vitb16_vsubr_f1_20260921_2010_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.736027; runs_vitb16_vsubr_f1_20260921_2010_fold_summary_best_paper_macro_dice_best=0.734755; runs_vitb16_vsubr_f1_20260921_2010_fold_summary_best_paper_dice_1_best=0.701545; runs_vitb16_vsubr_f1_20260921_2010_fold_summary_best_paper_dice_2_best=0.715840; runs_vitb16_vsubr_f1_20260921_2010_mean_paper_dice_1=0.701545; runs_vitb16_vsubr_f1_20260921_2010_mean_paper_dice_2=0.715840; runs_vitb16_vsubr_f1_20260921_2010_mean_paper_macro_dice=0.734755 | `runs/vitb16_vsubr_f1_20260921_2010` |
| 20260921 | `20260921_nnunet-2d_aaag.zip` | nnUNet-2D | AAAG | runs_aaag_nnunet2d_clean_f1_30_f1_metrics_train_paper_dice_1_best=0.769528; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_train_paper_dice_2_best=0.825907; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_train_paper_macro_dice_best=0.795985; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_dice_1_best=0.762086; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_dice_2_best=0.715817; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_macro_dice_best=0.738951; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_dice_1_best=0.770976; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_dice_2_best=0.723746; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_macro_dice_best=0.744304; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_dice_1_best=0.770976; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_dice_2_best=0.723746; runs_aaag_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.745616; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_macro_dice_best=0.738951; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_dice_1_best=0.762086; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_dice_2_best=0.715817; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_macro_dice_best=0.744304; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_dice_1_best=0.768437; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_dice_2_best=0.720171; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_macro_dice_best=0.745616; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_dice_1_best=0.768437; runs_aaag_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_dice_2_best=0.722796; runs_aaag_nnunet2d_clean_f1_30_mean_paper_dice_1=0.762086; runs_aaag_nnunet2d_clean_f1_30_mean_paper_dice_2=0.715817; runs_aaag_nnunet2d_clean_f1_30_mean_paper_macro_dice=0.738951 | `runs/aaag_nnunet2d_clean_f1_30` |
| 20260921 | `20260921_nnunet-2d_ukan.zip` | nnUNet-2D | UKAN | runs_ukan_nnunet2d_clean_f1_30_f1_metrics_train_paper_dice_1_best=0.769876; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_train_paper_dice_2_best=0.821588; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_train_paper_macro_dice_best=0.793767; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_dice_1_best=0.740362; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_dice_2_best=0.741230; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_macro_dice_best=0.726886; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_dice_1_best=0.770288; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_dice_2_best=0.749787; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_best_macro_dice_best=0.752010; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_dice_1_best=0.770288; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_dice_2_best=0.749787; runs_ukan_nnunet2d_clean_f1_30_f1_metrics_val_paper_sweep_ind_macro_dice_best=0.752010; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_macro_dice_best=0.726886; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_dice_1_best=0.712541; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_dice_2_best=0.741230; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_macro_dice_best=0.752010; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_dice_1_best=0.754233; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_dice_2_best=0.749787; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_macro_dice_best=0.752010; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_dice_1_best=0.754233; runs_ukan_nnunet2d_clean_f1_30_fold_summary_best_paper_sweep_ind_dice_2_best=0.749787; runs_ukan_nnunet2d_clean_f1_30_mean_paper_dice_1=0.712541; runs_ukan_nnunet2d_clean_f1_30_mean_paper_dice_2=0.741230; runs_ukan_nnunet2d_clean_f1_30_mean_paper_macro_dice=0.726886 | `runs/ukan_nnunet2d_clean_f1_30` |
| 20261015 | `20261015_dinov3-convnext-tiny_physleak-cdp.zip` | DINOv3-ConvNeXt-Tiny | PhysLeak-CDP | cdp_beta10_pos_pooled_rho_3seed=0.7478; cdp_nested_pooled_rho=0.748; prior_global_rho=0.7065; cdp_cell_auroc=0.894; supervised_oof_ceiling_rho=0.965; vitb16_r1_rho_mean=0.7101; vitb16_oracle_rho=0.7304; knn_lesion2_presence_auc=0.914 | `outputs/physleak_severity_20261014` |


## 推理优化：病灶保真约束下的 CPU 部署（2026-10-09，clean f1）

此次不重训，优化现存可完整加载的最强 f1 自动模型 HMVF：`runs/hmvf_mambavision_f1_30/f1/checkpoints/best.pt`，epoch 21，SHA256 `03c688c3dcf7cd3e30a1e86d854c2fa540b5e0193394766613dcf4faf2c51fdc`。历史 VA-RDH K16 / VS-UBR 最佳及 10 月采样实验权重未找到，未纳入本次优化，不能把本次结果标为这些历史模型的部署表现。

评估固定 clean f1 全部 444 张、768×768、batch=1、阈值 0.90/0.90、无 TTA。原模型 GPU eager FP16 macro=0.7802335209。接受要求为 macro 降幅 ≤0.2pp、各类 ≤0.3pp、逐病例配对无新增整图漏检（GT 阳性且预测为空）。PTQ 校准为 f2–f5 干净图随机 128 张，seed=42，不使用 f1。

| 方案 | Dice-1 | Dice-2 | Macro | Δ Macro (pp) | 新增整图漏检 L1/L2 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 原模型 GPU eager FP16 | 0.797852 | 0.762616 | 0.780234 | 0 | — | 基准 |
| GPU 合批 + channels_last + compile | 0.797869 | 0.762579 | 0.780224 | −0.00093 | 0/0 | 通过 |
| OpenVINO FP32（8 线程复评） | 0.797870 | 0.762541 | 0.780205 | −0.00282 | 0/0 | 通过，备选包 |
| **OpenVINO 解码器 INT8（最终 9 线程复评）** | **0.797650** | **0.763843** | **0.780747** | **+0.05131** | **0/0** | **默认 CPU 包** |
| 大范围 INT8，保留敏感分支 | 0.794831 | 0.761781 | 0.778306 | −0.19274 | 1/0 | 拒绝：病例 9604 新增漏检；Dice-1 −0.3021pp |
| 上述 INT8 再保护首 stage | 0.796393 | 0.754231 | 0.775312 | −0.49211 | 2/1 | 拒绝 |

默认策略保留 backbone、SSM 全局分支、最后分割卷积为 FP32，仅量化其余 decoder。整图漏检数为 L1=1、L2=6，与基准相同；稀有类只有 38 张阳性图，不能将小幅 Dice 上浮视为已证实的泛化改进。降分辨率 736/704/672/640 全折筛查均失败，Dice-2 分别 −1.248/−1.193/−1.203/−1.641pp，保留 768。

速度实测使用 Xeon Platinum 8255C + RTX 2080 Ti；容器 CPU 配额 9 核。最终计时串行运行，CPU 主对照均 9 线程，8 次 warmup 后计时 40 次，从预处理后 NumPy batch 到 CPU logits（包括 GPU 数据传输）：

- CPU 原始 PyTorch NCHW：平均 **672.6 ms**；channels_last **505.1 ms**；OpenVINO 合批 FP32 **345.5 ms**；仅 decoder INT8 **219.4 ms**，相对 NCHW **3.07×**，相对 channels_last **2.30×**。
- GPU 原始 eager **32.33 ms** → 完整 compile 组合 **18.57 ms（1.74×）**；此次缓存后的首次调用仍需 10.51 秒。单独合批 eager 为 26.74 ms；单独 channels_last 没有收益。
- 实际独立 CPU 入口（无 torch/CUDA）从已解码 PIL 图片到标签图，50 次热启动平均 **246.0 ms**，P95 **268.3 ms**；构造/模型校验/编译 **1.54 秒**，峰值 RSS **563 MiB**。不含磁盘解码与写图。
- 默认权重 109.26 MiB，FP32 117.95 MiB；没有达到四倍体积压缩。8/9 线程有极少阈值边缘像素差异，因此默认包使用重新完整复评的 9 线程结果。

新增 SSM 方向合批、CPU 图导出、训练折 PTQ、固定阈值全折复评、校验后打包、纯 NumPy/Pillow/OpenVINO 预测入口。可方形扫描 4 次循环合为 1 次，保持权重、递推公式与方向累加次序，不宣称 FLOPs 变成四分之一。OpenVINO 单独合批没有稳定收益，主要 CPU 加速来自图编译和选择性 INT8。

验证：23 项相关测试通过；全部 444 张预处理与训练管线最大差 0；最终纯 CPU 部署全部 444 张逐图面积与 Dice 对齐最终 9 线程复评，`torch_imported=false`，执行设备仅 CPU。

默认包 `outputs/inference_optimization_20261009/deployment/`；全 FP32 备选 `deployment_fp32/`（8 线程）。示例 `example_prediction/`。全部证据及图表在 `outputs/inference_optimization_20261009/`；方法、完整消融、复现命令和论文叙事边界见 [推理优化报告](docs/inference_optimization_2026-10-09.md)。本结果为单折部署验证，后续方法论主张需固定策略的五折或独立测试验证。


## PhysLeak / CDP：无监督稠密 FA 渗漏负荷与 5 项后续验证（2026-10-14 ~ 10-15，5 折全量 2,055 图）

**目标与口径**：不读 mask 或临床标签，给出稠密渗漏潜势图，并把它汇聚成图像级渗漏负荷。评估只在预测 sha256 冻结之后读取 mask。目标是 **mask 并集面积代理，不是临床分级**。指标为每折内先 rank 再 pooled Spearman ρ；cell 级（32×32）AUROC 以 cell 中 mask 占比 ≥0.5 为正类。bootstrap 按图像重采样；数据没有患者 ID，同一患者的多张图可能相关，CI 可能偏窄。对比基线是 10-13 的全局方法（DINOv3 CLS + 合成方向 + gamma 正交投影），ρ = 0.7065。

**方法（CDP，Counterfactual Difference Probe）**：
- 冻结 DINOv3 ConvNeXt-Tiny，取 stage1–3 特征，统一到 32×32 网格（C = 1536，输入 512）。
- 以 Hessian 脊线检出的血管段为源，物理合成渗漏反事实 I′：扩散 σ 1.5–8，幅度 0.1–0.5。
- 逐 cell 用岭回归拟合 w·(F(I′)−F(I)) ≈ ΔI。同时惩罚 β‖w·(F(N(I))−F(I))‖²，其中 N 为滋扰视图（整体增亮、gamma 0.8/1.2）。
- λ 只用训练折内的合成 SSE 来选。图像负荷取 FOV 内 mean、正部 mean（pos）或 top10%。
- 实现：`src/bs/physleak.py`、`scripts/physleak_{stats,fit_predict,evaluate,nested_select,select_beta}.py`。`tests/test_physleak.py` 5/5 通过。

### 1. 主实验与 β 扫描（ConvNeXt-T，seed 42/2026/3407 均值）

| 方法 | pooled ρ | 配对 Δ vs 先前 95%CI | lesion2 存在 AUROC | cell AUROC |
|---|---:|---|---:|---:|
| 先前全局方法（10-13） | 0.7065 | — | 0.581 | 无稠密图 |
| 强度 / DoG / PatchCore 式 kNN | 0.383 / 0.385 / 0.072 | 显著更差 | 0.562 / 0.556 / **0.913** | 0.811 / 0.727 / 0.513 |
| CDP β0（无滋扰惩罚，pos） | 0.464 | 显著更差 | 0.630 | 0.818 |
| CDP β1 mean（**预注册主方法**） | 0.487 | 显著更差 | 0.714 | 0.873 |
| CDP β3 pos | 0.715 | [−0.021, +0.025] | 0.697 | 0.891 |
| **CDP β10 pos** | **0.748±0.007** | **[+0.014, +0.057]** | 0.675 | **0.894** |
| CDP β30 pos | 0.741 | [+0.008, +0.049] | 0.669 | 0.887 |
| CDP β100 pos | 0.704 | [−0.030, +0.012] | 0.694 | 0.872 |

- **β10/pos 逐折 ρ**：0.671 / 0.775 / 0.755 / 0.760 / 0.798。先前方法为 0.633 / 0.751 / 0.744 / 0.729 / 0.677。
- **嵌套选择**：外层留一折，内层在 21 个 (β, 汇聚) 配置中按面积 ρ 选优，15/15 次都选中 β10/pos。嵌套 pooled ρ 为 0.7427 / 0.7556 / 0.7451（均值 0.748）。
- **滋扰惩罚是决定性组件**：gamma 四分位翻转率 β0 为 66–70%，β10 降到 2.6–3.0%。合成拟合 R² 从 β0 的 0.994 降到 β10 的 0.906。
- **生成器消融**：物理生成器优于 Gauss 斑块（0.665 vs 0.521，β1 top10）。
- **诚实说明**：预注册的 β1/mean 只有 0.487，未达标。β10/pos 是在看过 mask 后挑出的，属于后验选择。嵌套选择只能说明「在这 21 个配置里挑选的乐观偏差很小」，不能替代前瞻验证。

### 2. 后续验证 ① 监督上限（参考，非无监督）

HMVF+VS-UBR 5 折 OOF：768 输入，sigmoid ≥ 0.5，无 TTA，用同一 within-fold rank 协议评估。
- best.pt 所在 epoch：f1 28、f2 9、f3 23、f4 26、f5 8。f2 和 f5 偏欠训。

| 指标 | 监督 OOF | CDP β10/pos（seed42） |
|---|---:|---:|
| pooled ρ（面积代理） | **0.965** [0.962, 0.968] | 0.743 |
| 逐折 ρ | 0.970 / 0.978 / 0.978 / 0.965 / 0.973 | — |
| cell AUROC / AP | 0.982 / 0.846 | 0.894 / 0.445 |
| lesion2 存在 AUC | 0.960（lesion2 头最大概率） | 0.68 |

结论：这个代理指标在有标签时几乎可以完全学到，所以 CDP 与监督之间约 **0.22 ρ** 的差距是方法差距，不是标签噪声。

### 3. 后续验证 ② kNN–lesion2 混杂审计（seed42，`scripts/physleak_audits.py` A 部分）

| 指标 | patch_kNN | CDP β10 |
|---|---:|---:|
| lesion2 存在 AUC（全部） | 0.914 | 0.681 |
| 按来源 jpg / png / tif 分层 | 0.891 / 0.946 / 0.917 | 0.703 / 0.679 / 0.661 |
| 折内平均 | 0.914 | 0.676 |
| 面积五分位层内平均 | 0.919 | 0.683 |
| 来源（扩展名）η² | 0.016 | 0.029 |
| lesion2+ 图内：lesion2 cell vs lesion1-only cell AUC | 0.817 | 0.311 |
| lesion1 cell vs 背景 AUC | 0.468 | 0.877 |
| lesion2+ 图内与 lesion2 面积 ρ | 0.04 | 0.32 |

- kNN 的 0.91 不能用来源、折或总面积解释。在 lesion2+ 图像中，kNN 得分前 10% 的 cell 里 lesion2 占 10.4%（基础率 2.3%）。
- 它对 lesion1 和面积不敏感，本质上是 **lesion2 存在性检测器**，与跟踪 lesion1/面积的 CDP 互补。
- lesion2 的临床语义仍未经确认。

### 4. 后续验证 ③ 预注册 β 规则 R1 的前瞻检验（ViT-B/16，blocks 6/9/12，C = 2304）

- **R1 规则**：τ = 0.03，取 burden_pos 下滋扰翻转率 ≤ τ 的最小 β。
- **跑前登记的哈希**（`physleak_vitb16/protocol_sha256_before_run.txt`）：规则脚本 sha256 8334b965…，协议文档 d7df1796…。
- **两条预注册标准**：R1 均值 ≥ 0.7065；与 oracle 的差距 ≤ 0.03。

| seed | R1 选中 | ρ（burden_pos） | Δ vs 先前 CI 下界 |
|---|---|---:|---:|
| 42 | β30 | 0.6997 | −0.035 |
| 2026 | β10 | 0.7270 | −0.007 |
| 3407 | β30 | 0.7036 | −0.030 |

- **两条标准都通过**：R1 均值 0.7101 ≥ 0.7065；oracle 为 β3/pos 0.7304，差距 0.020 ≤ 0.03。
- **但通过得很弱**：3 个 seed 中有 2 个单独低于先前方法，也没有任何一个 seed 的 CI 排除 0。
- ViT 上 β0.3–β3/pos 是一个平台（0.7285–0.7304），R1 在 ViT 上选的 β 偏大。seed42 的翻转率：β0 0.677、β1 0.043、β10 0.031、β30 0.026。
- ViT 整体弱于 ConvNeXt：oracle 0.730 vs 0.748，cell AUROC 0.872–0.878。ViT 上 patch_kNN 的 ρ 为 0.22，lesion2 AUC 0.84。
- 结论：「无标签 β 规则不会输给先前方法」有初步前瞻证据；「规则能找到最优 β」不成立。

### 5. 后续验证 ④ CDP 分歧审计（ConvNeXt β10，seed42，B 部分）

- **假阳性**定义为折内 z 分数前 5% 的 mask 阴性 cell。
  - 视盘启发区占 FP cell 的 18.1%，而在全部阴性 cell 中只占 2.7%，**富集 6.8 倍**。FOV 边缘（1.0×）和血管（0.93×）无富集。
  - 注意：视盘启发区本身有 21.9% 的 cell 是 mask 阳性（其他区域 7.5%），所以这个定位并不精确。
- **图像残差**（score rank − area rank）：
  - **最被低估**的 50 张：面积中位数 0.162，FOV 更亮，即弥漫性大面积渗漏被低估。
  - **最被高估**的 50 张：面积中位数 0.013，其中 28% 为 lesion2+（全体为 11%）。
  - 面积最大的四分位内 ρ 只有 0.23；残差与 lesion1 面积的 ρ 为 −0.33。
  - 来源效应弱：平均残差 tif +0.013、jpg −0.016、png −0.034；按来源的 ρ 为 0.711 / 0.716 / 0.753。

### 6. 后续验证 ⑤ 外部数据

没有可访问的、带标签的独立葡萄膜炎 FA 队列（见 `docs/severity_external_validation_feasibility_20261013.md`）。Duke DME 数据链接本轮再次超时或返回 404。**未做外部验证。**

### 总结与局限

- **结论**：
  - CDP 在完全无标签的设定下给出了稠密渗漏图（cell AUROC 0.89，强度基线 0.81）。图像级面积代理 ρ = 0.748，配对显著优于先前的 0.707。
  - 换到第二个特征空间（ViT）后排序仍能复现（约 0.73），但离监督上限 0.965 很远。
  - 主要错误模式：低估弥漫性大面积渗漏；视盘假阳性。
  - kNN 提供了 CDP 缺少的 lesion2 存在性信息，可以考虑「CDP 面积 + kNN lesion2 存在」的双轴报告。这一组合是后验想法，需要重新预注册后才能检验。
- **局限**：
  - 目标只是 mask 面积代理，不能宣称临床分级。
  - 没有患者 ID，bootstrap 按图像进行。
  - 最佳 β 和汇聚方式是后验选择；R1 前瞻通过得很弱。
  - 只在单一数据集上验证，没有外部队列。
  - 监督上限只作参考（f2/f5 checkpoint 偏欠训）。

**产物**：
- 原始产物：`/root/autodl-fs/uveitis_severity_20261013/`
  - `physleak/`：主实验
  - `physleak_beta_sweep/`：β 扫描与嵌套选择
  - `physleak_vitb16/`：R1 前瞻检验，含 `r1_forward_test_result.json`
  - `physleak_followup/`：监督上限与审计，含 `audit_seed42/cdp_disagreement_cases.png`
  - fold_stats.pt 是中间缓存，已删除或未归档；稠密图 npz 只保留在 autodl-fs。
- 项目内精简副本：`outputs/physleak_severity_20261014/`
- 文档：
  - [PhysLeak 主结果](docs/severity_physleak_findings_20261014.md)
  - [β 规则预注册协议](docs/severity_physleak_beta_rule_protocol_20261014.md)
  - [5 项后续验证](docs/severity_physleak_followup_20261014.md)
- 归档 ZIP 见本文件「Experiment Archive Records」表中 PhysLeak-CDP 一行。

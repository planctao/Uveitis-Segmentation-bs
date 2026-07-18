# 实验记录

> 本文件记录每次训练实验的关键信息，用于权重版本管理。权重存储在 `runs/` 下但不纳入 git。

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
| MAE-SAM2 (本文) | Hiera-Small | SA-1B + MAE | 0.5983 (训练中) | 待完成 |

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

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


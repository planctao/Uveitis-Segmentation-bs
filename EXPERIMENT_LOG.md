# 实验记录

> 本文件记录每次训练实验的关键信息，用于权重版本管理。权重存储在 `runs/` 下但不纳入 git。

| # | 日期 | Run Name | Backbone | Config | Fold | Epochs | 关键指标 | 权重路径 | 改动摘要 | 备注 |
|---|------|----------|----------|--------|------|--------|----------|----------|----------|------|
| 1 | 2026-06-15 | dinov3_vitb16_1fold_768_20260615_1537 | ViT-B/16 | configs/dinov3_vitb16_multilabel_itksnap.yaml | f1 | 30 | best_macro_dice=0.7337; dice_1=0.7221; dice_2=0.7453 | runs/dinov3_vitb16_1fold_768_20260615_1537/f1/checkpoints/ | Baseline: DINOv3 ViT-B/16 + TokenFPNHead 无创新点; bs=1 grad_accum=4 768x768 | best_epoch=23; val_loss=0.552 |
| 2 | 2026-06-17 | dinov3_wbe_f1 | ViT-B/16 + WBE v1 | configs/dinov3_vitb16_multilabel_wbe.yaml | f1 | 30 | best_macro_dice=0.7248; dice_1=0.7179; dice_2=0.7318 | runs/dinov3_wbe_f1/f1/checkpoints/ | 新增小波边界增强WBE v1模块(4-scale, bottleneck=256, 6.44M参数); bs=8 grad_accum=2 | best_epoch=18; WBE未提点(-0.89%); batch_size不同影响公平对比 |
| 3 | 2026-06-17 | dinov3_wbe_v2_f1 | ViT-B/16 + WBE v2 | configs/dinov3_vitb16_multilabel_wbe_v2.yaml | f1 | 30 | best_macro_dice=0.7253; dice_1=0.7194; dice_2=0.7312; sweep_macro=0.7492 | runs/dinov3_wbe_v2_f1/f1/checkpoints/ | WBE升级v2: 借鉴PFESA加入SNR零参数边缘先验+Structure Attention+snr_gate自适应融合; bs=8 grad_accum=2 | best_epoch=13; 仍未超baseline(-0.84%); 过拟合严重(ep13后无提升) |
| 4 | 2026-06-25 | mae_sam2_f1_20260625_1404 | SAM2 Hiera-Small (MAE预训练) | configs/sam2_mae_multilabel.yaml | f1 | 50 | best_mae_val_loss=0.2365 | runs/mae_sam2_f1_20260625_1404/f1/checkpoints/best.pt | Stage 1 MAE自监督预训练: mask_ratio=0.75, MSE重建损失, lr=1e-4 warmup5+cosine; bs=4 768x768 AMP | best_epoch=50; encoder权重将用于Stage 2微调; train_loss=0.2131 |
| 5 | 2026-06-25 | mae_sam2_ft_f1_20260625_1404 | SAM2 Hiera-Small (分割微调) | configs/sam2_mae_multilabel.yaml | f1 | 50 (进行中) | best_macro_dice=0.5983; dice_1=0.6234; dice_2=0.5732 (epoch 18, 训练中) | runs/mae_sam2_ft_f1_20260625_1404/f1/checkpoints/best.pt | Stage 2 分割微调: 加载MAE encoder权重, 0.9 Dice+0.1 BCE损失; bs=4 768x768 AMP | 5折pipeline运行中(f1微调中, f2-f5待跑); 对比DINOv3 ConvNeXt 0.7710差距明显 |

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

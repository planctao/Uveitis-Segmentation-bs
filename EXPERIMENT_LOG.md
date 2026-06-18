# 实验记录

> 本文件记录每次训练实验的关键信息，用于权重版本管理。权重存储在 `runs/` 下但不纳入 git。

| # | 日期 | Run Name | Backbone | Config | Fold | Epochs | 关键指标 | 权重路径 | 改动摘要 | 备注 |
|---|------|----------|----------|--------|------|--------|----------|----------|----------|------|
| 1 | 2026-06-15 | dinov3_vitb16_1fold_768_20260615_1537 | ViT-B/16 | configs/dinov3_vitb16_multilabel_itksnap.yaml | f1 | 30 | best_macro_dice=0.7337; dice_1=0.7221; dice_2=0.7453 | runs/dinov3_vitb16_1fold_768_20260615_1537/f1/checkpoints/ | Baseline: DINOv3 ViT-B/16 + TokenFPNHead 无创新点; bs=1 grad_accum=4 768x768 | best_epoch=23; val_loss=0.552 |
| 2 | 2026-06-17 | dinov3_wbe_f1 | ViT-B/16 + WBE v1 | configs/dinov3_vitb16_multilabel_wbe.yaml | f1 | 30 | best_macro_dice=0.7248; dice_1=0.7179; dice_2=0.7318 | runs/dinov3_wbe_f1/f1/checkpoints/ | 新增小波边界增强WBE v1模块(4-scale, bottleneck=256, 6.44M参数); bs=8 grad_accum=2 | best_epoch=18; WBE未提点(-0.89%); batch_size不同影响公平对比 |
| 3 | 2026-06-17 | dinov3_wbe_v2_f1 | ViT-B/16 + WBE v2 | configs/dinov3_vitb16_multilabel_wbe_v2.yaml | f1 | 30 | best_macro_dice=0.7253; dice_1=0.7194; dice_2=0.7312; sweep_macro=0.7492 | runs/dinov3_wbe_v2_f1/f1/checkpoints/ | WBE升级v2: 借鉴PFESA加入SNR零参数边缘先验+Structure Attention+snr_gate自适应融合; bs=8 grad_accum=2 | best_epoch=13; 仍未超baseline(-0.84%); 过拟合严重(ep13后无提升) |

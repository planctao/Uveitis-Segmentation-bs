# 当前实验结果（2026-09-21）

## 评估口径

- 数据：`split_dataorigin`，768×768 FA 图像，验证折剔除文件名含 `_aug` 的副本。
- 任务：两通道 multilabel 分割；`lesion_1 = label 1 or 3`，`lesion_2 = label 2 or 3`。
- 主指标：验证集累计像素上的 threshold-sweep macro Dice。
- 除特别说明外，表中为 f1 单折结果。

## 已完成结果

| 实验 | Backbone / 模块 | 最佳 epoch | Macro Dice | Dice-1 | Dice-2 | 阈值 |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 baseline | DINOv3 ViT-B/16 + TokenFPN | 24 | 0.7350 | — | — | sweep |
| ViT + RDH-PDE | DINOv3 ViT-B/16 + RDH | 25 | 0.7101 | 0.7325 | 0.6877 | sweep |
| ViT + S3RD | DINOv3 ViT-B/16 + SSM/RDH | 3 | 0.6398 | 0.7017 | 0.5779 | sweep |
| ConvNeXt clean | DINOv3 ConvNeXt-Tiny + FPN | — | 0.7769 | — | — | sweep |
| VS-UBR | ConvNeXt-Tiny + VS-UBR (`vessel_weight=0.5`) | 24 | **0.7902** | 0.7976 | 0.7828 | shared sweep |
| HMVF | ConvNeXt-Tiny + HMVF | 21 | 0.780249 | 0.797852 | 0.762647 | 0.9 |
| HMVF + VS-UBR | ConvNeXt-Tiny + HMVF + VS-UBR | 28 | 0.777419 | 0.795973 | 0.758864 | 0.8 |

## HMVF + VS-UBR 独立阈值结果

在每个病灶通道允许使用独立阈值时，最佳结果为 epoch 28：

- Macro Dice：`0.779480`
- Dice-1：`0.795973`
- Dice-2：`0.762987`
- 阈值：`0.8 / 0.9`

这仍略低于单独 HMVF 的 `0.780249`，也明显低于单独 VS-UBR 的 `0.7902`。当前结论是：在 ConvNeXt-Tiny f1 实验中，HMVF 与 VS-UBR 没有表现出叠加增益。

## 五折参考结果

- ConvNeXt clean：shared sweep `0.7849 ± 0.0219`。
- VS-UBR（`vessel_weight=0.5`）：independent sweep `0.7881 ± 0.0194`。

HMVF 和 HMVF + VS-UBR 目前只有 f1 单折结果，尚未进行五折确认。

## 实验状态

- HMVF：已完成 30 epoch。
- HMVF + VS-UBR：已完成 30 epoch，最佳 checkpoint 为 epoch 28。
- MCADS + VS-UBR：正在运行，尚无最终指标，不纳入以上比较。

原始训练输出位于本地 `runs/` 目录。该目录按仓库策略被 `.gitignore` 忽略，因此本次提交只同步可读的结果报告，不同步 checkpoint 和大体积训练日志。

---
name: log-experiment
description: 在每次训练实验完成后，将实验结果、权重文件路径、改动模块、创新点等详细信息追加到实验记录表中。当 agent 执行了新的训练实验、修改了模型/损失/增强等模块后启动训练、或用户要求记录实验时触发。
---

# 实验记录 (Experiment Log)

## 触发时机

以下情况**必须**执行本 skill：

1. agent 帮用户启动了一次新训练（执行了 `train_*.py` 脚本）
2. agent 修改了代码模块后启动了训练
3. 用户明确要求 `/log-experiment` 或 "记录实验"
4. 用户告知实验已结束、贴出训练日志要求归档

## 记录文件

实验记录表位于：

```
Uveitis-Segmentation-bs/EXPERIMENT_LOG.md
```

如果文件不存在，先创建（包含表头）。每次记录**追加**一行到表格末尾，不要覆盖已有记录。

## 记录格式

文件使用 Markdown 表格，列定义如下：

```markdown
| # | 日期 | Run Name | Backbone | Config | Fold | Epochs | 关键指标 | 权重路径 | 改动摘要 | 备注 |
|---|------|----------|----------|--------|------|--------|----------|----------|----------|------|
```

各列说明：

| 列 | 内容 | 示例 |
|----|------|------|
| # | 自增序号 | 1, 2, 3... |
| 日期 | 实验启动日期 YYYY-MM-DD | 2026-06-17 |
| Run Name | `--run-name` 参数值 | dinov3_vitb16_5fold_20260617 |
| Backbone | 使用的 backbone | ViT-B/16 / ConvNeXt-Tiny / UNet-scratch |
| Config | 使用的配置文件（相对路径） | configs/dinov3_vitb16_multilabel_itksnap.yaml |
| Fold | 训练的折 | f1 / f1-f5 / f1,f3 |
| Epochs | 训练 epoch 数 | 30 |
| 关键指标 | best macro Dice 或其他核心指标 | macro_dice=0.761 |
| 权重路径 | checkpoint 所在目录（相对项目根） | runs/dinov3_vitb16_5fold_20260617/f1/checkpoints/ |
| 改动摘要 | 本次实验相比上次的**核心改动**，用简短描述 | 增加 ForegroundResizedCrop 增强; pos_weight 调为 [3,60] |
| 备注 | 额外信息（失败原因、中断说明、对比结论等） | NaN at ep15, 降LR后恢复 |

## 执行步骤

1. **收集信息**：从训练命令、配置文件、训练日志（`train.log`、`metrics.csv`）中提取信息。如果指标尚未产出（训练刚启动），`关键指标` 列填 `pending`，后续补填。

2. **确定改动摘要**：回顾本次会话中对代码做的修改（git diff 或对话记忆），提炼为 1-2 句话。如果是纯复跑没有代码改动，写"无代码改动，超参调整：xxx"。

3. **追加记录**：读取 `EXPERIMENT_LOG.md`，在表格末尾追加新行。

4. **如果文件不存在**，用以下模板创建：

```markdown
# 实验记录

> 本文件记录每次训练实验的关键信息，用于权重版本管理。权重存储在 `runs/` 下但不纳入 git。

| # | 日期 | Run Name | Backbone | Config | Fold | Epochs | 关键指标 | 权重路径 | 改动摘要 | 备注 |
|---|------|----------|----------|--------|------|--------|----------|----------|----------|------|
```

5. **通知用户**：记录完成后简短告知用户已更新实验记录，并提示权重路径。

## 补填指标

当训练结束后用户贴出结果或要求查看训练日志时，agent 应主动更新对应行的 `关键指标` 列（将 `pending` 替换为实际值）。

## 示例记录

```markdown
| 1 | 2026-06-15 | dinov3_convnext_tiny_5fold_20260615_0250 | ConvNeXt-Tiny | configs/dinov3_convnext_tiny_multilabel_itksnap.yaml | f1-f5 | 30 | best_macro_dice=0.761 | runs/dinov3_convnext_tiny_5fold_20260615_0250/ | 首次ConvNeXt 5折全量训练; save_interval=999优化磁盘 | f2磁盘满中断后恢复 |
| 2 | 2026-06-16 | dinov3_vitb16_1fold_768_20260616 | ViT-B/16 | configs/dinov3_vitb16_multilabel_itksnap.yaml | f1 | 30 | best_macro_dice=0.742 | runs/dinov3_vitb16_1fold_768_20260616/f1/checkpoints/ | 768vs640消融对比-768组 | 对比640结果后768胜出 |
```

## 注意事项

- 改动摘要是最重要的列——如果将来忘了改了什么，靠它恢复记忆
- 表格单元格内避免使用 `|` 字符，用 `/` 或 `;` 替代
- 指标优先记录 `paper_macro_dice`（论文指标），其次 `fg_mean_dice`
- 如果一次跑了多折，可以合并为一行（Fold 列写 f1-f5），也可以拆开写

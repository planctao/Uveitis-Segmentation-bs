# FA-ClickPCM B0–B4 小规模消融结果

日期：2026-08-30
实验目的：在统一条件下验证 `previous-mask → focus crop → Progressive Merge → 点击噪声鲁棒` 是否比 DINO 粗分割 baseline 更稳定。

## 运行设置

- fold：`f1`
- 训练：每组 5 epoch，训练子集 128 张，验证子集 64 张，batch size 2，seed 42
- 输入：复用 `outputs/dino_refiner_cache/vsubr_vw05_compact`，不复制或重新生成 DINO cache
- 评价：oracle error-click，报告 0/1/3/5 clicks 的两病灶 macro Dice
- 安全：训练前每个 variant/epoch 检查磁盘；实验期间剩余空间约 40.7 GiB，未触发安全阈值（10 GiB）

配置文件：`configs/fa_clickpcm_ablation.yaml`
训练入口：`scripts/train_interactive_correction.py`
模型实现：`src/bs/interactive_correction.py`

## 消融定义

| 组 | 结构 |
|---|---|
| B0 | DINO coarse mask，无交互 |
| B1 | 全图 previous-mask correction |
| B2 | B1 + 256×256 focus crop |
| B3 | B2 + bounded Progressive Merge |
| B4 | B3 + signed Gaussian prompts、2 px jitter、10% click dropout |

## 结果

| variant | best epoch | 0 click | 1 click | 3 clicks | 5 clicks |
|---|---:|---:|---:|---:|---:|
| B0 | 0 | **0.808227** | **0.808227** | **0.808227** | **0.808227** |
| B1 | 1 | 0.808227 | 0.807049 | 0.804763 | 0.802467 |
| B2 | 1 | 0.808227 | 0.808151 | 0.808028 | 0.807913 |
| B3 | 1 | 0.808227 | 0.808206 | 0.808194 | 0.808171 |
| B4 | 1 | 0.808227 | 0.808163 | 0.808038 | 0.807896 |

完整 JSON 汇总：`outputs/interactive_correction_ablation/fa_clickpcm_ablation_small/ablation_summary.json`。

## 初步结论

1. **B1 全图纠错不稳定**：click 数增加时 macro Dice 从 0.807049 降至 0.802467，说明即使显式加入 previous mask，全图更新仍会破坏正确区域。
2. **B2 明显更稳定**：focus crop 将 5-click 结果保持在 0.807913，远好于 B1，但仍未超过 B0。
3. **B3 最接近 baseline 且退化最小**：3-click 为 0.808194，较 B0 仅低约 0.0033 个百分点；5-click 仍仅低约 0.0056 个百分点。说明 Progressive Merge 确实抑制了多轮漂移，但当前训练预算下没有产生可观增益。
4. **B4 没有带来额外收益**：点击噪声鲁棒训练下 3/5-click 略低于 B3，当前不能声称真实点击噪声已经改善性能。

## 解释与下一步

这是一次小规模筛选，不足以支持完整五折训练。当前结果更支持以下判断：

- 应保留 B2/B3 的局部更新与保守 merge，放弃 B1 式全图 residual；
- 在进入完整训练前，需要检查 click simulator 与训练目标是否匹配，并考虑使用更有信息量的 focus crop（例如 click 周围与 coarse-error 的联合 crop）；
- 应加入 Boundary F1、面积误差和 calibration 指标，避免仅凭 Dice 的微小差异做结论；
- 只有在修正训练/提示生成后 B3/B4 在更大 f1 子集上超过 B0，才值得继续 policy 和五折实验。

本实验未修改数据集，也未产生新的大缓存；各组 checkpoint 均小于 1 MB。

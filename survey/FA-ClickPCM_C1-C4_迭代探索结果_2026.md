# FA-ClickPCM C1–C4 迭代探索结果

日期：2026-08-30
目的：针对 B0–B4 中“局部更新稳定但没有提升”的现象，验证训练闭环、局部损失、prompt 编码和 crop 上下文是否是主要瓶颈。

## 实验设置

- fold：`f1`
- 训练/验证子集：128 / 64
- epoch：5；batch size：2；seed：42
- DINO cache：复用 `outputs/dino_refiner_cache/vsubr_vw05_compact`
- 验证：动态 oracle error-click，报告 0/1/3/5 clicks 的 macro Dice
- 磁盘：实验期间剩余空间约 40.7 GiB，未生成新的大缓存

配置：`configs/fa_clickpcm_refinement_scan.yaml`
入口：`scripts/train_clickpcm_refinement_scan.py`

## 变体

| 变体 | 改动 |
|---|---|
| C1 | 真正的多轮 rollout；每轮用上一轮预测重新生成下一次 error click |
| C2 | C1 + click 邻域局部 correction loss |
| C3 | C2 + signed Gaussian prompt 与显式 distance/reachability channels |
| C4 | C3 + 384×384 focus crop |

## 结果

基准 B0 macro Dice 为 `0.808227`。

| variant | best epoch | 0 click | 1 click | 3 clicks | 5 clicks |
|---|---:|---:|---:|---:|---:|
| C1 | 1 | 0.808227 | 0.808206 | 0.808148 | 0.808123 |
| C2 | 1 | 0.808227 | 0.808181 | 0.808143 | 0.808076 |
| C3 | 1 | 0.808227 | **0.808215** | **0.808182** | **0.808143** |
| C4 | 1 | 0.808227 | 0.808190 | 0.808129 | 0.808067 |

完整结果：`outputs/interactive_correction_refinement_scan/fa_clickpcm_refinement_scan_small/refinement_summary.json`。

## 结论

1. 真正的多轮 rollout 没有解决“点击后不增益”的问题，说明瓶颈不只是训练/验证 rollout 不一致。
2. 局部 correction loss 没有带来提升，反而使 5-click 结果略降，当前局部监督权重可能仍然在追逐噪声。
3. C3 的 signed Gaussian/distance prompt 是最接近 baseline 的变体，但 3-click 仍比 B0 低约 `0.0045pp`，不能视为有效提升。
4. 增大 crop 到 384 没有收益，说明问题不太可能只是局部上下文不足。

## 决策建议

在当前数据和 DINO 粗分割质量下，不再继续堆叠 correction 网络结构，也暂不启动五折训练。下一阶段应做两个低成本方向：

- **直接交互编辑上限**：用点击位置对 coarse mask 做局部形态学/区域生长修正，测量“用户点击本身”能否改善面积误差和 Boundary F1；
- **策略型系统**：让 uncertainty 只负责推荐需要确认的区域和停止时机，把医生点击用于局部 mask 编辑，而不是训练一个可能学成 identity 的 residual refiner。

如果直接编辑上限也不能改善临床相关指标，则应把毕设主贡献转为 DINO baseline 的不确定性校准、面积量化和交互可视化，而不是继续追求点击 Dice 增益。

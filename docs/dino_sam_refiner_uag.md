# UAG-SAM：面向 FA 渗漏的可部署交互细化方案

本文档是对现有 DINO+SAM-style MVP 的问题审计，以及下一阶段毕设创新方案的实验设计。当前只完成代码闭环和合成张量测试；没有把未经训练的 UAG 结果写成性能结论。

## 1. 现有方案到底做了什么

现有系统是一个两阶段模型：

1. DINOv3 ConvNeXt-Tiny + VS-UBR 产生两通道粗分割；
2. 从粗分割与标注的残差中生成正/负点击，用 13 通道 U-Net 风格 `InteractiveResidualRefiner` 预测残差 logits。

已有 MVP 五折结果显示，oracle click 数从 0/1/3/5 时 macro Dice 为 0.7809/0.7853/0.7918/0.7965，说明“粗分割 + 交互修正”方向有价值。但这些数字应准确称为 **oracle-click 离线上限曲线**，不能直接等价为医生真实点击后的部署性能。

## 2. 主要问题与不足

### 2.1 名称上的 SAM 与实际模型不一致

系统没有使用 SAM/SAM2 的 image encoder、mask decoder 或 prompt encoder，只借用了正负点的提示思想。这个选择在 FA 域是合理的（直接微调 SAM2 的 clean-f1 macro 只有 0.6236），但论文中应称为“DINO + SAM-style prompt refiner”，不能声称是 SAM2 分割器。

### 2.2 oracle 点击造成评估偏乐观

`oracle_refine_with_target` 用 GT 残差选择下一点击；训练的 `simulate_click_heatmaps` 也直接读取 GT。部署函数 `refine_with_clicks` 虽然不读 GT，但没有自动找错策略。因此现有曲线回答的是“如果医生每次准确点在错误像素附近会怎样”，没有回答“医生应该点哪里、点几次可以停止”。

### 2.3 点击表示过硬且缺少方向/可靠度

旧特征把点击渲染成二值 disk，并把正、负图直接拼接。硬阈值 candidate 会丢失边界概率、距离和提示强度；点击半径固定，也没有模拟医生点偏 1–3 像素、漏点或重复点的情况。结果容易在实验室 oracle 点击下有效，遇到真实标注噪声时退化。

### 2.4 细化器只看缓存 logits，不看 DINO 中间表征

缓存中保存的是图像、GT 和最终 logits。细化器因此只能从 13 个像素级通道重建边界，无法利用 ConvNeXt 的高分辨率多尺度特征、血管结构或 RDH 的传播状态；它更像一个后处理 U-Net，而不是与粗分割器协同的 prompt-conditioned decoder。

### 2.5 残差更新没有显式的点击一致性约束

`dino_logits + delta_logits` 的更新没有保证正点击附近概率上升、负点击附近概率下降，也没有单调性或早停置信度。多轮迭代可能出现“点击有效但另一处被破坏”的情况，当前 Dice 曲线无法区分局部修正与全局漂移。

### 2.6 指标和临床交互证据不足

当前主指标是 paper Dice，主要报告 0/1/3/5-click；缺少 boundary Dice/HD95、稀有 lesion_2 的召回、ECE/Brier 校准、每图点击数、点击后收益和停止率，也没有 oracle-policy gap 或真实/扰动点击鲁棒性。这些缺口会削弱毕设的临床可用性论证。

### 2.7 小数据和数据卫生风险

项目历史曾发现 `_aug` 副本混入验证集，修正后 macro 由 0.7913 调整为 0.7830。交互细化器还依赖离线缓存，必须保证缓存严格按折生成，且 train/val 使用同一 DINO checkpoint 口径；否则点击收益会被重复病例或 checkpoint 泄漏放大。

## 3. 新方案：UAG-SAM v1

UAG（Uncertainty-Aware Guided）不是再堆一个大 backbone，而是把交互闭环补完整：

```text
DINO/VS-UBR 粗 logits
        │
        ├─ 不确定性 + 局部边界 → 自动正/负点击建议（无 GT）
        ├─ 医生点击/建议点击 → 高斯热图 + signed prompt + reliability
        └─ UAG residual refiner → residual logits + DINO logits
```

### 3.1 软提示和可靠度编码

`build_soft_prompt_features` 保留旧 13 通道，并增加每个 lesion 通道的：

- `signed = positive - negative`：明确点击方向；
- `reliability = (positive + negative) × uncertainty`：不确定区域的提示影响更大，避免高置信误点主导全局更新。

因此输入为 17 通道。推荐使用 `mode: gaussian`，让提示影响随距离平滑衰减，而不是固定二值 disk。

### 3.2 不确定性驱动的自动点击策略

`recommend_click_points` 完全不读取 GT。对当前概率 `p` 计算：

```text
u = 1 - 2|p - 0.5|
b = |p - AvgPool(p)|
s+ = u·max(0, 0.5-p)(0.5 + norm(b))
s- = u·max(0, p-0.5)(0.5 + norm(b))
```

`s+` 产生疑似漏检的正点击，`s-` 产生疑似误检的负点击，再用非极大值抑制保证空间分散。若两类 priority 都低于 `min_priority`，`should_stop_interaction` 会建议停止继续点击。该策略不是 GT oracle，适合作为真实部署的可复现 baseline；医生点击仍可通过 `refine_with_clicks` 覆盖。

### 3.3 不确定性门控残差网络

`UncertaintyGatedResidualRefiner` 仍然是轻量 U-Net，但：

- 用 uncertainty map 生成空间 gate，增加边界/低置信区域的局部容量；
- 用 signed/reliability 的全局统计生成 FiLM 调制，令网络知道本轮提示是“扩张”还是“收缩”；
- residual head 继续零初始化，保证新 checkpoint 初始时严格保持 DINO logits，不会因为新模块随机扰动 baseline。

### 3.4 点击噪声鲁棒和局部一致损失

`perturb_click_points` 支持像素级 jitter 和 click dropout；UAG 配置默认 jitter=2px、dropout=0.10。`click_consistency_loss` 对正点击施加正 logit margin、对负点击施加负 logit margin，约束点击的局部方向。

在 iterative 训练中，最终一轮累计点击图也会回填到辅助损失；这避免旧实现里 `features=None` 导致 click-BCE/一致性项被静默关闭。

### 3.5 迭代稳定策略（stability ablation）

多轮更新使用统一的残差控制入口 `_controlled_residual_update`：

- `residual_step_limit > 0`：将每轮 residual logit 通过 `limit * tanh(delta / limit)` 限制在单步范围内，避免累计更新把前一轮误差放大；
- `stop_gradient: true`：每轮用 detached 的 logits 和 prompt features 作为状态，切断跨轮反向传播，降低 rollout graph 的梯度放大；
- `teacher_forcing_ratio > 0`：训练时仅在两轮之间将状态向有界的 target teacher logits 做比例混合，最后一轮仍完全由 refiner 产生；验证和部署默认 `eval_teacher_forcing_ratio: 0`，不泄漏标注。

这些选项均在 `clicks` 下配置，默认关闭以兼容旧实验。稳定性复验应固定 batch、base_channels、seed 和样本子集，只改变上述开关。

## 4. 代码入口

| 文件 | 用途 |
|---|---|
| `src/bs/click_simulator.py` | 点击噪声、软提示、无 GT 点击策略和停止分数 |
| `src/bs/interactive_refiner.py` | UAG refiner、policy loop、点击一致损失 |
| `scripts/train_interactive_refiner.py` | 兼容旧 MVP，并按配置选择 UAG 模型/特征 |
| `scripts/evaluate_interactive_policy.py` | 评估不读取 GT 的 policy-click 曲线 |
| `configs/dino_sam_refiner_uag.yaml` | UAG v1 训练配置 |
| `tests/test_interactive_uag.py` | 17 通道、零初始化、无 GT policy、噪声和梯度测试 |

## 5. 建议实验矩阵

所有结果使用 clean validation（剔除 `_aug`）、同一五折 DINO cache 和同一 threshold sweep 口径。

| 组别 | 变体 | 目的 |
|---|---|---|
| A | 旧 13ch + random oracle | 复现 MVP |
| B | 17ch soft prompt + oracle | 测试提示编码贡献 |
| C | B + jitter/dropout | 测试标注噪声鲁棒性 |
| D | C + uncertainty gate/FiLM（UAG） | 测试网络创新贡献 |
| E | D + policy clicks | 报告无 GT 自动闭环 |
| F | D 的医生点击模拟 | jitter=0/2/4px、dropout=0/0.1/0.2 |

每组至少报告：`macro Dice`、`dice_1`、`dice_2`、boundary Dice、HD95、ECE/Brier、平均点击收益 `ΔDice/点击`、达到 95% 最终 Dice 所需点击数，以及 oracle-policy gap。

## 6. 进入论文主线的门槛

UAG 不应因为单折偶然涨点就成为主结论。建议满足以下条件再进入主表：

1. 五折 macro 至少不低于 VS-UBR baseline，且 3/5-click 曲线单调；
2. lesion_2 Dice 不出现系统性下降；
3. noisy-click 相对 exact-click 的下降小于 1.5pp；
4. policy-click 与 oracle-click 的 gap 有量化结果，而不是只展示可视化；
5. 推理显存、延迟和参数量相对 DINO 粗模型增量可报告。

若 UAG 未达到门槛，仍可作为“从 oracle 交互走向部署闭环的失败/部分成功探索”，保留旧 MVP 的正向结果，不把未经验证的模块包装成提升。

## 7. 推荐运行顺序

先做静态检查和小样本 smoke：

```bash
PYTHONPATH=src python -m py_compile \
  src/bs/click_simulator.py src/bs/interactive_refiner.py \
  scripts/train_interactive_refiner.py scripts/evaluate_interactive_policy.py

PYTHONPATH=src python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner_uag.yaml \
  --fold f1 --epochs 1 --max-train-samples 8 --max-val-samples 4 \
  --batch-size 2 --num-workers 0
```

数据集和 DINO cache 就绪后，再按 f1→f5 顺序完成正式训练；每折同时运行 oracle evaluator 和 `evaluate_interactive_policy.py`，最后汇总曲线与点击效率。

无 GT policy 的单折评估示例：

```bash
PYTHONPATH=src python scripts/evaluate_interactive_policy.py \
  --config configs/dino_sam_refiner_uag.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_uag_v1/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,1,3,5 --stop-priority 0.05 \
  --output outputs/interactive_eval/uag_policy_f1.json
```

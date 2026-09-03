# DINO + SAM-style 多次点击细化

仓库里的这组实验不是把通用 SAM2 直接当作最终分割器，而是采用更适合 FA 渗漏的两阶段方案：

1. DINOv3 ConvNeXt-Tiny 先给出两通道病灶粗分割；
2. 根据粗分割的漏检/误检生成 SAM 风格的正、负点提示，并把累计点击交给一个残差细化器。

这一区分很重要：项目已有的直接 SAM2 微调结果（见 `EXPERIMENT_LOG.md`）受到 FA 域差异影响，低于 DINO 基线；交互式方案只使用 SAM 的 prompt 思路，最终语义分割由 DINO/refiner 完成。

## 从 `split_dataorigin.zip` 开始

压缩包内已经包含 `dataset/dataset/split_dataorigin/...` 目录结构。在项目根目录解压即可；`dataset/` 只作为只读输入：

```bash
cd /root/autodl-tmp/Uveitis-Segmentation-bs
unzip -q /root/autodl-tmp/split_dataorigin.zip -d .
```

把 DINOv3 ConvNeXt-Tiny 预训练权重放到配置指定的位置（权重不会提交到 git）：

```bash
mkdir -p weights
cp /root/autodl-tmp/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
```

先确认数据配对和环境：

```bash
python scripts/check_environment.py
python scripts/index_dataset.py --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml
```

## 复现实验

下面的命令按折训练 DINO 粗分割。每一折的验证集只使用该折，另外四折用于训练；这和 `cache_dino_predictions.py` 的缓存逻辑一致。

```bash
for fold in f1 f2 f3 f4 f5; do
  python scripts/train_dinov3_multilabel.py \
    --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
    --run-name vsubr_vw05_${fold} --fold ${fold}
done
```

缓存 DINO logits、图像和标签，供细化器重复使用。默认 `uint8` 图像缓存会在加载时恢复 ImageNet 归一化，并且每个样本都强制复制为独立连续存储，避免把整个 batch backing storage 重复写入：

```bash
python scripts/cache_dino_predictions.py \
  --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
  --checkpoint-template 'runs/vsubr_vw05_{fold}/{fold}/checkpoints/best.pt' \
  --output-root outputs/dino_refiner_cache/vsubr_vw05_compact \
  --folds f1,f2,f3,f4,f5 --splits train,val --image-storage uint8
```

按仓库原始 MVP 口径训练交互细化器（默认 `0/1/3/5` 次点击）：

```bash
python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner.yaml
```

原始配置使用 `strategy: random`、`cumulative: false`，用于和 `EXPERIMENT_LOG.md` 中已经记录的 MVP 数字对齐。严格的 SAM 风格闭环配置使用 `configs/dino_sam_refiner_iterative.yaml`：它用 `strategy: farthest` 选择空间分散的误差点，并保留历史点击；每一轮先运行 refiner，再从最新残差选择下一点：

```bash
python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner_iterative.yaml
```

已有 checkpoint 可以单独评估：

```bash
python scripts/evaluate_interactive_refiner.py \
  --config configs/dino_sam_refiner.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,1,3,5 --output outputs/interactive_eval/f1.json
```

## MVP 初始化的 soft prompt adapter

随机初始化 17 通道 refiner 容易破坏原有 MVP 的粗分割能力。训练器支持
`train.init_checkpoint`：当模型只发生 13→17 的输入通道扩展时，复制 MVP 的
旧 stem 权重并将新增四个 signed/reliability 通道置零，其他形状不匹配会直接
报错，避免实验意外从不相关权重开始。打开
`train.prompt_adapter_only` 后，原有参数被冻结；默认只额外训练扩展 stem 的
新通道和 residual output projection，并对 stem 权重注册梯度 mask，保证旧 13
通道不被修改。

五折复现实验（已有 MVP checkpoint，不需要重新训练粗分割器）示例：

```bash
for fold in f1 f2 f3 f4 f5; do
  python scripts/train_interactive_refiner.py \
    --config configs/dino_sam_refiner_soft_prompt_adapter_f1.yaml \
    --fold "$fold" \
    --project-name "dino_sam_refiner_soft_prompt_adapter_${fold}" \
    --init-checkpoint "outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full/${fold}/checkpoints/best.pt"
done
```

上面的命令是完整复现实验入口；本节已报告的五折数字中，f2–f5 为两 epoch
筛选 checkpoint，若要严格复现这些数字请使用对应的
`dino_sam_refiner_soft_prompt_adapter_f{2..5}_e2` 输出目录，或把
`--epochs 2` 作为筛选阶段设置。

评估时要固定 one-shot 口径（`iterative=false`）、seed、点击策略、半径和阈值
网格。例如：

```bash
PYTHONPATH=src python scripts/evaluate_interactive_thresholds.py \
  --config configs/dino_sam_refiner_soft_prompt_adapter_f1.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_soft_prompt_adapter_f1/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,3,5 --thresholds 0.5,0.6,0.7,0.8,0.9,0.95 \
  --radius 8 --strategy random \
  --output outputs/interactive_eval/soft_prompt_adapter_f1_thrscan.json
```

当前五折配对结果为 click3 `0.80006±0.02157`、click5 `0.80551±0.02289`，相对
MVP 的 `0.79920±0.02154` / `0.80470±0.02292` 分别仅提升 +0.086pp/+0.080pp。
因此该模块应作为轻量增量适配消融报告，主表仍保留 MVP。若要求 0-click 输出
逐像素严格不变，可在配置中设置
`train.prompt_adapter_train_output_projection: false`；此 stem-only 版本更
保守，但当前单折结果低于可训练 projection 的 adapter。

一个更有效的部署侧组合是保留该 adapter checkpoint、把点击半径改为 12（训练
仍使用 radius=8），再做同样的阈值校准：

```bash
PYTHONPATH=src python scripts/evaluate_interactive_thresholds.py \
  --config configs/dino_sam_refiner_soft_prompt_adapter_r12_eval.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_soft_prompt_adapter_f1/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,3,5 --thresholds 0.5,0.6,0.7,0.8,0.9,0.95 \
  --radius 12 --strategy random \
  --output outputs/interactive_eval/soft_prompt_adapter_f1_r12_fullthrscan.json
```

五折完整阈值网格复核的 click3/click5 为 `0.80229±0.02229` /
`0.80678±0.02294`，相对 radius=8 MVP 提升 +0.309pp/+0.208pp；逐折结果以
`outputs/interactive_eval/soft_prompt_adapter_f{1..5}_r12_fullthrscan.json` 为准。该
半径调整不引入可学习参数，应在论文中标记为 inference-side prompt-footprint
calibration，而不是网络结构涨点。

## MVP 的参数无关 residual gate

在原始 MVP 的 residual 输出上新增了一个可选的 `uncertainty_click` gate。它不增加可学习参数，只按当前 DINO 不确定性和正/负点击热图缩放 residual：高置信、未点击区域保留较小更新（默认 floor=0.25），不确定或点击附近允许完整更新。默认配置仍为 `residual_gate: none`，因此旧 checkpoint 的原始结果不变。

已有 MVP checkpoint 可以直接做配对复评，无需重训：

```bash
PYTHONPATH=src python scripts/evaluate_interactive_thresholds.py \
  --config configs/dino_sam_refiner.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,1,3,5 --thresholds 0.5,0.7,0.8,0.9 \
  --residual-gate uncertainty_click --gate-floor 0.25 --gate-click-gain 0.75 \
  --output outputs/interactive_eval/mvp_gate_f1.json
```

`scripts/summarize_interactive_gate.py` 可把五折 baseline/gated JSON 汇总成 CSV。正式报告应固定点击模拟随机种子，并同时给出原始 0.5 阈值和验证集校准阈值；gate 的部署安全 policy 评估仍使用 `evaluate_interactive_policy.py`，不读取 GT 选点。

注意：`evaluate_interactive_thresholds.py` 会严格读取 checkpoint 中的
`clicks.iterative`。原始 MVP (`iterative=false`) 是“一次采样 K 个点→一次
residual 更新”；只有 iterative 配置才是每轮根据最新预测重新采样并更新。两种
口径不能混合比较。输出 JSON 会记录 seed、strategy、radius、iterative 和 gate
配置，便于论文审计；`scripts/summarize_interactive_oneshot.py` 可汇总五折
mean±std。

在当前 compact MVP 五折复评中，修正后的 one-shot random oracle-click 结果为
click0/3/5 = **0.7879±0.0196 / 0.7992±0.0215 / 0.8047±0.0229**。这些数值是
论文中 MVP 主表应采用的口径；旧的 gate 小节历史数字不再作为增益证据。

对严格 iterative rollout，可在不改权重的情况下启用稳定化校准：
`--residual-step-limit 0.5` 限制单轮 logit 更新，`--total-residual-limit 1.0`
进一步限制相对初始 DINO 预测的累计漂移。后者以初始 DINO logits 为锚点，
因此会在每轮 rollout 更新中生效；这是参数无关的部署安全策略，不能和重新训练
带来的模型增益混写。当前 f1 配对实验显示两者能显著缓解 iterative 退化，但仍
略低于原始 MVP 的一次性 oracle-click 曲线。

对应的可复用配置是 `configs/dino_sam_refiner_iterative_stable.yaml`；把它与已有
MVP checkpoint 一起传给 `evaluate_interactive_thresholds.py` 或
`evaluate_interactive_policy.py` 即可复现实验，不需要重新训练权重。

`uncertainty_click_channelwise` 是按 lesion 通道独立门控的候选实现：点击
lesion_2 不会无意中放大 lesion_1 的 residual。它只适用于两通道输出，默认不启用，
需在独立配对消融中确认后再纳入主线。

部署侧的自动点击推荐仍应单独报告，入口是
`scripts/evaluate_interactive_policy.py`。该脚本不读取标签来选点；可用
`--output-threshold 0.9` 做固定输出校准，并用同样的
`--residual-step-limit 0.5 --total-residual-limit 1.0` 限制闭环漂移。其结果只能
说明无 GT policy 的可部署性能，不能与 oracle-click 曲线直接混为一谈。比较不同
策略时固定 `--seed`，以便复用同一套自动点击随机流。

## 轻量提示足迹校准（MVP-R12）

FA 渗漏的边界通常比单个像素点击更弥散。MVP 细化器本身不变，只把正/负点击的渲染半径从 8 调到 12，并在输出端固定使用验证集校准得到的 0.9 阈值。配置见 `configs/dino_sam_refiner_mvp_r12.yaml`；该配置的训练和推理参数量与原始 MVP 完全相同，适合 2080 Ti。

`clicks.radius` 现在同时接受标量和每病灶列表，例如 `radius: [12, 8]`。列表形式用于验证“弥散 lesion_1 使用较宽提示、稀有小 lesion_2 使用较窄提示”的形态先验；旧配置的标量半径行为保持不变。离线复评可直接运行：

```bash
PYTHONPATH=src python scripts/evaluate_interactive_thresholds.py \
  --config configs/dino_sam_refiner.yaml \
  --checkpoint outputs/interactive_refiner_runs/dino_sam_refiner_vsubr_vw05_compact_full/f1/checkpoints/best.pt \
  --fold f1 --clicks 0,3,5 --radius 12 \
  --thresholds 0.8,0.9 --batch-size 4
```

这是推理侧的参数校准，不应和重新训练的模型增益混写；论文中应同时报告原始 radius=8 和 MVP-R12 的固定阈值结果，并说明阈值只在训练折/校准集确定。

若需要强制系统在医生点击位置遵循正/负极性，可在离线评估中加入
`--click-constraint-strength 8`。这是只修改点击覆盖像素的安全投影，不是新的
可学习参数；应作为交互安全性消融，不作为 backbone 的性能提升来表述。

评估器还支持两个不改权重的校准开关。`--residual-channel-scale 1.0,1.2`
可以分别缩放 lesion_1/lesion_2 的 residual；当前五折配对结果只带来约
0.03–0.05pp 的均值变化，未纳入主配置。若配置文件包含
`adaptive_threshold`，评估器会按每张图的概率分位数计算阈值；在本项目的
lesion_2 APQT 试验中反而低于固定 0.9 阈值，因此默认关闭。形态学后处理也可
通过 `postprocess` 配置启用，但当前实现的连通域操作是 CPU 侧离线诊断，速度
较慢，不能作为默认部署路径。

## 训练/部署边界

离线 Dice 曲线为了客观比较，使用标签生成 oracle 误差点；这部分只用于评估，不能在真实部署时使用。部署时调用 `bs.interactive_refiner.refine_with_clicks`，将医生实际点击的 `[B,C,K,2]` 坐标传入即可。该函数不读取标签，并返回每轮 logits 历史，适合导出 0/1/3/5-click 可视化。

`train_interactive_refiner.py` 会把输出写入 `outputs/interactive_refiner_runs/`，包括每折的 `metrics.csv`、`train.log` 和 `checkpoints/{best,latest}.pt`。缓存和权重也都位于 `outputs/`/`weights/`，不会改写数据集。

# DiffLeak：面向 FA 渗漏的双物理先验分割模型

## 故事线

荧光素眼底血管造影中的渗漏不是普通的静态纹理：它有一个起漏种子，随后在组织内扩散，并受到血管/视野边界的阻挡；而小面积的 lesion_2 在训练集中又极少出现。DiffLeak 把这两个事实放进同一条训练故事：

1. **DALS（Diffusion-Aware Lesion Synthesis）** 在训练阶段生成“像渗漏一样扩散”的稀有病灶样本。
2. **RDH（Reaction-Diffusion Head）** 在前向阶段把 decoder 预测的起漏种子演化为最终病灶概率图。

因此模型不是单纯增加卷积层，而是同时回答“训练时如何看见更多稀有渗漏”和“推理时渗漏如何从种子传播”两个问题。

## 创新点 1：DALS

`src/bs/leakage_synthesis.py` 从训练折掩码中建立实例库，只收集 lesion_2 连通域。每次增强执行：

1. 对实例做尺度、翻转和旋转扰动；
2. 用高斯热核生成外观/浓度扩散 halo；
3. 将扩散外观与目标掩码同步写回；与 lesion_1 重叠时自动生成 label 3；
4. 只在训练集启用，验证集绝不参与实例库构建。

这比普通 copy-paste 多了“外观扩散和标签扩散耦合”，目标是提高极端稀有 lesion_2 的召回率，而不是制造硬边缘贴图。

## 创新点 2：RDH

设 decoder 特征为 `z`，起漏种子为 `s=sigmoid(W_s z)`，特征/荧光共同决定传导率 `c`。PDE 版本迭代：

```text
u_0 = s
u_{t+1} = clamp(u_t + dt * (div(c * grad(u_t))
              + rho * s * u_t * (1-u_t) - lambda * u_t), 0, 1)
```

`c` 使用 Perona-Malik 型梯度抑制，强荧光边界处传导变小。`rdh.stable_constraints=true` 的 RDH-v2 进一步将 `dt` 限制到二维显式扩散稳定区间，并用 softplus 保证 `rho/lambda` 非负；`flux_scheme=edge` 使用相邻像素传导率平均值离散 `div(c grad u)`。所有步骤可微，且只增加约千级参数。

## 训练和比较协议

- 数据：`mask_only_itksnap`，验证折剔除 `_aug` 离线增强副本。
- 输入：`768x768`；ConvNeXt-Tiny DINOv3 backbone。
- 指标：两通道 paper Dice 的 macro；新实验固定 `threshold=[0.90,0.90]`，关闭每 epoch threshold sweep。
- 既有 clean-f1 ConvNeXt 门槛：macro Dice `0.7769`；既有 ViT 门槛：`0.7350`。
- 只让超过 `0.7769` 的创新路线进入后续跨折训练；汇总器：

```bash
PYTHONPATH=src python scripts/summarize_innovation_runs.py \
  outputs/runs/<run-a>/summary.json outputs/runs/<run-b>/summary.json \
  --output-md outputs/reports/innovation_screen.md \
  --output-json outputs/reports/innovation_screen.json
```

## 2080 Ti 部署约束

部署导出不携带 optimizer：

```bash
python scripts/export_deployment_checkpoint.py \
  --checkpoint outputs/runs/<winner>/f1/checkpoints/best.pt \
  --output outputs/deployment/diffleak_fp16.pt --precision fp16
```

推理基准：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/benchmark_inference.py \
  --checkpoint outputs/deployment/diffleak_fp16.pt \
  --warmup 10 --iterations 50 --output outputs/deployment/benchmark.json
```

现有 RDH f1 checkpoint 的 FP16 batch-1 代理实测：30.75M 参数、58.7 MiB 自包含权重、`1x3x768x768` 前向峰值 allocated `0.53 GiB`，A100 上平均 `11.5 ms`。这是 2080 Ti 的显存代理验证，不冒充真实 2080 Ti 延迟测试；模型使用标准 ConvNeXt、卷积、插值和逐元素算子，可在 11 GiB 卡上留有充分余量。

## 相关工作定位

- Perona & Malik, *Scale-space and edge detection using anisotropic diffusion*, IEEE TPAMI, 1990：RDH 的边缘阻挡传导启发。
- Fisher-KPP reaction-diffusion：RDH 的种子驱动增长项启发。
- Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*, CVPR 2021：DALS 的实例合成基线；DALS 增加了 FA 渗漏热扩散和多标签重叠规则。
- Bai et al., *Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation*, CVPR 2023, arXiv:2305.00673：医学分割中的 copy-paste 参照。该方法解决半监督分布差异，DALS 则针对全监督 FA 稀有病灶和扩散外观。
- Aranya & Desai, *SRA-Seg: Synthetic to Real Alignment for Semi-Supervised Medical Image Segmentation*, arXiv:2602.02944, 2026：最新合成医学分割工作也指出硬拼接边界会造成 synthetic-real gap，并采用 soft edge blending；DALS 的热核 halo 与此问题意识一致，但不使用 teacher/pseudo-label 或 DINOv2 对齐。
- *PICS in Pics* (arXiv:2311.07002) 与 *HNOSeg-XS* (arXiv:2507.08205) 是 PDE/神经算子医学分割参照；它们分别面向活动轮廓和分辨率鲁棒 3D 分割，不建模 FA 起漏种子到渗漏区域的反应扩散演化。
- Oquab et al., *DINOv2* 以及 DINOv3 官方预训练模型：视觉基础 backbone；本项目创新集中在 FA 先验和轻量分割头，而不是重新声称 backbone 创新。

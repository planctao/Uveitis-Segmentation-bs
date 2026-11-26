# AAAG-UNet、Mamba 与 nnU-Net 探索记录（2026-09）

## 公平比较协议

- 数据：`dataset/dataset/split_dataorigin`，共 2459 对 768×768 FA 图像与 ITK-SNAP 掩码。
- 任务：两通道 multilabel；`lesion_1 = label 1 or 3`，`lesion_2 = label 2 or 3`。
- 验证：验证折剔除文件名含 `_aug` 的离线增强副本；f1 因此为 444 张，避免历史上已确认的重复病例偏差。
- 主指标：全验证折累计像素的 per-lesion threshold sweep macro Dice；所有对比使用同一阈值网格。
- 正式基线：DINOv3 ConvNeXt-Tiny clean 五折 `0.7849±0.0219`，f1 `0.7769`；当前强方案 VS-UBR f1 `0.7902`，五折 independent sweep `0.7881±0.0194`。
- 硬件：RTX 2080 Ti 22,528 MiB，PyTorch 2.5.1+cu124。

## AAAG-UNet（Medical Physics 2026）

原文：Tan et al., “An improved nnUNet with Axial Attention Mechanism and Attention Gated Units for Efficient and Accurate Kidney Segmentation in CT Images,” DOI `10.1002/mp.70583`（[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13425451/)）。

论文的两个核心模块是：

1. AAM：在 nnU-Net 编码器下采样层后分别沿空间轴计算注意力，以低于全局二维/三维注意力的代价建立长程依赖。
2. AGU：以 decoder feature 为 gate，对对应 encoder skip 做 additive spatial attention；原文还描述了 channel calibration，用于抑制背景和缩小 encoder/decoder semantic gap。

本项目采用忠实的 2D 适配：高度/宽度轴向注意力、残差融合，以及带 spatial/channel calibration 的 skip attention gate。为适配 768×768 和 22GB 显存，高分辨率 AAM 先自适应池化到最长轴 64，深层特征保持原分辨率。实现位于 `src/bs/aaag_unet.py`。

风险：原文明确指出 AAM/AGU 偏向显著全局结构与清晰轮廓，可能损伤低对比度细粒度区域；FA 渗漏边界正是低对比度，因此必须与相同 nnU-Net baseline 做受控实验，不能假定论文增益可直接迁移。

## HMVF：MambaVision 启发的混合 Mamba 特征融合

主要依据：

- Hatamizadeh & Kautz, “MambaVision: A Hybrid Mamba-Transformer Vision Backbone,” CVPR 2025（[CVF 页面](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html)）。论文的关键结论是视觉任务不宜依赖纯 SSM，混合局部/全局 mixer 与后层 attention 更有效。
- 2025 年 Mamba 3D 医学分割综合研究指出 depthwise convolution 能补足局部建模，多尺度 Mamba 有益，而复杂扫描并非总是必要（arXiv:2503.19308）。
- UD-Mamba 指出固定空间扫描对模糊边界局部特征不友好（arXiv:2502.02024）。

仓库既有 S3RD 直接用 SSM 传播最终 logits，clean f1 为 `0.7807`，主要瓶颈是小数据和稀有 lesion_2。新方案 HMVF 不重复该结构：

1. 保留预训练 ConvNeXt + FPN 主路；
2. 在 fused decoder feature 上并联 depthwise local branch 与低分辨率四向 selective-SSM global branch；
3. 用学习门控融合两支；
4. output projection 零初始化，使初始网络严格等价 baseline；
5. SSM recurrence 强制 fp32 并用 `tanh` 约束残差，适配 Turing GPU AMP。

正式 f1 30 epoch 的最佳结果为 epoch 21 sweep macro `0.780249`，Dice-1 `0.797852`、Dice-2 `0.762647`，相对 clean f1 baseline `+0.34pp`，与旧 S3RD 基本持平，低于 VS-UBR f1 `0.7902`。峰值显存 `12.21GB`。该结果是轻微正向，需要跨折验证确认是否超出训练噪声。

## U-PEN Mamba：Progressive Expansion（PEL-3）迁移消融

arXiv:2609.24049v1《U-PEN Mamba》的可迁移核心之一，是通过 progressive expansion 显式构造高阶特征交互。考虑到当前模型已有强预训练 ConvNeXt 和 FPN，本轮没有直接替换整套编码器/解码器，而是做了一个受控的 `PE-lite` 适配：先对 fused FPN feature 做 GroupNorm 和 1×1 bottleneck，再用 `tanh` 限幅，拼接一阶、二阶和三阶项，经过 1×1 mixing 与 depthwise 3×3 卷积后以 zero-init residual 回注。这样初始化时严格等价于 baseline，可将性能差异更集中地归因到 progressive expansion。

同时测试了 `PE-lite + HMVF`，用于判断显式高阶局部交互与已有 Mamba 全局融合是否互补。四组统一使用 clean f1（train 1965 / val 444）、768×768、batch size 8、seed 42、5 epoch cosine 和同一 Asymmetric Focal-Tversky-BCE；因此这是短程方向筛选，不能与 30 epoch 主表直接等价比较。

| 5 epoch variant | Best epoch | Dice-1 | Dice-2 | Macro | Peak VRAM | 附加参数 |
|---|---:|---:|---:|---:|---:|---:|
| matched baseline | 4 | 0.76103 | 0.76614 | **0.76358** | 10.54GB | 0 |
| PE-lite | 5 | 0.76139 | 0.72581 | 0.74360 | 13.34GB | 178,081 |
| HMVF | 5 | 0.75991 | 0.73774 | 0.74882 | 12.21GB | 117,553 |
| PE-lite + HMVF | 4 | 0.75246 | 0.74462 | 0.74854 | 15.01GB | 295,634 |

结论是明确的负筛选：`PE-lite` 相对 matched baseline 下降约 `2.00pp`，损失主要来自稀有的 lesion_2；组合相对 baseline 下降约 `1.50pp`。与 HMVF 单独使用相比，组合的 lesion_2 增加约 `0.69pp`，但 lesion_1 下降约 `0.74pp`，macro 反而下降约 `0.03pp`，同时显存和参数进一步增加，没有观察到互补增益。

因此不把 progressive expansion 并入默认配置，也不扩展到 30 epoch 或五折。实现与 run 保留为可复现的负消融。这个结论只针对 U-PEN 的 progressive expansion 在当前 FA 小样本分割中的迁移，不否定 Mamba 特征融合本身：HMVF 的独立 30 epoch 结果仍为轻微正向。

## nnU-Net baseline 与 U-KAN 改造

nnU-Net baseline 使用 `nnunetv2==2.6.2` 所依赖的 official `dynamic-network-architectures` 2D `PlainConvUNet`：六层 `[32,64,128,256,320,320]`、InstanceNorm、LeakyReLU、每层两次卷积。数据读取、损失和评估仍复用项目当前 multilabel clean 协议，以保证与 DINOv3 实验可比。

正式 clean f1 30 epoch 结果：best epoch 28，fixed macro `0.729850`，shared/independent sweep macro `0.757127`，Dice-1 `0.768519`、Dice-2 `0.745735`，最优阈值均为 `0.9`，峰值显存 `6.84GB`。相对 ConvNeXt clean f1 baseline `0.7769` 低 `1.98pp`。这说明 nnU-Net 在本数据上可以稳定收敛并学到稀有类，但从头训练 encoder 仍明显弱于 DINOv3 预训练 ConvNeXt；因此后续 nnU-Net 改造必须与这个同协议结果比较，不能用旧的非 clean 数值。

AAAI 2025 U-KAN（[AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/32491)）在 tokenized intermediate representation 中加入 KAN layer，以可学习非线性基函数替代纯线性 token mapping。本项目实现轻量 spline-KAN bottleneck：逐通道三角样条响应 + 1×1 channel mixing + zero-init residual。它与 AAAG-UNet 共用完全相同的 nnU-Net baseline、训练数据和评估协议，便于将增益归因到模块本身。

另一篇 AAAI 2025 ConDSeg 强调医学图像的低对比度、soft boundary、foreground/background/uncertainty 解耦和 size-aware decoder。这与本 FA 数据特性高度相关，但仓库已有大量边界、不确定性、血管抑制和多尺度 decoder 实验；为避免方案雷同，本轮不直接复刻 ConDSeg，而把它作为解释 AAAG 风险和后续候选筛选的依据。

## MCADS depth-to-space restoration（CVPR 2025）

Wazir et al., “Rethinking Decoder Design: Improving Biomarker Segmentation Using Depth-to-Space Restoration and Residual Linear Attention”（[CVF 页面](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html)；[作者代码](https://github.com/saadwazir/MCADS-Decoder)）针对小样本 biomarker segmentation，指出预训练 encoder 的多尺度信息在低效 decoder 中容易损失。作者实现的前两级恢复使用 `depth_to_space`，融合后再加入 residual attention。

本项目的受控适配保留 DINOv3 ConvNeXt-FPN encoder，将原来从 1/4 logits 一次性双线性插值到 768×768 的路径替换为两个可学习的 sub-pixel restoration stage（96→48→24 通道），并在低分辨率 neck 上加入线性复杂度 residual attention。高分辨率通道数受到约束，以适配 22GB RTX 2080 Ti；训练目标使用已验证的 VS-UBR 设置。它与已有 EMCAD 的多尺度 depthwise attention、显式边界头和 PDE 扩散头均不同。正式训练前先进行 batch-8 显存 smoke，只有 smoke 通过才运行 f1 30 epoch。

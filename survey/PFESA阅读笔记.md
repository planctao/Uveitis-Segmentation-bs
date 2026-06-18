# PFESA 代码阅读笔记

**论文**：PFESA: FFT-Based Parameter-Free Edge and Structure Attention for Medical Image Segmentation  
**会议**：MICCAI 2025  
**代码**：https://github.com/59-lmq/PFESA  
**作者**：Mingqian Li et al., South China Normal University

---

## 1. 核心思想

Skip connection 中传递的特征包含冗余噪声、且边缘信息容易丢失。PFESA 提出一种 **零参数** 的频域注意力机制：

1. 用 FFT 将特征转到频域
2. 用高斯低通滤波器将频域分为 **低频**（全局结构）和 **高频**（边缘细节）
3. 对高频分量施加 **Edge Attention (EA)** — 基于 SNR 增强梯度敏感区域
4. 对低频分量施加 **Structure Attention (SA)** — 通过能量重分布抑制噪声
5. 两路注意力相加 → sigmoid → 与原始特征相乘

**核心卖点**：完全无可训练参数 → 不增加过拟合风险，特别适合小数据集。

---

## 2. 代码实现分析（核心 137 行）

```python
class PFESA(nn.Module):
    def __init__(self, base_ratio=0.1):
        self.activation = nn.Sigmoid()
        self.base_ratio = base_ratio  # 高斯滤波器截止频率
        self.eps = 1e-5

    def forward(self, x):
        # 1. FFT（对 H,W 维度）
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))  # 中心化

        # 2. 高斯低通掩码分离高/低频
        low_freq_mask = self._create_low_freq_mask(h, w)  # 高斯核
        high_freq_mask = 1 - low_freq_mask

        low_freq = abs(iFFT(x_freq * low_freq_mask))
        high_freq = abs(iFFT(x_freq * high_freq_mask))

        # 3. 双路注意力
        low_att = structure_attention(low_freq)   # sigmoid(标准化能量)
        high_att = edge_attention(high_freq)      # SNR = (x-μ)²/σ²

        # 4. 融合
        out_att = sigmoid(low_att + high_att)
        return out_att * x  # 注意力加权原始特征
```

### 2.1 Edge Attention (EA)

```python
def _edge_attention(self, x):
    # 逐通道 SNR 计算：(x - μ)² / σ²
    x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
    x_var = x.var(dim=[2,3], keepdim=True)
    return x_minus_mu_square / (x_var + eps)
```

**直觉**：高频分量中，偏离均值越大的像素越可能是边缘 → 给予更高的注意力权重。

### 2.2 Structure Attention (SA)

```python
def _structure_attention(self, x):
    energy = x.pow(2)
    energy_mu = energy.mean(dim=[2,3], keepdim=True)
    energy_var = energy.var(dim=[2,3], keepdim=True)
    y = (energy - energy_mu) / (energy_var + eps)
    return sigmoid(y)
```

**直觉**：低频分量中，能量高于平均的像素保留（结构性区域），能量低的被抑制（噪声）。

### 2.3 频率分离（高斯低通滤波）

```python
def _create_low_freq_mask(self, h, w):
    mask_ratio = self.base_ratio * min(h, w) / max(h, w)
    Y, X = meshgrid(linspace(-1,1,h), linspace(-1,1,w))
    mask = exp(-(Y² + X²) / (2 * mask_ratio²))
    return mask  # 中心=1（低频），边缘→0（高频）
```

`base_ratio=0.1` 表示高斯核的标准差，值越小 → 低通带越窄 → 更多频率被归为"高频"。

---

## 3. 在网络中的位置

PFESA 作为 **skip connection 上的注意力模块**：

```
Encoder → feature → PFESA(feature) → 传给 Decoder 的 skip connection
                                    ↘ MaxPool → 下一层 Encoder
```

代码中支持 U-Net、TransUNet、ResUNet++、V-Net 等多种骨干网络。

---

## 4. 实验结果（论文摘要）

- DSC 提升 **+3.3%** vs baseline（多数据集平均）
- 在 ISIC-2017、GlaS 等 2D 数据集 + LA 等 3D 数据集上均有效
- 对比了 SE、CBAM、ECA、SimAM、SIAM 等有参数注意力模块，PFESA 效果最好且零参数

---

## 5. 与我们项目（DINOv3 + WBE）的对比与融合分析

### 5.1 相似点

| 维度 | PFESA | 我们的 WBE |
|------|-------|-----------|
| 核心思想 | 频域分离高/低频 → 增强边缘 | 小波分解高/低频 → 增强边界 |
| 频域工具 | FFT + 高斯滤波 | Haar DWT（离散小波变换） |
| 目标 | 增强 skip connection 中的边缘信息 | 增强 ViT 中间特征的边界信息 |
| 参数量 | 0（完全无参数） | 6.44M（含 bottleneck） |
| 适用架构 | U-Net 系列的 skip connection | ViT + FPN decoder 的中间特征 |

### 5.2 差异点

| 维度 | PFESA | 我们的 WBE |
|------|-------|-----------|
| 位置 | Skip connection（encoder → decoder） | Backbone 输出后、Decoder 输入前 |
| 频域方法 | 全局 FFT + 高斯软分割 | 局部 Haar DWT（2x2 窗口） |
| 学习能力 | 无（固定规则） | 有（可学习通道注意力+融合卷积） |
| 计算量 | FFT O(N log N) | DWT O(N)（更高效） |
| 自适应性 | 固定 base_ratio | 学习哪些高频通道更重要 |

### 5.3 PFESA 对我们的启发

**可以直接借鉴的点：**

1. **FFT 替代/补充 DWT**：PFESA 证明了 FFT 频域分割在医学图像中的有效性。我们可以在 WBE 中新增一个 FFT 分支与 DWT 并行（dual-path 频域增强）。

2. **无参数 SNR 注意力作为补充**：PFESA 的 Edge Attention 本质是逐像素 SNR，完全无参数。我们可以在 WBE 的高频分支中，用这种 SNR 作为"初始注意力"再叠加可学习的 SE 权重。

3. **Structure Attention 思路**：对低频分量做能量标准化再 sigmoid，可以抑制低频中的噪声。当前 WBE 只是简单地将 ll + hf_enhanced，没有对低频做类似处理。

**可以改进/超越的点：**

1. **PFESA 无法自适应**：固定 base_ratio=0.1，对所有通道/所有图像用同一截止频率。我们的 WBE 通过可学习的通道注意力来自适应选择哪些高频重要 → 这是我们的优势。

2. **PFESA 设计给 U-Net skip connection**：而我们的架构是 ViT + FPN，没有传统意义的 skip connection。直接搬过来不合适，需要适配。

3. **FFT 的全局性 vs DWT 的局部性**：FFT 是全局变换，DWT 是局部变换。对于 48×48 的 ViT token map，两者差异不大，但 DWT 更适合捕捉局部边界。

---

## 6. 是否适合用到我们的论文？

### 6.1 直接搬用 PFESA → ❌ 不太合适

原因：
- PFESA 是为 U-Net skip connection 设计的，我们没有传统 skip connection
- 它是零参数模块，在小数据集上过拟合风险已经很低，但也意味着表达力有限
- 直接放到 ViT 的 768 维特征上，FFT 的计算量不小（768 通道 × FFT）
- 和我们已有的 WBE（DWT 路线）存在方法论冲突，论文叙事会混乱

### 6.2 借鉴其中部分思想 → ✅ 可以

**推荐融合方案：FFT-Enhanced WBE**

在现有 WBE 的基础上，引入 PFESA 的 SNR 驱动的无参数注意力作为高频分支的初始化权重：

```python
class WBE_v2(nn.Module):
    def forward(self, x):
        ll, lh, hl, hh = haar_dwt(x)
        hf = lh + hl + hh

        # PFESA 启发：用 SNR 作为无参数的初始边缘检测
        snr_weight = (hf - hf.mean(dim=[2,3], keepdim=True)).pow(2) / \
                     (hf.var(dim=[2,3], keepdim=True) + 1e-5)
        snr_weight = torch.sigmoid(snr_weight)

        # 叠加可学习的通道注意力
        hf_enhanced = self.hf_conv(hf) * self.hf_attn(hf) * snr_weight

        combined = ll + hf_enhanced
        ...
```

这样可以在论文中写：
> "受 PFESA (MICCAI 2025) 启发，我们在小波高频分支中引入基于 SNR 的无参数边缘先验，再与可学习的通道注意力协同工作..."

### 6.3 论文叙事建议

- **不要直接用 PFESA**（它是另一种独立的注意力方法，和我们不是一个体系）
- **可以 cite 它作为 related work**，说明频域增强在医学分割中的趋势
- **借鉴 SNR 思想**作为 WBE 的一个消融变体（WBE + SNR prior）
- **强调我们的区别**：DWT（局部）vs FFT（全局）、可学习 vs 无参数、ViT backbone vs U-Net

---

## 7. 总结

| 评估维度 | 结论 |
|----------|------|
| 论文质量 | MICCAI 2025，代码清晰，实验充分 |
| 核心创新 | 零参数频域注意力（FFT + SNR 双路） |
| 直接搬用 | ❌ 架构不匹配（U-Net skip vs ViT） |
| 部分借鉴 | ✅ SNR 边缘先验可融入 WBE 高频分支 |
| 论文叙事 | 可作为 related work 引用，强调与我们方法的互补性 |
| 实现难度 | ⭐ 极低（核心只有 ~20 行无参数代码） |

---

## 参考

```bibtex
@inproceedings{li2025pfesa,
  title={PFESA: FFT-Based Parameter-Free Edge and Structure Attention for Medical Image Segmentation},
  author={Li, Mingqian and Yan, Zhiqian and Yan, Miaoning and Liang, Yaodong and Zhang, Qingmao and Ma, Qiongxiong},
  booktitle={MICCAI},
  year={2025}
}
```

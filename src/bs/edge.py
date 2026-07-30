"""Edge-guided decoding for FA leakage segmentation —— 边缘检测驱动的分割增强。

三条边缘检测文献路线在本文件落地（均为低参数、强先验，规避 WBE(6.44M) 那类
重模块在 ~2k 小数据上的过拟合）：

  1) Pixel Difference Convolution (PiDiNet, ICCV 2021)
     把 Sobel/LBP 式差分算子融进普通卷积：中心差分卷积 (CPDC) 输出等于
     ``conv(x) - x_center * Σ(W)``，参数量与普通 3x3 卷积完全相同，但对常数区域
     响应为 0、对局部梯度/边缘天然敏感。

  2) Edge-guided decoding (CTO IPMI 2023 / ET-Net MICCAI 2019)
     由 PDC 边缘分支产生边缘特征，门控增强主分割特征，并输出边缘图接受辅助监督；
     推理期仅返回分割 logits，零额外开销。

  3) 软边缘目标
     FA 渗漏边界弥散、标注主观，硬边缘 GT 噪声大；用热核(高斯)扩散把边界带软化，
     直面"荧光浓度渐变"的物理本质（与 DSB 软边界监督同源）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CentralDifferenceConv2d(nn.Module):
    """中心像素差分卷积 (CPDC)，PiDiNet (ICCV 2021)。

    ``y = Σ_i w_i · (x_i - x_c) = conv(x, W) - x_c · Σ_i w_i``

    等价于"普通卷积减去中心像素×核权重和"，因此参数量与普通卷积一致，但对常数
    区域响应恒为 0、只对局部梯度/边缘响应，是一个可学习的边缘算子。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        out_normal = self.conv(x)
        weight_sum = self.conv.weight.sum(dim=(2, 3))  # [out, in]
        out_center = F.conv2d(x, weight_sum[:, :, None, None])
        return out_normal - out_center


# 3x3 环 (顺时针, 从左上开始) 的 (row, col)
_RING3 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]


class AngularDifferenceConv2d(nn.Module):
    """角向像素差分卷积 (APDC)，PiDiNet (ICCV 2021)。

    对 3x3 环上相邻邻居作差：``y = Σ_i w_i (x_i - x_{next(i)})``，等价于用
    ``W - rotate_ring(W)`` 的等效权重做普通卷积 (中心置 0)。对**有取向的边缘/
    条状结构**(如血管、渗漏弧形边界)比各向同性 CPDC 更敏感，参数量与普通卷积一致。
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def _effective_weight(self) -> Tensor:
        weight = self.conv.weight
        effective = weight.clone()
        for k, (r, c) in enumerate(_RING3):
            pr, pc = _RING3[(k - 1) % len(_RING3)]  # 逆时针前一个
            effective[:, :, r, c] = weight[:, :, r, c] - weight[:, :, pr, pc]
        effective[:, :, 1, 1] = 0.0
        return effective

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self._effective_weight(), bias=self.conv.bias, padding=1)


# 8 个方向: (3x3环 r,c) -> (5x5 外环 r,c) / (5x5 内环 r,c)
_RADIAL_MAP = [
    ((0, 0), (0, 0), (1, 1)),
    ((0, 1), (0, 2), (1, 2)),
    ((0, 2), (0, 4), (1, 3)),
    ((1, 2), (2, 4), (2, 3)),
    ((2, 2), (4, 4), (3, 3)),
    ((2, 1), (4, 2), (3, 2)),
    ((2, 0), (4, 0), (3, 1)),
    ((1, 0), (2, 0), (2, 1)),
]


class RadialDifferenceConv2d(nn.Module):
    """径向像素差分卷积 (RPDC)，PiDiNet (ICCV 2021)。

    对 8 个方向作外环(半径2)减内环(半径1)差分：等效 5x5 卷积核在外环放 +w、
    内环放 -w、中心 0。捕捉**沿半径方向的强度跃变**(渗漏由中心亮向外衰减的径向边界)，
    可学习参数仅 8 个环权重/通道对，与 3x3 普通卷积同量级。
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _effective_weight(self) -> Tensor:
        eff = self.weight.new_zeros(self.weight.shape[0], self.weight.shape[1], 5, 5)
        for (r3, c3), (ro, co), (ri, ci) in _RADIAL_MAP:
            w_dir = self.weight[:, :, r3, c3]
            eff[:, :, ro, co] = eff[:, :, ro, co] + w_dir
            eff[:, :, ri, ci] = eff[:, :, ri, ci] - w_dir
        return eff

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self._effective_weight(), bias=self.bias, padding=2)


def make_pdc_conv(pdc_type: str, in_channels: int, out_channels: int, bias: bool = False) -> nn.Module:
    name = str(pdc_type).lower()
    if name in {"cpdc", "central"}:
        return CentralDifferenceConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
    if name in {"apdc", "angular"}:
        return AngularDifferenceConv2d(in_channels, out_channels, bias=bias)
    if name in {"rpdc", "radial"}:
        return RadialDifferenceConv2d(in_channels, out_channels, bias=bias)
    if name in {"vanilla", "conv"}:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
    raise ValueError(f"Unsupported PDC type: {pdc_type}")


class PDCBlock(nn.Module):
    """两层 PDC + GroupNorm + GELU 的边缘特征提取块 (pdc_type 可选 cpdc/apdc/rpdc)。"""

    def __init__(self, in_channels: int, out_channels: int, pdc_type: str = "cpdc") -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.body = nn.Sequential(
            make_pdc_conv(pdc_type, in_channels, out_channels, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            make_pdc_conv(pdc_type, out_channels, out_channels, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class PDCBank(nn.Module):
    """多方向 PDC 并行组：每种 pdc_type 一条 PDCBlock 分支，拼接后 1x1 融合。

    把各向同性(CPDC)、角向(APDC)、径向(RPDC)差分算子并联，让边缘分支同时感知
    孤立梯度、条状取向边缘与径向浓度跃变，覆盖 FA 渗漏/血管边界的多种几何。
    """

    def __init__(self, in_channels: int, out_channels: int, pdc_types: list[str]) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.branches = nn.ModuleList(
            [PDCBlock(in_channels, out_channels, pdc_type=t) for t in pdc_types]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(pdc_types), out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class EdgeGuidedHead(nn.Module):
    """PDC 边缘分支 + 边缘门控增强 + 主分割头。

    - 边缘分支：``PDCBlock`` 提取边缘敏感特征，``edge_out`` 输出每病灶一张边缘图。
    - 边缘门控：由边缘特征生成 (-1,1) 的调制量，``refined = feat * (1 + gate)``；
      gate 零初始化 → 初始等价于普通卷积头（"不劣于 conv"的安全下界）。
    - 始终返回 ``(seg_logits, edge_logits)``；是否暴露 edge 由上层解码器按训练/推理决定，
      推理期上层只取 ``seg_logits``，边缘分支零额外推理开销。
    """

    def __init__(self, in_channels: int, out_channels: int = 2, edge_channels: int = 0, edge_pdc_types: list[str] | None = None) -> None:
        super().__init__()
        edge_channels = int(edge_channels) if int(edge_channels) > 0 else max(32, in_channels // 2)
        groups = _group_count(in_channels)
        pdc_types = [str(t).lower() for t in (edge_pdc_types or ["cpdc"])] or ["cpdc"]
        self.edge_branch = (
            PDCBlock(in_channels, edge_channels, pdc_type=pdc_types[0])
            if len(pdc_types) == 1
            else PDCBank(in_channels, edge_channels, pdc_types)
        )
        self.edge_out = nn.Conv2d(edge_channels, out_channels, kernel_size=1)
        self.edge_gate = nn.Conv2d(edge_channels, in_channels, kernel_size=1)
        self.seg_fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
        )
        self.seg_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # gate 零初始化: tanh(0)=0 -> refined == feat, 初始退化为普通卷积头
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        edge_feat = self.edge_branch(feat)
        edge_logits = self.edge_out(edge_feat)
        gate = torch.tanh(self.edge_gate(edge_feat))
        refined = feat * (1.0 + gate)
        seg_logits = self.seg_out(self.seg_fuse(refined))
        return seg_logits, edge_logits


_GAC_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_GAC_IMAGENET_STD = (0.229, 0.224, 0.225)


def _grad_lap(field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """中心差分梯度 (ux, uy) 与 4 邻域拉普拉斯 (replicate 边界)。"""
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    north = padded[:, :, 0:-2, 1:-1]
    south = padded[:, :, 2:, 1:-1]
    east = padded[:, :, 1:-1, 2:]
    west = padded[:, :, 1:-1, 0:-2]
    grad_x = 0.5 * (east - west)
    grad_y = 0.5 * (south - north)
    laplacian = north + south + east + west - 4.0 * field
    return grad_x, grad_y, laplacian


class GeodesicActiveContourHead(nn.Module):
    """GAC: 边缘指示函数驱动的可微测地主动轮廓边界演化头。

    经典边缘型分割 PDE：轮廓在边缘停止函数 ``g(|∇I|)`` 引导下演化，被图像高梯度
    ("边缘")吸附。本头把分割概率 u 作为水平集代理，做 K 步演化::

        g = 1 / (1 + |∇I|² / κ²)                       # 图像边缘指示 (边缘处→0)
        u ← clamp(u + dt·[ g·Δu + β·⟨∇g, ∇u⟩ ], 0, 1)  # 边缘停止曲率流 + 边缘平流

    ``g·Δu`` 在同质区平滑、在边缘停止；``⟨∇g,∇u⟩`` 把边界吸附到图像边缘。另设一条
    PDC 学习边缘分支输出 ``edge_logits`` 接受边界带监督 (边缘检测)。``dt→0`` 时
    退化为普通 1x1 头 (不劣于 conv 的安全下界)，仅新增极少标量参数。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        edge_channels: int = 0,
        iters: int = 8,
        dt: float = 0.1,
        beta: float = 0.5,
        kappa: float = 0.1,
        edge_pdc_types: list[str] | None = None,
        guide_input: str = "normalized",
    ) -> None:
        super().__init__()
        edge_channels = int(edge_channels) if int(edge_channels) > 0 else max(32, in_channels // 2)
        groups = _group_count(in_channels)
        self.out_channels = int(out_channels)
        self.iters = int(iters)
        self.guide_input = str(guide_input).lower()
        pdc_types = [str(t).lower() for t in (edge_pdc_types or ["cpdc"])] or ["cpdc"]
        self.edge_branch = (
            PDCBlock(in_channels, edge_channels, pdc_type=pdc_types[0])
            if len(pdc_types) == 1
            else PDCBank(in_channels, edge_channels, pdc_types)
        )
        self.edge_out = nn.Conv2d(edge_channels, out_channels, kernel_size=1)
        self.seg_fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
        )
        self.seg_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dt = nn.Parameter(torch.full((out_channels,), float(dt)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(float(max(kappa, 1e-3)))))

    def _guide_intensity(self, guide: Tensor) -> Tensor:
        if self.guide_input == "imagenet":
            mean = guide.new_tensor(_GAC_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = guide.new_tensor(_GAC_IMAGENET_STD).view(1, 3, 1, 1)
            guide = (guide * std + mean).clamp(0.0, 1.0)
        return guide.max(dim=1, keepdim=True).values

    def forward(self, feat: Tensor, guide: Tensor | None = None) -> tuple[Tensor, Tensor]:
        seg_logits = self.seg_out(self.seg_fuse(feat))
        edge_logits = self.edge_out(self.edge_branch(feat))
        if guide is None or self.iters <= 0:
            return seg_logits, edge_logits
        device_type = guide.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            intensity = self._guide_intensity(guide).float()
            grad_x, grad_y, _ = _grad_lap(intensity)
            grad_sq = grad_x * grad_x + grad_y * grad_y
            kappa = torch.exp(self.log_kappa).clamp(1e-3, 10.0)
            g = 1.0 / (1.0 + grad_sq / (kappa * kappa + 1e-6))  # 边缘处 -> 0
            g_x, g_y, _ = _grad_lap(g)
            dt = 0.24 * torch.sigmoid(self.dt).view(1, -1, 1, 1)  # 约束步长稳定
            beta = self.beta
            u = torch.sigmoid(seg_logits.float())
            for _ in range(self.iters):
                u_x, u_y, u_lap = _grad_lap(u)
                advection = g_x * u_x + g_y * u_y
                u = (u + dt * (g * u_lap + beta * advection)).clamp(0.0, 1.0)
            refined_logits = torch.logit(u.clamp(1e-4, 1.0 - 1e-4))
        return refined_logits.to(seg_logits.dtype), edge_logits


def boundary_band(values: Tensor, valid: Tensor, kernel_size: int) -> Tensor:
    """形态学梯度 (膨胀 - 腐蚀) 提取边界带，输出 [0,1]。"""
    if kernel_size <= 1:
        return (values * valid).clamp(0.0, 1.0)
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    values = values * valid
    dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=kernel // 2)
    eroded = 1.0 - F.max_pool2d(1.0 - values, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (dilated - eroded).clamp(0.0, 1.0) * valid


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = x.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius), groups=channels)
    x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=channels)
    return x


def lesion_edge_target(
    target: Tensor,
    valid: Tensor,
    band: int = 5,
    soft: bool = False,
    sigma: float = 1.5,
) -> Tensor:
    """由 2 通道病灶目标生成边界带监督目标 [B, C, H, W] ∈ [0,1]。

    - ``soft=False``：硬边界带 (膨胀-腐蚀)。
    - ``soft=True`` ：对硬边界带做高斯扩散，得到更宽、平滑的软边界带，贴合 FA
      渗漏"边界弥散、浓度渐变"的物理本质。
    """
    hard = boundary_band(target.float(), valid.float(), band)
    if not soft:
        return hard.clamp(0.0, 1.0)
    soft_edge = _gaussian_blur(hard, sigma)
    return soft_edge.clamp(0.0, 1.0)

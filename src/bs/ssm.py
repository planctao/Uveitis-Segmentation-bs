"""纯 PyTorch 选择性状态空间模型 (Selective SSM / Mamba 核心)，无需编译 mamba-ssm。

用于 S3RD (Selective State-Space Reaction-Diffusion)：以数据驱动的选择性状态空间
递推替代 RDH 中固定的 Perona-Malik 扩散，建模 FA 渗漏浓度沿空间的各向异性传播。

- ``selective_scan_1d`` : 沿长度 L 的选择性扫描 (Mamba 离散化递推, for 循环参考实现)。
- ``SelectiveSSM2D``    : VMamba 风格 4 方向 (沿 W/H 正反) 2D 扫描；序列长=H 或 W，
                         其余维并行；内部可降分辨率控制显存；out_proj 零初始化以保证
                         接入 RDH 时初始退化为 seed-only。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def selective_scan_1d(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """选择性扫描。

    u, delta : [N, d, L]      (输入与步长, delta>0)
    A        : [d, n]         (状态转移, 应为负值保证稳定)
    B, C     : [N, n, L]      (输入依赖的选择性投影)
    D        : [d]            (跳连)
    返回 y    : [N, d, L]
    """
    N, d, L = u.shape
    delta_u = delta * u
    delta_A = torch.exp(torch.einsum("ndl,dm->ndlm", delta, A))  # [N,d,L,n] in (0,1)
    delta_B_u = torch.einsum("ndl,nml->ndlm", delta_u, B)  # [N,d,L,n]
    h = u.new_zeros(N, d, A.shape[1])
    outputs = []
    for t in range(L):
        h = delta_A[:, :, t] * h + delta_B_u[:, :, t]  # [N,d,n]
        outputs.append(torch.einsum("ndm,nm->nd", h, C[:, :, t]))  # [N,d]
    y = torch.stack(outputs, dim=-1)  # [N,d,L]
    return y + u * D.view(1, d, 1)


class SelectiveSSM2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        d_inner: int = 64,
        d_state: int = 16,
        directions: int = 4,
        guide_channels: int = 1,
        ssm_stride: int = 4,
    ) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.directions = int(directions)
        self.guide_channels = int(guide_channels)
        self.ssm_stride = max(1, int(ssm_stride))

        self.in_proj = nn.Conv2d(in_channels + guide_channels, self.d_inner, kernel_size=1)
        self.dt_proj = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1)
        self.bc_proj = nn.Conv2d(self.d_inner, 2 * self.d_state, kernel_size=1)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log) 恒负
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Conv2d(self.d_inner, out_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)  # 零初始化 -> 初始退化为 seed
        nn.init.zeros_(self.out_proj.bias)

    def _dir_list(self) -> list[str]:
        if self.directions <= 1:
            return ["w"]
        if self.directions == 2:
            return ["w", "h"]
        return ["w", "w_rev", "h", "h_rev"]

    def _scan_dir(self, x: Tensor, delta: Tensor, A: Tensor, Bm: Tensor, Cm: Tensor, direction: str) -> Tensor:
        batch, d, height, width = x.shape
        along_w = direction in ("w", "w_rev")
        flip = direction.endswith("_rev")
        axis = -1 if along_w else -2
        xt, dt, bm, cm = (t.flip(axis) if flip else t for t in (x, delta, Bm, Cm))

        if along_w:
            length, parallel = width, batch * height
            reshape = lambda t, c: t.permute(0, 2, 1, 3).reshape(parallel, c, length)  # [B*H, c, W]
            restore = lambda t: t.reshape(batch, height, d, width).permute(0, 2, 1, 3)
        else:
            length, parallel = height, batch * width
            reshape = lambda t, c: t.permute(0, 3, 1, 2).reshape(parallel, c, length)  # [B*W, c, H]
            restore = lambda t: t.reshape(batch, width, d, height).permute(0, 2, 3, 1)

        y = selective_scan_1d(reshape(xt, d), reshape(dt, d), A, reshape(bm, self.d_state), reshape(cm, self.d_state), self.D)
        y = restore(y)
        return y.flip(axis) if flip else y

    def forward(self, feat: Tensor, guide: Tensor | None = None, return_aux: bool = False):
        full_size = feat.shape[-2:]
        x_in = feat if (guide is None or self.guide_channels == 0) else torch.cat([feat, guide], dim=1)
        if self.ssm_stride > 1:
            x_in = F.avg_pool2d(x_in, kernel_size=self.ssm_stride, ceil_mode=True)

        x = F.silu(self.in_proj(x_in))
        delta = F.softplus(self.dt_proj(x))  # [B,d_inner,h,w] > 0
        bc = self.bc_proj(x)
        Bm, Cm = bc[:, : self.d_state], bc[:, self.d_state :]
        A = -torch.exp(self.A_log)  # [d_inner,d_state] 负

        directions = self._dir_list()
        scanned = sum(self._scan_dir(x, delta, A, Bm, Cm, d) for d in directions) / len(directions)
        propagation = self.out_proj(scanned)
        if self.ssm_stride > 1:
            propagation = F.interpolate(propagation, size=full_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return propagation

        speed = delta.mean(dim=1, keepdim=True)  # 传播速率场 (选择性步长)
        speed = F.interpolate(speed, size=full_size, mode="bilinear", align_corners=False)
        return propagation, {"delta": speed}

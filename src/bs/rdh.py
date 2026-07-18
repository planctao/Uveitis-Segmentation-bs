"""RDH: Reaction-Diffusion Head —— 可解释的物理演化分割头。

把"荧光渗漏=扩散过程"直接建模进网络前向，作为 DiffLeak 的结构层核心创新：
  1) 源项/种子 s : 从解码特征预测渗漏"起漏点"                         (可解释)
  2) 传导系数 c : 由原图高荧光梯度(Perona-Malik) 与特征共同决定，
                  在血管/边界处 c->0 停止扩散                          (可解释)
  3) 反应-扩散演化 K 步 (可微):
       u_0 = s
       u_{t+1} = clamp(u_t + dt * [ div(c·∇u_t) + rho·s·u_t·(1-u_t) - lam·u_t ], 0, 1)
  4) 残差式输出 logit(u_K): 当 iters=0 或 dt->0 时退化为普通 1x1 seed 头，
     从而保证 RDH 不劣于原卷积头 (可优化下界)。

所有中间量 (seed / conductance / 每步 u_t) 均可导出可视化，属 interpretable-by-design，
参数量 <1k、受物理规律正则，适合小数据、抗过拟合。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.ssm import SelectiveSSM2D


def _neighbors(field: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """四邻域差分 (replicate 边界)，返回 (N, S, E, W) 相对中心的差。"""
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    north = padded[:, :, 0:-2, 1:-1] - field
    south = padded[:, :, 2:, 1:-1] - field
    east = padded[:, :, 1:-1, 2:] - field
    west = padded[:, :, 1:-1, 0:-2] - field
    return north, south, east, west


class ReactionDiffusionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        iters: int = 8,
        dt: float = 0.2,
        reaction: str = "fisher",
        use_image_conductance: bool = True,
        lambda_init: float = 0.1,
        rho_init: float = 1.0,
        kappa: float = 0.1,
        dynamics: str = "pde",
        d_state: int = 16,
        ssm_directions: int = 4,
        ssm_stride: int = 4,
        ssm_d_inner: int = 64,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.iters = int(iters)
        self.reaction = str(reaction)
        self.use_image_conductance = bool(use_image_conductance)
        self.dynamics = str(dynamics).lower()
        self.seed = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cond_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dt = nn.Parameter(torch.full((out_channels,), float(dt)))
        self.rho = nn.Parameter(torch.full((out_channels,), float(rho_init)))
        self.lam = nn.Parameter(torch.full((out_channels,), float(lambda_init)))
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(float(max(kappa, 1e-3)))))
        if self.dynamics == "ssm":
            self.ssm = SelectiveSSM2D(
                in_channels,
                out_channels=out_channels,
                d_inner=ssm_d_inner,
                d_state=d_state,
                directions=ssm_directions,
                guide_channels=1 if self.use_image_conductance else 0,
                ssm_stride=ssm_stride,
            )

    def _conductance(self, feat: Tensor, guide: Tensor | None) -> Tensor:
        conductance = torch.sigmoid(self.cond_conv(feat))  # 特征驱动的传导基 (0,1)
        if self.use_image_conductance and guide is not None:
            intensity = guide.max(dim=1, keepdim=True).values  # 高荧光通道
            north, south, east, west = _neighbors(intensity)
            grad_sq = north * north + south * south + east * east + west * west
            kappa = torch.exp(self.log_kappa).clamp(1e-3, 10.0)
            perona_malik = torch.exp(-grad_sq / (kappa * kappa + 1e-6))  # 边界处 ->0
            conductance = conductance * perona_malik
        return conductance.clamp(1e-4, 1.0)

    def _reaction(self, u: Tensor, seed: Tensor) -> Tensor:
        rho = self.rho.view(1, -1, 1, 1)
        lam = self.lam.view(1, -1, 1, 1)
        if self.reaction == "pull":
            return rho * (seed - u) - lam * u
        return rho * seed * u * (1.0 - u) - lam * u  # Fisher-KPP 生长

    def _evolve(self, feat: Tensor, guide: Tensor | None):
        seed_logits = self.seed(feat)
        seed = torch.sigmoid(seed_logits)
        conductance = self._conductance(feat, guide)
        dt = self.dt.view(1, -1, 1, 1)
        u = seed
        steps = [u]
        for _ in range(self.iters):
            north, south, east, west = _neighbors(u)
            divergence = conductance * (north + south + east + west)  # 各向异性拉普拉斯
            u = (u + dt * (divergence + self._reaction(u, seed))).clamp(0.0, 1.0)
            steps.append(u)
        return seed_logits, seed, conductance, u, steps

    def forward(self, feat: Tensor, guide: Tensor | None = None) -> Tensor:
        if self.dynamics == "ssm":
            seed_logits = self.seed(feat)
            guide_input = (
                guide.max(dim=1, keepdim=True).values
                if (self.use_image_conductance and guide is not None)
                else None
            )
            return seed_logits + self.ssm(feat, guide_input)
        _, _, _, u, _ = self._evolve(feat, guide)
        return torch.logit(u.clamp(1e-4, 1.0 - 1e-4))

    @torch.no_grad()
    def evolution(self, feat: Tensor, guide: Tensor | None = None) -> dict[str, Tensor]:
        """返回可视化用的中间物理量 (不参与训练)。"""
        if self.dynamics == "ssm":
            seed_logits = self.seed(feat)
            guide_input = (
                guide.max(dim=1, keepdim=True).values
                if (self.use_image_conductance and guide is not None)
                else None
            )
            propagation, aux = self.ssm(feat, guide_input, return_aux=True)
            seed = torch.sigmoid(seed_logits)
            final = torch.sigmoid(seed_logits + propagation)
            return {
                "seed": seed,
                "conductance": aux["delta"],
                "final": final,
                "steps": torch.stack([seed, final], dim=0),
            }
        seed_logits, seed, conductance, u, steps = self._evolve(feat, guide)
        return {
            "seed": seed,
            "conductance": conductance,
            "final": u,
            "steps": torch.stack(steps, dim=0),  # [iters+1, B, out, H, W]
        }

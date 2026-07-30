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

VA-RDH (diffusion_mode="anisotropic"): 把各向同性 Perona-Malik 标量传导升级为由图像结构张量
导出的相干增强 (Weickert) 各向异性扩散张量。渗漏沿血管相干方向强扩散、跨血管方向弱扩散，
对应 "FA 渗漏源自受损血管并沿血管树蔓延" 的临床先验；仅新增一个可学习相干对比标量。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.ssm import SelectiveSSM2D


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(max(float(value), 1e-6), dtype=torch.float64)
    return float(torch.log(torch.expm1(tensor)).item())


def _stable_dt_parameter(value: float, maximum: float = 0.24) -> float:
    ratio = min(max(float(value) / maximum, 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(ratio, dtype=torch.float64)).item())


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
        stable_constraints: bool = False,
        flux_scheme: str = "center",
        guide_input: str = "normalized",
        diffusion_mode: str = "isotropic",
        struct_pre_sigma: float = 1.0,
        struct_rho_sigma: float = 2.0,
        ced_alpha: float = 0.005,
        ced_contrast: float = 1.0,
        ced_direction: str = "along",
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.iters = int(iters)
        self.reaction = str(reaction)
        self.use_image_conductance = bool(use_image_conductance)
        self.dynamics = str(dynamics).lower()
        self.stable_constraints = bool(stable_constraints)
        self.flux_scheme = str(flux_scheme).lower()
        self.guide_input = str(guide_input).lower()
        self.diffusion_mode = str(diffusion_mode).lower()
        self.struct_pre_sigma = float(struct_pre_sigma)
        self.struct_rho_sigma = float(struct_rho_sigma)
        self.ced_alpha = float(ced_alpha)
        self.ced_direction = str(ced_direction).lower()
        if self.flux_scheme not in {"center", "edge"}:
            raise ValueError(f"Unsupported RDH flux scheme: {flux_scheme}")
        if self.guide_input not in {"normalized", "imagenet"}:
            raise ValueError(f"Unsupported RDH guide input: {guide_input}")
        if self.diffusion_mode not in {"isotropic", "anisotropic"}:
            raise ValueError(f"Unsupported RDH diffusion mode: {diffusion_mode}")
        if self.ced_direction not in {"along", "across"}:
            raise ValueError(f"Unsupported RDH ced direction: {ced_direction}")
        self.seed = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cond_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        dt_parameter = _stable_dt_parameter(dt) if self.stable_constraints else float(dt)
        rho_parameter = _inverse_softplus(rho_init) if self.stable_constraints else float(rho_init)
        lambda_parameter = _inverse_softplus(lambda_init) if self.stable_constraints else float(lambda_init)
        self.dt = nn.Parameter(torch.full((out_channels,), dt_parameter))
        self.rho = nn.Parameter(torch.full((out_channels,), rho_parameter))
        self.lam = nn.Parameter(torch.full((out_channels,), lambda_parameter))
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(float(max(kappa, 1e-3)))))
        if self.diffusion_mode == "anisotropic":
            self.log_ced_contrast = nn.Parameter(torch.log(torch.tensor(float(max(ced_contrast, 1e-3)))))
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

    def _guide_intensity(self, guide: Tensor) -> Tensor:
        if self.guide_input == "imagenet":
            mean = guide.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = guide.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            guide = (guide * std + mean).clamp(0.0, 1.0)
        return guide.max(dim=1, keepdim=True).values

    def _coefficients(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.stable_constraints:
            return 0.24 * torch.sigmoid(self.dt), F.softplus(self.rho), F.softplus(self.lam)
        return self.dt, self.rho, self.lam

    def _conductance(self, feat: Tensor, guide: Tensor | None) -> Tensor:
        conductance = torch.sigmoid(self.cond_conv(feat))  # 特征驱动的传导基 (0,1)
        if self.use_image_conductance and guide is not None:
            intensity = self._guide_intensity(guide)  # 高荧光通道
            north, south, east, west = _neighbors(intensity)
            grad_sq = north * north + south * south + east * east + west * west
            kappa = torch.exp(self.log_kappa).clamp(1e-3, 10.0)
            perona_malik = torch.exp(-grad_sq / (kappa * kappa + 1e-6))  # 边界处 ->0
            conductance = conductance * perona_malik
        return conductance.clamp(1e-4, 1.0)

    def _gaussian_blur(self, x: Tensor, sigma: float) -> Tensor:
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

    def _learnable_conductance(self, feat: Tensor) -> Tensor:
        """特征驱动的各向异性扩散幅值 g∈(0,1)，不含 Perona-Malik 标量门。"""
        return torch.sigmoid(self.cond_conv(feat)).clamp(1e-4, 1.0)

    def _structure_tensor(self, intensity: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """由高荧光强度图估计平滑结构张量 J = Gρ * (∇I ∇Iᵀ)。"""
        smoothed = self._gaussian_blur(intensity, self.struct_pre_sigma)
        north, south, east, west = _neighbors(smoothed)
        grad_x = 0.5 * (east - west)
        grad_y = 0.5 * (south - north)
        j11 = self._gaussian_blur(grad_x * grad_x, self.struct_rho_sigma)
        j12 = self._gaussian_blur(grad_x * grad_y, self.struct_rho_sigma)
        j22 = self._gaussian_blur(grad_y * grad_y, self.struct_rho_sigma)
        return j11, j12, j22

    def _ced_tensor(self, j11: Tensor, j12: Tensor, j22: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """相干增强扩散张量 (Weickert): 沿血管相干方向强扩散, 跨结构方向弱扩散。

        对 2x2 对称结构张量做闭式特征分解, 用 (μ1-μ2)² 相干度调制沿向扩散系数,
        经血管取向重建扩散张量分量 (Dxx, Dxy, Dyy)。参数仅一个可学习对比标量。
        """
        half_trace = 0.5 * (j11 + j22)
        diff = 0.5 * (j11 - j22)
        radius = torch.sqrt(diff * diff + j12 * j12 + 1e-12)
        mu1 = half_trace + radius  # 较大特征值 (跨结构/梯度方向)
        mu2 = half_trace - radius  # 较小特征值 (沿结构/相干方向)
        trace = (mu1 + mu2).clamp_min(1e-9)
        coherence = ((mu1 - mu2) / trace) ** 2  # 尺度无关相干度 ∈ [0,1]
        contrast = torch.exp(self.log_ced_contrast).clamp(1e-3, 1e3)
        alpha = float(self.ced_alpha)
        lam_weak = j11.new_tensor(alpha)
        lam_strong = alpha + (1.0 - alpha) * (1.0 - torch.exp(-coherence / contrast))
        # lam1: μ1(图像梯度=跨血管)方向系数; lam2: μ2(沿血管)方向系数
        if self.ced_direction == "across":
            lam1, lam2 = lam_strong, lam_weak  # 跨血管强扩散: 渗漏向外渗透
        else:
            lam1, lam2 = lam_weak, lam_strong  # 沿血管强扩散: 血管周围套染
        # 用 μ1 特征向量方向角: cos2θ=(a-c)/(2R), sin2θ=b/R
        two_radius = 2.0 * radius + 1e-9
        cos2 = (j11 - j22) / two_radius
        sin2 = (2.0 * j12) / two_radius
        sum_l = 0.5 * (lam1 + lam2)
        dif_l = 0.5 * (lam1 - lam2)
        d_xx = sum_l + dif_l * cos2
        d_yy = sum_l - dif_l * cos2
        d_xy = dif_l * sin2
        return d_xx, d_xy, d_yy

    def _anisotropic_divergence(
        self, u: Tensor, d_xx: Tensor, d_xy: Tensor, d_yy: Tensor, g: Tensor
    ) -> Tensor:
        """通量式中心差分离散 div(g·D∇u), D 为图像导出的各向异性扩散张量。"""
        padded = F.pad(u, (1, 1, 1, 1), mode="replicate")
        u_x = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, 0:-2])
        u_y = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, 0:-2, 1:-1])
        flux_x = d_xx * u_x + d_xy * u_y
        flux_y = d_xy * u_x + d_yy * u_y
        flux_x_p = F.pad(flux_x, (1, 1, 0, 0), mode="replicate")
        flux_y_p = F.pad(flux_y, (0, 0, 1, 1), mode="replicate")
        div_x = 0.5 * (flux_x_p[:, :, :, 2:] - flux_x_p[:, :, :, 0:-2])
        div_y = 0.5 * (flux_y_p[:, :, 2:, :] - flux_y_p[:, :, 0:-2, :])
        return g * (div_x + div_y)

    def _reaction(self, u: Tensor, seed: Tensor) -> Tensor:
        _, rho_value, lambda_value = self._coefficients()
        rho = rho_value.view(1, -1, 1, 1)
        lam = lambda_value.view(1, -1, 1, 1)
        if self.reaction == "pull":
            return rho * (seed - u) - lam * u
        return rho * seed * u * (1.0 - u) - lam * u  # Fisher-KPP 生长

    def _diffusion(self, u: Tensor, conductance: Tensor) -> Tensor:
        north, south, east, west = _neighbors(u)
        if self.flux_scheme == "edge":
            c_north, c_south, c_east, c_west = _neighbors(conductance)
            return (
                (conductance + 0.5 * c_north) * north
                + (conductance + 0.5 * c_south) * south
                + (conductance + 0.5 * c_east) * east
                + (conductance + 0.5 * c_west) * west
            )
        return conductance * (north + south + east + west)

    def _evolve(self, feat: Tensor, guide: Tensor | None):
        seed_logits = self.seed(feat)
        seed = torch.sigmoid(seed_logits)
        dt_value, _, _ = self._coefficients()
        dt = dt_value.view(1, -1, 1, 1)
        use_aniso = self.diffusion_mode == "anisotropic" and guide is not None
        if use_aniso:
            device_type = guide.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                intensity = self._guide_intensity(guide).float()
                j11, j12, j22 = self._structure_tensor(intensity)
                d_xx, d_xy, d_yy = self._ced_tensor(j11, j12, j22)
            conductance = self._learnable_conductance(feat)
        else:
            conductance = self._conductance(feat, guide)
            d_xx = d_xy = d_yy = None
        u = seed
        steps = [u]
        for _ in range(self.iters):
            if use_aniso:
                divergence = self._anisotropic_divergence(u, d_xx, d_xy, d_yy, conductance)
            else:
                divergence = self._diffusion(u, conductance)
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

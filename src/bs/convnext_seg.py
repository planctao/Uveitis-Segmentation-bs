from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.coleak import CoupledLeakageHead
from bs.dual_branch import CoreContourDualBranchHead, RdhDualBranchFusionHead
from bs.edge import EdgeGuidedHead, GeodesicActiveContourHead
from bs.edge import PDCBank
from bs.rdh import ReactionDiffusionHead
from bs.zab import ZABLeakageHead


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class ChannelSpatialAttention(nn.Module):
    """Lightweight CBAM-style attention for fused decoder features."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, int(reduction)))
        kernel = int(spatial_kernel)
        if kernel % 2 == 0:
            kernel += 1
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        channel_gate = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_gate

        spatial_avg = x.mean(dim=1, keepdim=True)
        spatial_max = x.max(dim=1, keepdim=True).values
        spatial_gate = torch.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x * spatial_gate


class EMCADLiteAttention(nn.Module):
    """A lightweight EMCAD-style multi-scale convolutional attention block.

    EMCAD uses multi-scale convolutional attention in the decoder.  This
    compact variant keeps the existing FPN and applies three depthwise
    convolutions (3/5/7 kernels), followed by channel and spatial gates.  A
    residual scale keeps the initial network close to the original decoder,
    which is important for the small medical dataset used here.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # The FPN concatenation has 4*decoder_channels (typically 768)
        # channels.  EMCAD-style attention is normally applied after a
        # channel bottleneck; doing the depthwise branches at full width is
        # unnecessarily expensive on a 22GB GPU.
        inner_channels = max(32, channels // 4)
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, inner_channels),
            nn.GELU(),
        )
        self.multi_scale = nn.ModuleList(
            [
                nn.Conv2d(inner_channels, inner_channels, kernel_size=k, padding=k // 2, groups=inner_channels, bias=False)
                for k in (3, 5, 7)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, inner_channels),
            nn.GELU(),
        )
        hidden = max(1, inner_channels // max(1, int(reduction)))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(inner_channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, inner_channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.out_proj = nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        reduced = self.in_proj(x)
        multi = sum(branch(reduced) for branch in self.multi_scale) / float(len(self.multi_scale))
        multi = self.mix(multi)
        pooled = F.adaptive_avg_pool2d(multi, output_size=1)
        channel_gate = torch.sigmoid(self.channel_mlp(pooled))
        gated = multi * channel_gate
        spatial_avg = gated.mean(dim=1, keepdim=True)
        spatial_max = gated.max(dim=1, keepdim=True).values
        spatial_gate = torch.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x + self.residual_scale * self.out_proj(gated * spatial_gate)


class DynamicSkipMixture(nn.Module):
    """TA-MoSC-style dynamic mixture of the four FPN skip tensors."""

    def __init__(self, channels: int, num_scales: int = 4) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_scales = int(num_scales)
        hidden = max(8, channels // 8)
        self.score = nn.Sequential(
            nn.Conv2d(channels * num_scales, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_scales, kernel_size=1, bias=True),
        )
        # Start from uniform skip mixing, then learn image-dependent routing.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        chunks = x.split(self.channels, dim=1)
        pooled = F.adaptive_avg_pool2d(x, output_size=1)
        weights = torch.softmax(self.score(pooled), dim=1)
        mixed = sum(chunk * weights[:, idx : idx + 1] for idx, chunk in enumerate(chunks))
        # Keep the original fused width so the existing decoder neck/head is
        # unchanged; the routed feature is broadcast as a residual mixture.
        return x + (mixed.repeat(1, self.num_scales, 1, 1) - x) * 0.25


class WaveletBoundaryAttention(nn.Module):
    """Lightweight WBE/PFESA-style wavelet boundary enhancement.

    The original WBE implementation was written for ViT features and applies a
    relatively wide block at every intermediate scale.  On this ConvNeXt FPN,
    the concatenated decoder tensor can have 768 channels, so doing the DWT at
    full width is wasteful.  This variant projects to a small bottleneck,
    enhances Haar high-frequency coefficients, and adds the result back through
    a small residual branch.  The output projection is zero initialized so the
    model starts exactly at the no-wavelet decoder and learns the boundary
    correction progressively.
    """

    def __init__(self, channels: int, bottleneck: int = 96) -> None:
        super().__init__()
        inner = max(32, int(bottleneck))
        self.down = nn.Sequential(
            nn.Conv2d(channels, inner, kernel_size=1, bias=False),
            nn.GroupNorm(8, inner),
            nn.GELU(),
        )
        self.high = nn.Sequential(
            nn.Conv2d(inner, inner, kernel_size=3, padding=1, groups=inner, bias=False),
            nn.GroupNorm(8, inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, kernel_size=1, bias=False),
            nn.GroupNorm(8, inner),
            nn.GELU(),
        )
        hidden = max(8, inner // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, inner, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(inner, inner, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inner),
            nn.GELU(),
        )
        self.up = nn.Conv2d(inner, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _haar(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return (
            (x00 + x01 + x10 + x11) * 0.25,
            (x00 - x01 + x10 - x11) * 0.25,
            (x00 + x01 - x10 - x11) * 0.25,
            (x00 - x01 - x10 + x11) * 0.25,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        reduced = self.down(x)
        ll, lh, hl, hh = self._haar(reduced)
        # Magnitude is more stable than a signed sum for thin boundaries.
        hf = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-6)
        hf = self.high(hf) * self.channel_gate(hf)
        enhanced = self.fuse(ll + hf)
        enhanced = F.interpolate(enhanced, size=identity.shape[-2:], mode="bilinear", align_corners=False)
        return identity + self.residual_scale * self.up(enhanced)


class BoundaryRefinementAttention(nn.Module):
    """Lightweight boundary-guided refinement (ET-Net/CTO-style).

    A depthwise high-pass response and a learned edge gate reweight the neck
    feature before VA-RDH.  It is deliberately residual and zero initialized:
    at initialization this is exactly the original VA-RDH, while training can
    learn to emphasize uncertain lesion contours without replacing the PDE
    head or adding a second prediction/loss branch.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.highpass = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.edge_score = nn.Sequential(
            nn.Conv2d(channels, max(16, channels // 8), kernel_size=1, bias=False),
            nn.GroupNorm(8, max(16, channels // 8)),
            nn.GELU(),
            nn.Conv2d(max(16, channels // 8), 1, kernel_size=1, bias=True),
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        # A Laplacian-like initialization gives a useful edge prior while the
        # learned branch adapts it to the feature statistics.
        with torch.no_grad():
            self.highpass.weight.zero_()
            self.highpass.weight[:, 0, 1, 1] = 1.0
            self.highpass.weight[:, 0, 0, 1] = -0.25
            self.highpass.weight[:, 0, 2, 1] = -0.25
            self.highpass.weight[:, 0, 1, 0] = -0.25
            self.highpass.weight[:, 0, 1, 2] = -0.25

    def forward(self, x: Tensor) -> Tensor:
        hp = self.highpass(x)
        magnitude = hp.abs().mean(dim=1, keepdim=True)
        learned = self.edge_score(x)
        gate = torch.sigmoid(learned + magnitude.detach())
        refined = self.mix(hp * gate)
        return x + self.residual_scale * refined


class OrientedPDCNeckRefinement(nn.Module):
    """Oriented-PDC feature refinement inserted immediately before VA-RDH.

    CPDC/APDC/RPDC are used as a feature extractor only; unlike the standalone
    edge head, this branch has no auxiliary edge target.  A zero-initialized
    projection makes the initial function identical to VA-RDH and lets the
    oriented boundary cue be learned as a small residual correction.
    """

    def __init__(self, channels: int, branch_channels: int = 64) -> None:
        super().__init__()
        branch_channels = max(16, int(branch_channels))
        self.bank = PDCBank(channels, branch_channels, ["cpdc", "apdc", "rpdc"])
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        # Keep the added path dormant at initialization (strict no-op).
        nn.init.zeros_(self.project[0].weight)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual_scale * self.project(self.bank(x))


def build_attention(name: str, channels: int, reduction: int = 16) -> nn.Module:
    normalized = str(name).lower()
    if normalized in {"none", "identity", "off", "false", "0"}:
        return nn.Identity()
    if normalized in {"cbam", "channel_spatial", "channel-spatial"}:
        return ChannelSpatialAttention(channels=channels, reduction=reduction)
    if normalized in {"emcad", "emcad_lite", "emcad-lite", "multi_scale"}:
        return EMCADLiteAttention(channels=channels, reduction=reduction)
    if normalized in {"mosc", "ta-mosc", "dynamic_skip", "dynamic-skip"}:
        if channels % 4 != 0:
            raise ValueError("DynamicSkipMixture expects fused channels divisible by 4")
        return DynamicSkipMixture(channels=channels // 4, num_scales=4)
    if normalized in {"wbe", "wbe-lite", "wavelet", "wavelet-boundary"}:
        return WaveletBoundaryAttention(channels=channels, bottleneck=96)
    if normalized in {"boundary", "boundary-lite", "boundary-refine", "br"}:
        return BoundaryRefinementAttention(channels=channels)
    if normalized in {"pdc-neck", "oriented-pdc-neck", "pdc-refine", "oriented-pdc"}:
        # Applied after the neck, not to the wide FPN concatenation.
        return nn.Identity()
    raise ValueError(f"Unsupported ConvNeXt decoder attention: {name}")


class ConvNeXtFPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        decoder_channels: int = 192,
        out_channels: int = 2,
        attention: str = "none",
        attention_reduction: int = 16,
        deep_supervision: bool = False,
        head_type: str = "conv",
        edge_channels: int = 0,
        edge_pdc_types: list[str] | None = None,
        gac_iters: int = 8,
        gac_dt: float = 0.1,
        gac_beta: float = 0.5,
        gac_kappa: float = 0.1,
        gac_guide_input: str = "normalized",
        rdh_iters: int = 8,
        rdh_dt: float = 0.2,
        rdh_reaction: str = "fisher",
        rdh_use_image_conductance: bool = True,
        rdh_lambda: float = 0.1,
        rdh_rho: float = 1.0,
        rdh_kappa: float = 0.1,
        rdh_dynamics: str = "pde",
        rdh_d_state: int = 16,
        rdh_directions: int = 4,
        rdh_stride: int = 4,
        rdh_d_inner: int = 64,
        rdh_stable_constraints: bool = False,
        rdh_flux_scheme: str = "center",
        rdh_guide_input: str = "normalized",
        rdh_diffusion_mode: str = "isotropic",
        rdh_struct_pre_sigma: float = 1.0,
        rdh_struct_rho_sigma: float = 2.0,
        rdh_ced_alpha: float = 0.005,
        rdh_ced_contrast: float = 1.0,
        rdh_ced_direction: str = "along",
        coleak_topk_fraction: float = 0.05,
        coleak_presence_prior: float = 0.1,
        coleak_prior_strength: float = 0.5,
        zab_topk_fraction: float = 0.05,
        zab_presence_prior: float = 0.11,
        zab_area_prior: float = 0.005,
        zab_max_area_fraction: float = 0.1,
        zab_anatomy_strength: float = 0.75,
        zab_hierarchy_strength: float = 0.0,
        zab_bidirectional_strength: float = 0.0,
        zab_calibration_iterations: int = 3,
        zab_calibration_max_shift: float = 6.0,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.head_type = str(head_type).lower()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvNormAct(decoder_channels, decoder_channels) for _ in range(len(in_channels) - 1)]
        )
        fused_channels = decoder_channels * len(in_channels)
        self.attention = build_attention(attention, channels=fused_channels, reduction=attention_reduction)
        self.post_neck_refinement: nn.Module = nn.Identity()
        if self.head_type in {"rdh", "coleak", "zab", "edge", "gac", "dual_branch", "rdh_dual_branch"}:
            self.neck = nn.Sequential(
                ConvNormAct(fused_channels, decoder_channels),
                ConvNormAct(decoder_channels, decoder_channels),
                nn.Dropout2d(0.1),
            )
            if str(attention).lower() in {"pdc-neck", "oriented-pdc-neck", "pdc-refine", "oriented-pdc"}:
                self.post_neck_refinement = OrientedPDCNeckRefinement(decoder_channels, branch_channels=64)
        if self.head_type == "rdh":
            # 物理演化头：neck 产生特征，再由反应-扩散演化出分割
            self.rdh_head = ReactionDiffusionHead(
                decoder_channels,
                out_channels=out_channels,
                iters=rdh_iters,
                dt=rdh_dt,
                reaction=rdh_reaction,
                use_image_conductance=rdh_use_image_conductance,
                lambda_init=rdh_lambda,
                rho_init=rdh_rho,
                kappa=rdh_kappa,
                dynamics=rdh_dynamics,
                d_state=rdh_d_state,
                ssm_directions=rdh_directions,
                ssm_stride=rdh_stride,
                ssm_d_inner=rdh_d_inner,
                stable_constraints=rdh_stable_constraints,
                flux_scheme=rdh_flux_scheme,
                guide_input=rdh_guide_input,
                diffusion_mode=rdh_diffusion_mode,
                struct_pre_sigma=rdh_struct_pre_sigma,
                struct_rho_sigma=rdh_struct_rho_sigma,
                ced_alpha=rdh_ced_alpha,
                ced_contrast=rdh_ced_contrast,
                ced_direction=rdh_ced_direction,
            )
            # 保留可选的多尺度辅助监督。主输出仍由 RDH 产生；当
            # decoder_deep_supervision=true 时，训练阶段额外返回各级 FPN
            # 辅助 logits，推理阶段接口仍为单个 logits tensor。
        elif self.head_type == "coleak":
            self.coleak_head = CoupledLeakageHead(
                in_channels=decoder_channels,
                global_channels=decoder_channels,
                topk_fraction=coleak_topk_fraction,
                presence_prior=coleak_presence_prior,
                prior_strength=coleak_prior_strength,
            )
            self.deep_supervision = False
        elif self.head_type == "zab":
            self.zab_head = ZABLeakageHead(
                in_channels=decoder_channels,
                global_channels=decoder_channels,
                topk_fraction=zab_topk_fraction,
                presence_prior=zab_presence_prior,
                area_prior=zab_area_prior,
                max_area_fraction=zab_max_area_fraction,
                anatomy_strength=zab_anatomy_strength,
                hierarchy_strength=zab_hierarchy_strength,
                bidirectional_strength=zab_bidirectional_strength,
                calibration_iterations=zab_calibration_iterations,
                calibration_max_shift=zab_calibration_max_shift,
            )
            self.deep_supervision = False
        elif self.head_type == "edge":
            # PDC 边缘分支 + 边缘门控增强的分割头（CTO/ET-Net 式边缘引导）
            self.edge_head = EdgeGuidedHead(
                decoder_channels,
                out_channels=out_channels,
                edge_channels=edge_channels,
                edge_pdc_types=edge_pdc_types,
            )
            self.deep_supervision = False
        elif self.head_type == "gac":
            # 测地主动轮廓边界演化头（边缘指示驱动 + PDC 学习边缘监督）
            self.gac_head = GeodesicActiveContourHead(
                decoder_channels,
                out_channels=out_channels,
                edge_channels=edge_channels,
                iters=gac_iters,
                dt=gac_dt,
                beta=gac_beta,
                kappa=gac_kappa,
                edge_pdc_types=edge_pdc_types,
                guide_input=gac_guide_input,
            )
            self.deep_supervision = False
        elif self.head_type == "dual_branch":
            # CSD-DB: source/core 检测 + contour/edge 检测双分支融合头
            self.dual_branch_head = CoreContourDualBranchHead(
                decoder_channels,
                out_channels=out_channels,
                branch_channels=edge_channels,
                edge_pdc_types=edge_pdc_types,
            )
            self.deep_supervision = False
        elif self.head_type == "rdh_dual_branch":
            # VA-RDH 主分割 + CSD-DB 双检测辅助，residual_scale 零初始化保证初始等价 RDH
            self.rdh_dual_branch_head = RdhDualBranchFusionHead(
                decoder_channels,
                out_channels=out_channels,
                branch_channels=edge_channels,
                edge_pdc_types=edge_pdc_types,
                rdh_kwargs={
                    "iters": rdh_iters,
                    "dt": rdh_dt,
                    "reaction": rdh_reaction,
                    "use_image_conductance": rdh_use_image_conductance,
                    "lambda_init": rdh_lambda,
                    "rho_init": rdh_rho,
                    "kappa": rdh_kappa,
                    "dynamics": rdh_dynamics,
                    "d_state": rdh_d_state,
                    "ssm_directions": rdh_directions,
                    "ssm_stride": rdh_stride,
                    "ssm_d_inner": rdh_d_inner,
                    "stable_constraints": rdh_stable_constraints,
                    "flux_scheme": rdh_flux_scheme,
                    "guide_input": rdh_guide_input,
                    "diffusion_mode": rdh_diffusion_mode,
                    "struct_pre_sigma": rdh_struct_pre_sigma,
                    "struct_rho_sigma": rdh_struct_rho_sigma,
                    "ced_alpha": rdh_ced_alpha,
                    "ced_contrast": rdh_ced_contrast,
                    "ced_direction": rdh_ced_direction,
                },
            )
            self.deep_supervision = False
        else:
            self.fuse = nn.Sequential(
                ConvNormAct(fused_channels, decoder_channels),
                ConvNormAct(decoder_channels, decoder_channels),
                nn.Dropout2d(0.1),
                nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
            )
        self.aux_heads = (
            nn.ModuleList([nn.Conv2d(decoder_channels, out_channels, kernel_size=1) for _ in range(len(in_channels) - 1)])
            if self.deep_supervision
            else nn.ModuleList()
        )

    def forward(
        self, features: list[Tensor], output_size: tuple[int, int], images: Tensor | None = None
    ) -> Tensor | tuple[Tensor, list[Tensor]] | tuple[Tensor, dict[str, Tensor]]:
        pyramid = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for idx in range(len(pyramid) - 1, 0, -1):
            upsampled = F.interpolate(pyramid[idx], size=pyramid[idx - 1].shape[-2:], mode="bilinear", align_corners=False)
            pyramid[idx - 1] = self.smooth[idx - 1](pyramid[idx - 1] + upsampled)

        target_size = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                for feature in pyramid
            ],
            dim=1,
        )
        fused = self.attention(fused)
        if self.head_type == "rdh":
            feat = self.neck(fused)
            feat = self.post_neck_refinement(feat)
            guide = None
            needs_guide = self.rdh_head.use_image_conductance or self.rdh_head.diffusion_mode == "anisotropic"
            if images is not None and needs_guide:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.rdh_head(feat, guide)
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            if self.deep_supervision and self.training:
                aux_logits = [
                    F.interpolate(head(feature), size=output_size, mode="bilinear", align_corners=False)
                    for head, feature in zip(self.aux_heads, pyramid[1:])
                ]
                return logits, aux_logits
            return logits
        if self.head_type == "coleak":
            feat = self.neck(fused)
            logits, auxiliary = self.coleak_head(feat, pyramid[-1])
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            return (logits, auxiliary) if self.training else logits
        if self.head_type == "zab":
            feat = self.neck(fused)
            logits, auxiliary = self.zab_head(feat, pyramid[-1])
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            return (logits, auxiliary) if self.training else logits
        if self.head_type == "edge":
            feat = self.neck(fused)
            seg_logits, edge_logits = self.edge_head(feat)
            seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
            if not self.training:
                return seg_logits
            edge_logits = F.interpolate(edge_logits, size=output_size, mode="bilinear", align_corners=False)
            return seg_logits, {"edge_logits": edge_logits}
        if self.head_type == "gac":
            feat = self.neck(fused)
            guide = None
            if images is not None:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            seg_logits, edge_logits = self.gac_head(feat, guide)
            seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
            if not self.training:
                return seg_logits
            edge_logits = F.interpolate(edge_logits, size=output_size, mode="bilinear", align_corners=False)
            return seg_logits, {"edge_logits": edge_logits}
        if self.head_type == "dual_branch":
            feat = self.neck(fused)
            seg_logits, auxiliary = self.dual_branch_head(feat)
            seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
            if not self.training:
                return seg_logits
            resized_auxiliary = {
                key: F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
                if value.ndim == 4
                else value
                for key, value in auxiliary.items()
            }
            return seg_logits, resized_auxiliary
        if self.head_type == "rdh_dual_branch":
            feat = self.neck(fused)
            guide = None
            needs_guide = self.rdh_dual_branch_head.rdh_head.use_image_conductance or self.rdh_dual_branch_head.rdh_head.diffusion_mode == "anisotropic"
            if images is not None and needs_guide:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            seg_logits, auxiliary = self.rdh_dual_branch_head(feat, guide)
            seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
            if not self.training:
                return seg_logits
            resized_auxiliary = {
                key: F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
                if value.ndim == 4
                else value
                for key, value in auxiliary.items()
            }
            return seg_logits, resized_auxiliary
        logits = self.fuse(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        if not self.deep_supervision or not self.training:
            return logits

        aux_logits = [
            F.interpolate(head(feature), size=output_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.aux_heads, pyramid[1:])
        ]
        return logits, aux_logits


class DinoV3ConvNeXtSegmentationModel(nn.Module):
    def __init__(
        self,
        dinov3_code_dir: str | Path,
        weights_path: str | Path | None,
        variant: str = "tiny",
        decoder_channels: int = 192,
        freeze_backbone: bool = False,
        decoder_attention: str = "none",
        decoder_attention_reduction: int = 16,
        decoder_deep_supervision: bool = False,
        head_type: str = "conv",
        edge_channels: int = 0,
        edge_pdc_types: list[str] | None = None,
        gac_iters: int = 8,
        gac_dt: float = 0.1,
        gac_beta: float = 0.5,
        gac_kappa: float = 0.1,
        gac_guide_input: str = "normalized",
        rdh_iters: int = 8,
        rdh_dt: float = 0.2,
        rdh_reaction: str = "fisher",
        rdh_use_image_conductance: bool = True,
        rdh_lambda: float = 0.1,
        rdh_rho: float = 1.0,
        rdh_kappa: float = 0.1,
        rdh_dynamics: str = "pde",
        rdh_d_state: int = 16,
        rdh_directions: int = 4,
        rdh_stride: int = 4,
        rdh_d_inner: int = 64,
        rdh_stable_constraints: bool = False,
        rdh_flux_scheme: str = "center",
        rdh_guide_input: str = "normalized",
        rdh_diffusion_mode: str = "isotropic",
        rdh_struct_pre_sigma: float = 1.0,
        rdh_struct_rho_sigma: float = 2.0,
        rdh_ced_alpha: float = 0.005,
        rdh_ced_contrast: float = 1.0,
        rdh_ced_direction: str = "along",
        coleak_topk_fraction: float = 0.05,
        coleak_presence_prior: float = 0.1,
        coleak_prior_strength: float = 0.5,
        zab_topk_fraction: float = 0.05,
        zab_presence_prior: float = 0.11,
        zab_area_prior: float = 0.005,
        zab_max_area_fraction: float = 0.1,
        zab_anatomy_strength: float = 0.75,
        zab_hierarchy_strength: float = 0.0,
        zab_bidirectional_strength: float = 0.0,
        zab_calibration_iterations: int = 3,
        zab_calibration_max_shift: float = 6.0,
    ) -> None:
        super().__init__()
        code_dir = str(Path(dinov3_code_dir).resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from dinov3.hub.backbones import dinov3_convnext_small, dinov3_convnext_tiny

        builders = {
            "tiny": dinov3_convnext_tiny,
            "small": dinov3_convnext_small,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
        self.backbone = builders[variant](pretrained=False)
        if weights_path is not None:
            state_dict = torch.load(Path(weights_path).resolve(), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)
        self.decode_head = ConvNeXtFPNDecoder(
            in_channels=list(self.backbone.embed_dims),
            decoder_channels=decoder_channels,
            out_channels=2,
            attention=decoder_attention,
            attention_reduction=decoder_attention_reduction,
            deep_supervision=decoder_deep_supervision,
            head_type=head_type,
            edge_channels=edge_channels,
            edge_pdc_types=edge_pdc_types,
            gac_iters=gac_iters,
            gac_dt=gac_dt,
            gac_beta=gac_beta,
            gac_kappa=gac_kappa,
            gac_guide_input=gac_guide_input,
            rdh_iters=rdh_iters,
            rdh_dt=rdh_dt,
            rdh_reaction=rdh_reaction,
            rdh_use_image_conductance=rdh_use_image_conductance,
            rdh_lambda=rdh_lambda,
            rdh_rho=rdh_rho,
            rdh_kappa=rdh_kappa,
            rdh_dynamics=rdh_dynamics,
            rdh_d_state=rdh_d_state,
            rdh_directions=rdh_directions,
            rdh_stride=rdh_stride,
            rdh_d_inner=rdh_d_inner,
            rdh_stable_constraints=rdh_stable_constraints,
            rdh_flux_scheme=rdh_flux_scheme,
            rdh_guide_input=rdh_guide_input,
            rdh_diffusion_mode=rdh_diffusion_mode,
            rdh_struct_pre_sigma=rdh_struct_pre_sigma,
            rdh_struct_rho_sigma=rdh_struct_rho_sigma,
            rdh_ced_alpha=rdh_ced_alpha,
            rdh_ced_contrast=rdh_ced_contrast,
            rdh_ced_direction=rdh_ced_direction,
            coleak_topk_fraction=coleak_topk_fraction,
            coleak_presence_prior=coleak_presence_prior,
            coleak_prior_strength=coleak_prior_strength,
            zab_topk_fraction=zab_topk_fraction,
            zab_presence_prior=zab_presence_prior,
            zab_area_prior=zab_area_prior,
            zab_max_area_fraction=zab_max_area_fraction,
            zab_anatomy_strength=zab_anatomy_strength,
            zab_hierarchy_strength=zab_hierarchy_strength,
            zab_bidirectional_strength=zab_bidirectional_strength,
            zab_calibration_iterations=zab_calibration_iterations,
            zab_calibration_max_shift=zab_calibration_max_shift,
        )
        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        self.backbone.train(trainable)

    def train(self, mode: bool = True) -> "DinoV3ConvNeXtSegmentationModel":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_multiscale_features(self, images: Tensor) -> list[Tensor]:
        features = []
        x = images
        for downsample, stage in zip(self.backbone.downsample_layers, self.backbone.stages):
            x = downsample(x)
            x = stage(x)
            features.append(x)
        return features

    def forward(
        self, images: Tensor
    ) -> Tensor | tuple[Tensor, list[Tensor]] | tuple[Tensor, dict[str, Tensor]]:
        output_size = tuple(images.shape[-2:])
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.extract_multiscale_features(images)
        else:
            features = self.extract_multiscale_features(images)
        return self.decode_head(features, output_size, images=images)

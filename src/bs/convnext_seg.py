from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.coleak import CoupledLeakageHead
from bs.dual_branch import CoreContourDualBranchHead, RdhDualBranchFusionHead
from bs.edge import EdgeGuidedHead, GeodesicActiveContourHead
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


def build_attention(name: str, channels: int, reduction: int = 16) -> nn.Module:
    normalized = str(name).lower()
    if normalized in {"none", "identity", "off", "false", "0"}:
        return nn.Identity()
    if normalized in {"cbam", "channel_spatial", "channel-spatial"}:
        return ChannelSpatialAttention(channels=channels, reduction=reduction)
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
        if self.head_type in {"rdh", "coleak", "zab", "edge", "gac", "dual_branch", "rdh_dual_branch"}:
            self.neck = nn.Sequential(
                ConvNormAct(fused_channels, decoder_channels),
                ConvNormAct(decoder_channels, decoder_channels),
                nn.Dropout2d(0.1),
            )
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
            self.deep_supervision = False  # RDH 暂不与深监督组合
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
            guide = None
            needs_guide = self.rdh_head.use_image_conductance or self.rdh_head.diffusion_mode == "anisotropic"
            if images is not None and needs_guide:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.rdh_head(feat, guide)
            return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
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

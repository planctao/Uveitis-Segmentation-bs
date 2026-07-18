from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.rdh import ReactionDiffusionHead


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
        if self.head_type == "rdh":
            # 物理演化头：neck 产生特征，再由反应-扩散演化出分割
            self.neck = nn.Sequential(
                ConvNormAct(fused_channels, decoder_channels),
                ConvNormAct(decoder_channels, decoder_channels),
                nn.Dropout2d(0.1),
            )
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
            )
            self.deep_supervision = False  # RDH 暂不与深监督组合
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
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
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
            if images is not None and self.rdh_head.use_image_conductance:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.rdh_head(feat, guide)
            return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
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
        weights_path: str | Path,
        variant: str = "tiny",
        decoder_channels: int = 192,
        freeze_backbone: bool = False,
        decoder_attention: str = "none",
        decoder_attention_reduction: int = 16,
        decoder_deep_supervision: bool = False,
        head_type: str = "conv",
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

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
        output_size = tuple(images.shape[-2:])
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.extract_multiscale_features(images)
        else:
            features = self.extract_multiscale_features(images)
        return self.decode_head(features, output_size, images=images)

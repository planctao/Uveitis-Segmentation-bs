"""MAE-SAM2: Mask Autoencoder-Enhanced SAM2 for retinal vascular leakage segmentation.

Implements:
  - Hiera backbone (matching original SAM2 checkpoint keys)
  - FPN neck for multi-scale features
  - MAE decoder for self-supervised pre-training
  - Segmentation head for fine-tuning

Reference: "MAE-SAM2: Mask Autoencoder-Enhanced SAM2 for Clinical Retinal
Vascular Leakage Segmentation" (arXiv:2509.10554)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Hiera attention (matches checkpoint: attn.qkv, attn.proj)
# ---------------------------------------------------------------------------


class HieraAttention(nn.Module):
    """Multi-head attention with fused QKV matching SAM2 checkpoint keys."""

    def __init__(self, dim: int, num_heads: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x


class HieraMLP(nn.Module):
    """MLP matching checkpoint: mlp.layers.0, mlp.layers.1."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers[0](x)
        x = F.gelu(x)
        x = self.layers[1](x)
        return x


class HieraBlock(nn.Module):
    """Single Hiera block.

    Stage-transition blocks have a ``proj`` linear for the residual connection
    (in_dim → out_dim) and perform 2×2 query pooling.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        out_dim: int | None = None,
        mlp_ratio: float = 4.0,
        window_size: int = 8,
        is_global: bool = False,
        stage_transition: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.window_size = window_size
        self.is_global = is_global
        self.stage_transition = stage_transition

        self.norm1 = nn.LayerNorm(dim)
        self.attn = HieraAttention(dim, num_heads, out_dim=self.out_dim)

        if stage_transition:
            self.proj = nn.Linear(dim, self.out_dim)
        else:
            assert dim == self.out_dim

        self.norm2 = nn.LayerNorm(self.out_dim)
        hidden = int(self.out_dim * mlp_ratio)
        self.mlp = HieraMLP(self.out_dim, hidden, self.out_dim)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        """Args: x (B, H*W, C), H, W. Returns: x (B, H'*W', C'), H', W'."""
        B, N, C = x.shape
        residual = x

        x = self.norm1(x)

        if self.is_global or self.window_size is None or self.window_size <= 0:
            # Global attention over all patches
            x = self.attn(x)
        else:
            # Windowed attention: partition into windows, attend within each
            x = self._windowed_attn(x, H, W)

        if self.stage_transition:
            residual = self.proj(residual)  # (B, N, out_dim)

        x = residual + x
        x = x + self.mlp(self.norm2(x))

        # Query pooling
        if self.stage_transition and H > 1 and W > 1:
            x = x.view(B, H, W, self.out_dim)
            x = x.view(B, H // 2, 2, W // 2, 2, self.out_dim).mean(dim=(2, 4))
            x = x.view(B, (H // 2) * (W // 2), self.out_dim)
            H, W = H // 2, W // 2

        return x, H, W

    def _windowed_attn(self, x: Tensor, H: int, W: int) -> Tensor:
        """Windowed multi-head self-attention."""
        B, N, C = x.shape
        ws = self.window_size
        # Pad to multiple of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_p, W_p = H + pad_h, W + pad_w
            x = x.view(B, H_p * W_p, C)
        else:
            H_p, W_p = H, W

        # Partition into windows: (B, nH, ws, nW, ws, C) -> (B*nH*nW, ws*ws, C)
        x = x.view(B, H_p, W_p, C)
        nH, nW = H_p // ws, W_p // ws
        x = x.view(B, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * nH * nW, ws * ws, C)

        # Attention within each window
        x = self.attn(x)

        # Merge windows back
        x = x.view(B, nH, nW, ws, ws, -1).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H_p * W_p, -1)

        # Remove padding
        if pad_h or pad_w:
            x = x.view(B, H_p, W_p, -1)
            x = x[:, :H, :W, :].reshape(B, H * W, -1)

        return x


# ---------------------------------------------------------------------------
# Hiera trunk
# ---------------------------------------------------------------------------


class HieraPatchEmbed(nn.Module):
    """Patch embedding: Conv2d 7×7 stride 4."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class HieraTrunk(nn.Module):
    """Hiera trunk matching checkpoint keys: trunk.patch_embed, trunk.pos_embed,
    trunk.pos_embed_window, trunk.blocks.{N}.*"""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: list[int] = [1, 2, 11, 2],
        global_att_blocks: list[int] = [7, 10, 13],
        window_pos_embed_bkg_spatial_size: list[int] = [7, 7],
        mlp_ratio: float = 4.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stages = stages
        self.global_att_blocks = set(global_att_blocks)

        self.patch_embed = HieraPatchEmbed(in_chans, embed_dim)

        pos_h, pos_w = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, pos_h, pos_w))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, pos_h + 1, pos_w + 1))

        blocks: list[HieraBlock] = []
        cur_dim = embed_dim
        block_idx = 0
        for stage_idx, num_blocks in enumerate(stages):
            for j in range(num_blocks):
                is_transition = j == 0 and stage_idx > 0
                if is_transition:
                    out_dim = cur_dim * 2
                    n_heads = num_heads * (2 ** stage_idx)
                    blocks.append(HieraBlock(
                        dim=cur_dim, num_heads=n_heads, out_dim=out_dim,
                        mlp_ratio=mlp_ratio, is_global=block_idx in self.global_att_blocks,
                        stage_transition=True,
                    ))
                    cur_dim = out_dim
                else:
                    n_heads = num_heads * (2 ** stage_idx)
                    blocks.append(HieraBlock(
                        dim=cur_dim, num_heads=n_heads, out_dim=cur_dim,
                        mlp_ratio=mlp_ratio, is_global=block_idx in self.global_att_blocks,
                        stage_transition=False,
                    ))
                block_idx += 1

        self.blocks = nn.ModuleList(blocks)
        self.stage_dims = []
        d = embed_dim
        for s in range(len(stages)):
            if s > 0:
                d *= 2
            self.stage_dims.append(d)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Returns list of stage feature maps [(B,C,H,W), ...] from fine to coarse."""
        x, H, W = self.patch_embed(x)
        B, N, C = x.shape

        # Add position embedding
        x = x.view(B, H, W, C)
        pos = F.interpolate(self.pos_embed, size=(H, W), mode="bilinear", align_corners=False)
        x = x + pos.permute(0, 2, 3, 1)
        x = x.view(B, N, C)

        stage_outputs = []
        block_idx = 0
        for stage_idx, num_blocks in enumerate(stages_var := self.stages):
            for j in range(num_blocks):
                block = self.blocks[block_idx]
                x, H, W = block(x, H, W)
                block_idx += 1
            B, N, C = x.shape
            x_spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)
            stage_outputs.append(x_spatial)

        return stage_outputs


# ---------------------------------------------------------------------------
# FPN neck (matches checkpoint: neck.convs.{N}.conv)
# ---------------------------------------------------------------------------


class FpnConvBlock(nn.Module):
    """Wrapper so checkpoint key is neck.convs.{N}.conv.*"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class FpnNeck(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        backbone_channel_list: list[int] = [768, 384, 192, 96],
        fpn_top_down_levels: list[int] = [2, 3],
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [FpnConvBlock(ch, d_model) for ch in backbone_channel_list]
        )
        self.fpn_top_down_levels = set(fpn_top_down_levels)

    def forward(self, feats: list[Tensor]) -> Tensor:
        """feats: high-res → low-res. Returns finest-res FPN output."""
        projected = [blk(f) for blk, f in zip(self.convs, feats)]
        for i in range(len(projected) - 1, 0, -1):
            if i in self.fpn_top_down_levels or (i - 1) in self.fpn_top_down_levels:
                target = projected[i - 1].shape[-2:]
                up = F.interpolate(projected[i], size=target, mode="nearest")
                projected[i - 1] = projected[i - 1] + up
        return projected[0]


# ---------------------------------------------------------------------------
# SAM2 image encoder
# ---------------------------------------------------------------------------


class SAM2ImageEncoder(nn.Module):
    """Full image encoder: Hiera trunk + FPN neck."""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: list[int] = [1, 2, 11, 2],
        global_att_blocks: list[int] = [7, 10, 13],
        window_pos_embed_bkg_spatial_size: list[int] = [7, 7],
        backbone_channel_list: list[int] = [768, 384, 192, 96],
        fpn_d_model: int = 256,
        fpn_top_down_levels: list[int] = [2, 3],
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.trunk = HieraTrunk(
            embed_dim=embed_dim, num_heads=num_heads, stages=stages,
            global_att_blocks=global_att_blocks,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            in_chans=in_chans,
        )
        self.neck = FpnNeck(
            d_model=fpn_d_model,
            backbone_channel_list=backbone_channel_list,
            fpn_top_down_levels=fpn_top_down_levels,
        )
        self.out_dim = fpn_d_model

    def forward(self, x: Tensor) -> Tensor:
        stage_feats = self.trunk(x)  # fine → coarse: [96ch, 192ch, 384ch, 768ch]
        feats_ordered = list(reversed(stage_feats))  # coarse → fine: [768, 384, 192, 96]
        return self.neck(feats_ordered)


# ---------------------------------------------------------------------------
# MAE decoder
# ---------------------------------------------------------------------------


class MAEDecoder(nn.Module):
    """Lightweight decoder: 4× (Conv3×3 + BN + ReLU + Upsample2×)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 3) -> None:
        super().__init__()
        ch = in_channels
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ))
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 2, mid_channels: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
        )
        self.cls = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cls(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Full MAE-SAM2 model
# ---------------------------------------------------------------------------


class MAESAM2(nn.Module):
    """MAE-SAM2 for retinal vascular leakage segmentation.

    Stage 1 – MAE pre-training: encoder + MAE decoder reconstruct masked image.
    Stage 2 – Fine-tuning: encoder + segmentation head for multi-label segmentation.
    """

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        model_variant: str = "small",
        num_classes: int = 2,
        seg_mid_channels: int = 128,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        if model_variant == "tiny":
            stages = [1, 2, 7, 2]
            global_att_blocks = [5, 7, 9]
            ckpt_file = "sam2_hiera_tiny.pt"
            ms_dir = f"sam2-hiera-tiny"
        else:
            stages = [1, 2, 11, 2]
            global_att_blocks = [7, 10, 13]
            ckpt_file = "sam2_hiera_small.pt"
            ms_dir = "sam2-hiera-small"

        self.encoder = SAM2ImageEncoder(
            embed_dim=96, num_heads=1, stages=stages,
            global_att_blocks=global_att_blocks,
        )
        self.mae_decoder = MAEDecoder(in_channels=256, out_channels=3)
        self.seg_head = SegmentationHead(
            in_channels=256, num_classes=num_classes, mid_channels=seg_mid_channels,
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Load pretrained weights
        if ckpt_path is not None:
            self._load_sam2_weights(ckpt_path)
        else:
            default = Path("/root/.cache/modelscope/hub/models/AI-ModelScope") / ms_dir / ckpt_file
            if default.exists():
                self._load_sam2_weights(default)

    def _load_sam2_weights(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt["model"]
        ie_sd = {k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")}
        result = self.encoder.load_state_dict(ie_sd, strict=False)
        n_miss = len(result.missing_keys)
        n_unexp = len(result.unexpected_keys)
        print(f"[MAE-SAM2] Loaded SAM2 encoder: missing={n_miss}, unexpected={n_unexp}")
        if n_miss:
            print(f"  Missing: {result.missing_keys[:5]}")
        if n_unexp:
            print(f"  Unexpected: {result.unexpected_keys[:5]}")

    def forward_mae(self, images: Tensor, mask_ratio: float = 0.75) -> dict[str, Tensor]:
        B, C, H, W = images.shape
        feat = self.encoder(images)
        fH, fW = feat.shape[-2:]
        num_patches = fH * fW
        num_masked = int(num_patches * mask_ratio)

        noise = torch.rand(B, num_patches, device=feat.device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, num_patches, device=feat.device)
        mask.scatter_(1, ids[:, :num_masked], 1.0)
        mask = mask.view(B, fH, fW, 1).permute(0, 3, 1, 2)

        masked_feat = feat * (1 - mask)
        pred = self.mae_decoder(masked_feat)
        if pred.shape[-2:] != images.shape[-2:]:
            pred = F.interpolate(pred, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {"pred": pred, "target": images, "mask": mask}

    def forward(self, images: Tensor) -> Tensor:
        output_size = tuple(images.shape[-2:])
        feat = self.encoder(images)
        return self.seg_head(feat, output_size)

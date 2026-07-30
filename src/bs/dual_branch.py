from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.edge import PDCBank, PDCBlock, lesion_edge_target
from bs.multilabel import masks_to_paper_targets
from bs.rdh import ReactionDiffusionHead


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


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


def source_core_target(
    target: Tensor,
    valid: Tensor,
    erosion_kernel: int = 5,
    soft_sigma: float = 2.0,
) -> Tensor:
    """由病灶 mask 生成 source/core 热图目标。

    先腐蚀得到稳定高置信核心；极小病灶腐蚀后为空则回退原 mask，避免 lesion_2 消失；
    再用高斯扩散得到柔和起漏源热图，并按每图/通道最大值归一化到 [0,1]。
    """
    source = (target.float() * valid.float()).clamp(0.0, 1.0)
    if erosion_kernel > 1:
        kernel = erosion_kernel if erosion_kernel % 2 == 1 else erosion_kernel + 1
        eroded = 1.0 - F.max_pool2d(1.0 - source, kernel_size=kernel, stride=1, padding=kernel // 2)
        eroded = (eroded * valid.float()).clamp(0.0, 1.0)
        keep_eroded = eroded.flatten(2).sum(dim=-1, keepdim=True).view(eroded.shape[0], eroded.shape[1], 1, 1) > 0
        source = torch.where(keep_eroded, eroded, source)
    if soft_sigma > 0.0:
        source = _gaussian_blur(source, soft_sigma) * valid.float()
        max_value = source.flatten(2).amax(dim=-1).clamp_min(1e-6).view(source.shape[0], source.shape[1], 1, 1)
        source = source / max_value
    return source.clamp(0.0, 1.0)


class CoreContourDualBranchHead(nn.Module):
    """CSD-DB: core/source + contour/edge 双分支检测头。

    Source 分支检测高荧光渗漏核心/起漏点；Contour 分支用 Oriented-PDC 检测病灶轮廓；
    两者以零初始化 gate 调制主分割特征，使初始行为接近普通分割头，训练后再学习“源点
    与轮廓共同决定分割”的检测-分割闭环。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        branch_channels: int = 0,
        edge_pdc_types: list[str] | None = None,
    ) -> None:
        super().__init__()
        branch_channels = int(branch_channels) if int(branch_channels) > 0 else max(32, in_channels // 2)
        pdc_types = [str(t).lower() for t in (edge_pdc_types or ["cpdc", "apdc", "rpdc"])] or ["cpdc", "apdc", "rpdc"]
        self.source_branch = nn.Sequential(
            ConvNormAct(in_channels, branch_channels),
            ConvNormAct(branch_channels, branch_channels),
        )
        self.source_out = nn.Conv2d(branch_channels, out_channels, kernel_size=1)
        self.edge_branch = (
            PDCBlock(in_channels, branch_channels, pdc_type=pdc_types[0])
            if len(pdc_types) == 1
            else PDCBank(in_channels, branch_channels, pdc_types)
        )
        self.edge_out = nn.Conv2d(branch_channels, out_channels, kernel_size=1)
        self.source_gate = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.edge_gate = nn.Conv2d(branch_channels, in_channels, kernel_size=1)
        self.seg_fuse = nn.Sequential(
            ConvNormAct(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.source_gate.weight)
        nn.init.zeros_(self.source_gate.bias)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)

    def forward(self, feat: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        source_feat = self.source_branch(feat)
        source_logits = self.source_out(source_feat)
        edge_feat = self.edge_branch(feat)
        edge_logits = self.edge_out(edge_feat)
        source_gate = self.source_gate(torch.sigmoid(source_logits))
        edge_gate = self.edge_gate(edge_feat)
        gate = torch.tanh(source_gate + edge_gate)
        refined = feat * (1.0 + gate)
        seg_logits = self.seg_fuse(refined)
        return seg_logits, {
            "source_logits": source_logits,
            "edge_logits": edge_logits,
        }


class RdhDualBranchFusionHead(nn.Module):
    """VA-RDH 主分割 + CSD-DB source/contour 双检测辅助的安全融合头。

    RDH logits 是主输出；CSD-DB 产生 source/edge 辅助检测和一个 dual residual logits。
    residual_scale 零初始化，所以初始严格等价于 RDH 主头，训练中再学习是否吸收双分支校正。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        branch_channels: int = 0,
        edge_pdc_types: list[str] | None = None,
        rdh_kwargs: dict | None = None,
        residual_init: float = 0.0,
        residual_gain: float = 0.25,
    ) -> None:
        super().__init__()
        self.rdh_head = ReactionDiffusionHead(in_channels, out_channels=out_channels, **(rdh_kwargs or {}))
        self.dual_head = CoreContourDualBranchHead(
            in_channels,
            out_channels=out_channels,
            branch_channels=branch_channels,
            edge_pdc_types=edge_pdc_types,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.residual_gain = float(residual_gain)

    def forward(self, feat: Tensor, guide: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        rdh_logits = self.rdh_head(feat, guide)
        dual_logits, auxiliary = self.dual_head(feat)
        scale = self.residual_gain * torch.tanh(self.residual_scale)
        logits = rdh_logits + scale * dual_logits
        auxiliary = dict(auxiliary)
        auxiliary["dual_logits"] = dual_logits
        auxiliary["rdh_logits"] = rdh_logits
        return logits, auxiliary


class DualBranchLoss(nn.Module):
    """分割损失 + source/core 检测 + contour/edge 检测 + source-inside 一致性。"""

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        edge_weight: float = 0.5,
        edge_band: int = 5,
        edge_soft: bool = True,
        edge_soft_sigma: float = 1.5,
        edge_dice_weight: float = 0.5,
        edge_pos_weight: list[float] | tuple[float, float] = (5.0, 20.0),
        source_weight: float = 0.3,
        source_erosion_kernel: int = 5,
        source_soft_sigma: float = 2.0,
        source_dice_weight: float = 0.5,
        source_pos_weight: list[float] | tuple[float, float] = (4.0, 30.0),
        consistency_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.edge_weight = float(edge_weight)
        self.edge_band = int(edge_band)
        self.edge_soft = bool(edge_soft)
        self.edge_soft_sigma = float(edge_soft_sigma)
        self.edge_dice_weight = float(edge_dice_weight)
        edge_pos = torch.as_tensor(edge_pos_weight, dtype=torch.float32).flatten()
        if edge_pos.numel() == 1:
            edge_pos = edge_pos.repeat(2)
        self.register_buffer("edge_pos_weight", edge_pos)
        self.source_weight = float(source_weight)
        self.source_erosion_kernel = int(source_erosion_kernel)
        self.source_soft_sigma = float(source_soft_sigma)
        self.source_dice_weight = float(source_dice_weight)
        source_pos = torch.as_tensor(source_pos_weight, dtype=torch.float32).flatten()
        if source_pos.numel() == 1:
            source_pos = source_pos.repeat(2)
        self.register_buffer("source_pos_weight", source_pos)
        self.consistency_weight = float(consistency_weight)
        self.eps = float(eps)

    def _aux_bce_dice(
        self,
        logits: Tensor,
        target: Tensor,
        valid: Tensor,
        pos_weight: Tensor,
        dice_weight: float,
    ) -> Tensor:
        pos = pos_weight.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        valid = valid.to(device=logits.device, dtype=logits.dtype).expand_as(logits)
        target = target.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        if dice_weight <= 0.0:
            return bce
        probs = torch.sigmoid(logits) * valid
        target = target * valid
        dims = (0, 2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = (probs + target).sum(dim=dims).clamp_min(self.eps)
        dice = (1.0 - (2.0 * inter + self.eps) / (denom + self.eps)).mean()
        return bce + float(dice_weight) * dice

    def forward(self, logits: Tensor, mask: Tensor, auxiliary: dict[str, Tensor] | None = None) -> Tensor:
        loss = self.segmentation_loss(logits, mask)
        if auxiliary is None:
            return loss
        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        valid = valid.to(device=logits.device)
        target = target.to(device=logits.device)

        if self.edge_weight > 0.0 and "edge_logits" in auxiliary:
            edge_logits = auxiliary["edge_logits"]
            edge_target = lesion_edge_target(
                target.to(edge_logits.device),
                valid.to(edge_logits.device).expand_as(target),
                self.edge_band,
                self.edge_soft,
                self.edge_soft_sigma,
            )
            loss = loss + self.edge_weight * self._aux_bce_dice(
                edge_logits,
                edge_target,
                valid.to(edge_logits.device),
                self.edge_pos_weight,
                self.edge_dice_weight,
            )

        source_logits = auxiliary.get("source_logits")
        if source_logits is not None:
            source_target = source_core_target(
                target.to(source_logits.device),
                valid.to(source_logits.device).expand_as(target),
                self.source_erosion_kernel,
                self.source_soft_sigma,
            )
            if self.source_weight > 0.0:
                loss = loss + self.source_weight * self._aux_bce_dice(
                    source_logits,
                    source_target,
                    valid.to(source_logits.device),
                    self.source_pos_weight,
                    self.source_dice_weight,
                )
            if self.consistency_weight > 0.0:
                source_prob = torch.sigmoid(source_logits.detach())
                seg_prob = torch.sigmoid(logits)
                consistency = F.relu(source_prob - seg_prob)
                valid_full = valid.to(logits.device, dtype=logits.dtype).expand_as(logits)
                loss = loss + self.consistency_weight * (consistency * valid_full).sum() / valid_full.sum().clamp_min(1.0)
        return loss

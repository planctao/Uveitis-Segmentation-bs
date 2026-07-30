from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.multilabel import masks_to_paper_targets


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _logit_fp32(probability: Tensor, eps: float = 1e-6) -> Tensor:
    output_dtype = probability.dtype
    probability = probability.float().clamp(eps, 1.0 - eps)
    logits = torch.log(probability) - torch.log1p(-probability)
    return logits.to(dtype=output_dtype)


def calibrate_logits_to_area(
    logits: Tensor,
    area_fraction: Tensor,
    iterations: int = 3,
    max_shift: float = 6.0,
) -> Tensor:
    """Shift each score map so its probability mass matches an image-level burden."""
    if iterations < 0:
        raise ValueError("calibration iterations must be non-negative")
    if max_shift <= 0.0:
        raise ValueError("calibration max_shift must be positive")

    output_dtype = logits.dtype
    calibrated = logits.float()
    target = area_fraction.float().reshape(logits.shape[0], 1, 1, 1).clamp(1e-7, 1.0 - 1e-5)
    target_logit = _logit_fp32(target)
    for _ in range(iterations):
        current = torch.sigmoid(calibrated).mean(dim=(-2, -1), keepdim=True)
        shift = (target_logit - _logit_fp32(current)).clamp(-max_shift, max_shift)
        calibrated = calibrated + shift
    return calibrated.to(dtype=output_dtype)


class SparseEvidencePool(nn.Module):
    """Combine global context with the strongest sparse responses."""

    def __init__(self, fraction: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError("top-k fraction must be in (0, 1]")
        self.fraction = float(fraction)

    def forward(self, features: Tensor) -> Tensor:
        flat = features.flatten(2)
        k = max(1, int(math.ceil(flat.shape[-1] * self.fraction)))
        return torch.cat([flat.mean(dim=-1), flat.topk(k, dim=-1).values.mean(dim=-1)], dim=1)


class ZABLeakageHead(nn.Module):
    """Zero-inflated anatomy- and burden-aware head for rare macular leakage."""

    def __init__(
        self,
        in_channels: int,
        global_channels: int | None = None,
        topk_fraction: float = 0.05,
        presence_prior: float = 0.11,
        area_prior: float = 0.005,
        max_area_fraction: float = 0.1,
        anatomy_strength: float = 0.75,
        hierarchy_strength: float = 0.0,
        bidirectional_strength: float = 0.0,
        calibration_iterations: int = 3,
        calibration_max_shift: float = 6.0,
    ) -> None:
        super().__init__()
        if not 0.0 < presence_prior < 1.0:
            raise ValueError("presence_prior must be in (0, 1)")
        if not 0.0 < area_prior < max_area_fraction < 1.0:
            raise ValueError("area_prior and max_area_fraction must satisfy 0 < prior < max < 1")
        if anatomy_strength < 0.0:
            raise ValueError("anatomy_strength must be non-negative")
        if hierarchy_strength < 0.0:
            raise ValueError("hierarchy_strength must be non-negative")
        if not 0.0 <= bidirectional_strength <= 1.0:
            raise ValueError("bidirectional_strength must be in [0, 1]")

        groups = _group_count(in_channels)
        self.retinal_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.anatomy_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self.macular_context = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
        )
        self.macular_rank_head = nn.Conv2d(in_channels, 1, kernel_size=1)

        global_channels = int(global_channels or in_channels)
        hidden = max(32, global_channels // 2)
        self.evidence_pool = SparseEvidencePool(topk_fraction)
        self.global_neck = nn.Sequential(
            nn.Linear(2 * global_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.presence_head = nn.Linear(hidden, 1)
        self.area_head = nn.Linear(hidden, 1)

        self.max_area_fraction = float(max_area_fraction)
        self.anatomy_strength = float(anatomy_strength)
        self.hierarchy_strength = float(hierarchy_strength)
        self.bidirectional_strength = float(bidirectional_strength)
        self.calibration_iterations = int(calibration_iterations)
        self.calibration_max_shift = float(calibration_max_shift)

        nn.init.zeros_(self.anatomy_head[-1].weight)
        nn.init.zeros_(self.anatomy_head[-1].bias)
        nn.init.constant_(self.presence_head.bias, math.log(presence_prior / (1.0 - presence_prior)))
        area_ratio = area_prior / max_area_fraction
        nn.init.constant_(self.area_head.bias, math.log(area_ratio / (1.0 - area_ratio)))

    def forward(self, features: Tensor, global_features: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        retinal_logits = self.retinal_head(features)
        retinal_probability = torch.sigmoid(retinal_logits)

        anatomy_logits = self.anatomy_head(features)
        anatomy_probability = torch.sigmoid(anatomy_logits)
        context = self.macular_context(
            torch.cat([features, retinal_probability, anatomy_probability], dim=1)
        )
        rank_logits = self.macular_rank_head(context)
        anatomy_evidence = 4.0 * torch.tanh(_logit_fp32(anatomy_probability) / 4.0)
        # Macular leakage is almost always nested in retinal leakage in this
        # dataset. Only negative retinal evidence is used, so the hierarchy can
        # suppress implausible outside responses without amplifying easy pixels.
        hierarchy_evidence = retinal_logits.float().clamp(min=-4.0, max=0.0).to(retinal_logits.dtype)
        spatial_logits = (
            rank_logits
            + self.anatomy_strength * anatomy_evidence
            + self.hierarchy_strength * hierarchy_evidence
        )

        pooled_source = global_features if global_features is not None else features
        global_embedding = self.global_neck(self.evidence_pool(pooled_source))
        presence_logits = self.presence_head(global_embedding)
        conditional_area_fraction = self.max_area_fraction * torch.sigmoid(self.area_head(global_embedding))
        expected_area_fraction = torch.sigmoid(presence_logits) * conditional_area_fraction
        macular_logits = calibrate_logits_to_area(
            spatial_logits,
            expected_area_fraction,
            iterations=self.calibration_iterations,
            max_shift=self.calibration_max_shift,
        )

        if self.bidirectional_strength > 0.0:
            # The two paper targets overlap heavily. Let confident macular
            # evidence recover a small amount of retinal probability, while
            # retaining a bounded exception instead of forcing equality.
            retinal_probability = torch.sigmoid(retinal_logits.float())
            macular_probability = torch.sigmoid(macular_logits.float())
            union_probability = 1.0 - (1.0 - retinal_probability) * (
                1.0 - self.bidirectional_strength * macular_probability
            )
            retinal_logits = _logit_fp32(union_probability).to(dtype=retinal_logits.dtype)

        logits = torch.cat([retinal_logits, macular_logits], dim=1)
        auxiliary = {
            "anatomy_logits": anatomy_logits,
            "rank_logits": rank_logits,
            "hierarchy_evidence": hierarchy_evidence,
            "presence_logits": presence_logits,
            "conditional_area_fraction": conditional_area_fraction,
            "expected_area_fraction": expected_area_fraction,
        }
        return logits, auxiliary


class ZABLoss(nn.Module):
    """Segmentation plus weak anatomy and zero-inflated burden supervision."""

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        presence_weight: float = 0.15,
        area_weight: float = 0.1,
        mass_weight: float = 0.05,
        anatomy_weight: float = 0.1,
        hierarchy_weight: float = 0.0,
        presence_pos_weight: float = 8.0,
        min_confidence_pixels: int = 64,
        anatomy_min_pixels: int = 64,
        anatomy_sigma: float = 0.12,
        anatomy_max_sigma: float = 0.25,
        mass_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if anatomy_sigma <= 0.0 or anatomy_max_sigma < anatomy_sigma:
            raise ValueError("anatomy sigmas must satisfy 0 < sigma <= max_sigma")
        if mass_scale <= 0.0:
            raise ValueError("mass_scale must be positive")
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.presence_weight = float(presence_weight)
        self.area_weight = float(area_weight)
        self.mass_weight = float(mass_weight)
        self.anatomy_weight = float(anatomy_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.presence_pos_weight = float(presence_pos_weight)
        self.min_confidence_pixels = max(1, int(min_confidence_pixels))
        self.anatomy_min_pixels = max(1, int(anatomy_min_pixels))
        self.anatomy_sigma = float(anatomy_sigma)
        self.anatomy_max_sigma = float(anatomy_max_sigma)
        self.mass_scale = float(mass_scale)

    def _anatomy_targets(
        self,
        macular: Tensor,
        valid: Tensor,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        batch, source_height, source_width = macular.shape
        out_height, out_width = output_size
        grid_y = torch.linspace(0.0, 1.0, out_height, device=macular.device, dtype=torch.float32)
        grid_x = torch.linspace(0.0, 1.0, out_width, device=macular.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        targets = torch.zeros(batch, 1, out_height, out_width, device=macular.device, dtype=torch.float32)
        selected = torch.zeros(batch, device=macular.device, dtype=torch.bool)

        for batch_index in range(batch):
            pixels = torch.nonzero(macular[batch_index] & valid[batch_index], as_tuple=False)
            if pixels.shape[0] < self.anatomy_min_pixels:
                continue
            selected[batch_index] = True
            y = pixels[:, 0].float() / max(source_height - 1, 1)
            x = pixels[:, 1].float() / max(source_width - 1, 1)
            center_y = y.mean()
            center_x = x.mean()
            sigma_y = torch.clamp(1.5 * y.std(unbiased=False), self.anatomy_sigma, self.anatomy_max_sigma)
            sigma_x = torch.clamp(1.5 * x.std(unbiased=False), self.anatomy_sigma, self.anatomy_max_sigma)
            targets[batch_index, 0] = torch.exp(
                -0.5 * (((yy - center_y) / sigma_y) ** 2 + ((xx - center_x) / sigma_x) ** 2)
            )
        return targets, selected

    def forward(
        self,
        logits: Tensor,
        mask: Tensor,
        auxiliary: dict[str, Tensor] | None = None,
    ) -> Tensor:
        segmentation = self.segmentation_loss(logits, mask)
        if auxiliary is None:
            return segmentation

        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        macular = target[:, 1].to(device=logits.device, dtype=torch.bool)
        valid_pixels = valid[:, 0].to(device=logits.device, dtype=torch.bool)
        valid_count = valid_pixels.flatten(1).sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        area_pixels = (macular & valid_pixels).flatten(1).sum(dim=1).to(dtype=torch.float32)
        area_fraction = area_pixels / valid_count
        presence_target = (area_pixels > 0).to(dtype=torch.float32)
        positive_confidence = (area_pixels / float(self.min_confidence_pixels)).clamp(0.1, 1.0)
        sample_weight = torch.where(presence_target > 0.5, positive_confidence, torch.ones_like(positive_confidence))

        presence_logits = auxiliary["presence_logits"][:, 0].float()
        presence_bce = F.binary_cross_entropy_with_logits(
            presence_logits,
            presence_target,
            pos_weight=presence_logits.new_tensor(self.presence_pos_weight),
            reduction="none",
        )
        presence_loss = (presence_bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        positive = presence_target > 0.5
        conditional_area = auxiliary["conditional_area_fraction"][:, 0].float().clamp_min(1e-7)
        if bool(positive.any()):
            area_error = F.smooth_l1_loss(
                torch.log(conditional_area[positive]),
                torch.log(area_fraction[positive].clamp_min(1e-7)),
                reduction="none",
            )
            positive_weight = sample_weight[positive]
            area_loss = (area_error * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        else:
            area_loss = logits.sum() * 0.0

        probability = torch.sigmoid(logits[:, 1].float())
        predicted_fraction = (probability * valid_pixels).flatten(1).sum(dim=1) / valid_count
        predicted_burden = torch.log1p(predicted_fraction / self.mass_scale)
        target_burden = torch.log1p(area_fraction / self.mass_scale)
        mass_loss = F.smooth_l1_loss(predicted_burden, target_burden)

        retinal_probability = torch.sigmoid(logits[:, 0].float())
        hierarchy_exempt = macular & ~target[:, 0].to(device=logits.device, dtype=torch.bool)
        hierarchy_valid = valid_pixels & ~hierarchy_exempt
        hierarchy_violation = F.relu(probability - retinal_probability)
        if bool(hierarchy_valid.any()):
            hierarchy_loss = hierarchy_violation[hierarchy_valid].mean()
        else:
            hierarchy_loss = logits.sum() * 0.0

        anatomy_logits = auxiliary["anatomy_logits"]
        anatomy_target, anatomy_selected = self._anatomy_targets(
            macular,
            valid_pixels,
            tuple(anatomy_logits.shape[-2:]),
        )
        if bool(anatomy_selected.any()):
            anatomy_target = anatomy_target.to(device=anatomy_logits.device, dtype=anatomy_logits.dtype)
            selected_logits = anatomy_logits[anatomy_selected]
            selected_target = anatomy_target[anatomy_selected]
            anatomy_bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target)
            anatomy_probability = torch.sigmoid(selected_logits)
            dims = (1, 2, 3)
            intersection = (anatomy_probability * selected_target).sum(dim=dims)
            denominator = anatomy_probability.sum(dim=dims) + selected_target.sum(dim=dims)
            anatomy_dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
            anatomy_loss = anatomy_bce + anatomy_dice
        else:
            anatomy_loss = logits.sum() * 0.0

        return (
            segmentation
            + self.presence_weight * presence_loss
            + self.area_weight * area_loss
            + self.mass_weight * mass_loss
            + self.anatomy_weight * anatomy_loss
            + self.hierarchy_weight * hierarchy_loss
        )

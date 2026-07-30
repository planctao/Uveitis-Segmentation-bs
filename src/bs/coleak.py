from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TopKPresencePool(nn.Module):
    """Retain sparse lesion evidence instead of diluting it with global average pooling."""

    def __init__(self, fraction: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError("top-k fraction must be in (0, 1]")
        self.fraction = float(fraction)

    def forward(self, features: Tensor) -> Tensor:
        flat = features.flatten(2)
        k = max(1, int(math.ceil(flat.shape[-1] * self.fraction)))
        return flat.topk(k, dim=-1).values.mean(dim=-1)


class CoupledLeakageHead(nn.Module):
    """Conditional four-state head for retinal and macular leakage.

    The head models the original palette states through three probabilities:
    retinal leakage, macular leakage inside retinal leakage, and the uncommon
    macular-only exception. Their marginal recovers the two paper targets.
    """

    def __init__(
        self,
        in_channels: int,
        global_channels: int | None = None,
        topk_fraction: float = 0.05,
        presence_prior: float = 0.1,
        prior_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < presence_prior < 1.0:
            raise ValueError("presence_prior must be in (0, 1)")
        if prior_strength < 0.0:
            raise ValueError("prior_strength must be non-negative")

        groups = _group_count(in_channels)
        self.retinal_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.child_context = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
        )
        self.inside_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.outside_delta = nn.Conv2d(in_channels, 1, kernel_size=1)

        global_channels = int(global_channels or in_channels)
        hidden = max(16, global_channels // 4)
        self.presence_pool = TopKPresencePool(topk_fraction)
        self.presence_head = nn.Sequential(
            nn.Linear(global_channels, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        self.prior_strength = float(prior_strength)

        nn.init.zeros_(self.outside_delta.weight)
        nn.init.zeros_(self.outside_delta.bias)
        nn.init.constant_(self.presence_head[-1].bias, math.log(presence_prior / (1.0 - presence_prior)))

    @staticmethod
    def _logit(probability: Tensor, eps: float = 1e-5) -> Tensor:
        output_dtype = probability.dtype
        probability = probability.float().clamp(eps, 1.0 - eps)
        logits = torch.log(probability) - torch.log1p(-probability)
        return logits.to(dtype=output_dtype)

    def forward(self, features: Tensor, global_features: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        retinal_logits = self.retinal_head(features)
        retinal_probability = torch.sigmoid(retinal_logits)

        context = self.child_context(torch.cat([features, retinal_probability], dim=1))
        inside_logits = self.inside_head(context)
        outside_logits = inside_logits + self.outside_delta(context)
        inside_probability = torch.sigmoid(inside_logits)
        outside_probability = torch.sigmoid(outside_logits)

        macular_probability = (
            retinal_probability * inside_probability
            + (1.0 - retinal_probability) * outside_probability
        )
        presence_features = global_features if global_features is not None else features
        presence_logits = self.presence_head(self.presence_pool(presence_features))

        # A bounded log-odds prior suppresses negative images without hard-zeroing
        # small lesions when the image-level classifier is uncertain.
        bounded_presence = 4.0 * torch.tanh(presence_logits / 4.0)
        macular_logits = self._logit(macular_probability)
        macular_logits = macular_logits + self.prior_strength * bounded_presence[:, :, None, None]
        logits = torch.cat([retinal_logits, macular_logits], dim=1)
        auxiliary = {
            "inside_logits": inside_logits,
            "outside_logits": outside_logits,
            "presence_logits": presence_logits,
            "retinal_probability": retinal_probability,
        }
        return logits, auxiliary


class CoLeakLoss(nn.Module):
    """Paper segmentation loss plus conditional-state and presence supervision."""

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        inside_weight: float = 0.25,
        outside_weight: float = 0.1,
        presence_weight: float = 0.15,
        state_pos_weight: float = 4.0,
        presence_pos_weight: float = 8.0,
        hard_negative_ratio: float = 8.0,
        hard_negative_min_pixels: int = 256,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.inside_weight = float(inside_weight)
        self.outside_weight = float(outside_weight)
        self.presence_weight = float(presence_weight)
        self.state_pos_weight = float(state_pos_weight)
        self.presence_pos_weight = float(presence_pos_weight)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.hard_negative_min_pixels = int(hard_negative_min_pixels)

    def _resize_mask(self, mask: Tensor, size: tuple[int, int]) -> Tensor:
        if tuple(mask.shape[-2:]) == size:
            return mask
        return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].long()

    def _hard_binary_loss(self, logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        selected_losses: list[Tensor] = []
        for batch_index in range(logits.shape[0]):
            sample_valid = valid[batch_index].flatten()
            sample_target = target[batch_index].flatten()
            sample_losses = losses[batch_index].flatten()
            positive = sample_valid & (sample_target > 0.5)
            negative = sample_valid & ~positive

            positive_losses = sample_losses[positive] * self.state_pos_weight
            negative_losses = sample_losses[negative]
            if negative_losses.numel() > 0:
                positive_count = int(positive.sum().item())
                negative_count = max(
                    self.hard_negative_min_pixels,
                    int(math.ceil(positive_count * self.hard_negative_ratio)),
                )
                negative_count = min(negative_count, int(negative_losses.numel()))
                negative_losses = negative_losses.topk(negative_count).values
            if positive_losses.numel() or negative_losses.numel():
                selected_losses.append(torch.cat([positive_losses, negative_losses]).mean())
        if not selected_losses:
            return logits.sum() * 0.0
        return torch.stack(selected_losses).mean()

    def forward(
        self,
        logits: Tensor,
        mask: Tensor,
        auxiliary: dict[str, Tensor] | None = None,
    ) -> Tensor:
        loss = self.segmentation_loss(logits, mask)
        if auxiliary is None:
            return loss

        inside_logits = auxiliary["inside_logits"]
        outside_logits = auxiliary["outside_logits"]
        state_mask = self._resize_mask(mask, tuple(inside_logits.shape[-2:]))
        valid = state_mask != self.ignore_index
        retinal = (state_mask == 1) | (state_mask == 3)
        macular = (state_mask == 2) | (state_mask == 3)

        inside_loss = self._hard_binary_loss(
            inside_logits[:, 0],
            macular.to(dtype=inside_logits.dtype),
            valid & retinal,
        )
        outside_loss = self._hard_binary_loss(
            outside_logits[:, 0],
            macular.to(dtype=outside_logits.dtype),
            valid & ~retinal,
        )

        presence_logits = auxiliary["presence_logits"][:, 0]
        full_valid = mask != self.ignore_index
        full_macular = (mask == 2) | (mask == 3)
        presence_target = (full_macular & full_valid).flatten(1).any(dim=1).to(dtype=presence_logits.dtype)
        presence_weight = presence_logits.new_tensor(self.presence_pos_weight)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits,
            presence_target,
            pos_weight=presence_weight,
        )
        return (
            loss
            + self.inside_weight * inside_loss
            + self.outside_weight * outside_loss
            + self.presence_weight * presence_loss
        )

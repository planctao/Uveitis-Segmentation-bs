from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def masks_to_paper_targets(mask: Tensor, ignore_index: int = 255) -> tuple[Tensor, Tensor]:
    valid = mask != ignore_index
    lesion_1 = ((mask == 1) | (mask == 3)).float()
    lesion_2 = ((mask == 2) | (mask == 3)).float()
    target = torch.stack([lesion_1, lesion_2], dim=1)
    return target, valid.unsqueeze(1)


def _threshold_tensor(threshold: float | list[float] | tuple[float, ...], dtype: torch.dtype = torch.float32) -> Tensor:
    values = torch.as_tensor(threshold, dtype=dtype)
    if values.numel() == 1:
        values = values.repeat(2)
    if values.numel() != 2:
        raise ValueError(f"Expected one threshold or two per-lesion thresholds, got {threshold}")
    return values.view(1, 2, 1, 1)


class PaperDice:
    def __init__(self, ignore_index: int = 255, threshold: float | list[float] | tuple[float, ...] = 0.5) -> None:
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.intersections = torch.zeros(2, dtype=torch.float64)
        self.predicted = torch.zeros(2, dtype=torch.float64)
        self.targets = torch.zeros(2, dtype=torch.float64)

    def update(self, logits: Tensor, mask: Tensor) -> None:
        target, valid = masks_to_paper_targets(mask.detach().cpu(), self.ignore_index)
        probs = torch.sigmoid(logits.detach().cpu())
        thresholds = _threshold_tensor(self.threshold, probs.dtype)
        pred = probs >= thresholds
        valid = valid.expand_as(target)
        pred = pred & valid
        target = target.bool() & valid
        dims = (0, 2, 3)
        self.intersections += (pred & target).sum(dim=dims).to(torch.float64)
        self.predicted += pred.sum(dim=dims).to(torch.float64)
        self.targets += target.sum(dim=dims).to(torch.float64)

    def compute(self) -> dict[str, float]:
        dice = (2.0 * self.intersections / (self.predicted + self.targets).clamp_min(1.0)).nan_to_num(0.0)
        return {
            "paper_dice_1": float(dice[0].item()),
            "paper_dice_2": float(dice[1].item()),
            "paper_macro_dice": float(dice.mean().item()),
            "paper_pred_pixels_1": float(self.predicted[0].item()),
            "paper_pred_pixels_2": float(self.predicted[1].item()),
            "paper_target_pixels_1": float(self.targets[0].item()),
            "paper_target_pixels_2": float(self.targets[1].item()),
        }


class AsymmetricFocalTverskyBCE(nn.Module):
    def __init__(
        self,
        pos_weight: list[float] | tuple[float, float] = (1.0, 20.0),
        bce_weight: float = 0.5,
        tversky_weight: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        ignore_index: int = 255,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        valid = valid.to(device=logits.device, dtype=logits.dtype).expand_as(logits)
        target = target.to(device=logits.device, dtype=logits.dtype)
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

        probs = torch.sigmoid(logits)
        probs = probs * valid
        target = target * valid
        dims = (0, 2, 3)
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target) * valid).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        return self.bce_weight * bce + self.tversky_weight * focal_tversky

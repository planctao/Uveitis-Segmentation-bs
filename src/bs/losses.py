from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: list[float] | None = None,
        dice_include_background: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_include_background = dice_include_background
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        weight = self.class_weights.to(logits.device, dtype=logits.dtype) if self.class_weights is not None else None
        ce_loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, weight=weight)
        dice_loss = multiclass_dice_loss(
            logits,
            target,
            self.num_classes,
            self.ignore_index,
            include_background=self.dice_include_background,
        )
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def multiclass_dice_loss(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    include_background: bool = False,
) -> Tensor:
    valid = target != ignore_index
    target_safe = target.masked_fill(~valid, 0)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    valid = valid.unsqueeze(1)
    probs = probs * valid
    one_hot = one_hot * valid
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    if not include_background and dice.numel() > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()

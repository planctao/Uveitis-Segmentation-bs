from __future__ import annotations

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    def update(self, logits: Tensor, target: Tensor) -> None:
        pred = torch.argmax(logits.detach(), dim=1).cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        pred = pred[valid].view(-1)
        target = target[valid].view(-1)
        if target.numel() == 0:
            return
        encoded = target * self.num_classes + pred
        bincount = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += bincount.reshape(self.num_classes, self.num_classes).to(torch.float64)

    def compute(self) -> dict[str, float]:
        tp = torch.diag(self.confusion)
        support = self.confusion.sum(dim=1)
        predicted = self.confusion.sum(dim=0)
        union = support + predicted - tp
        iou = (tp / union.clamp_min(1.0)).nan_to_num(0.0)
        dice = ((2.0 * tp) / (support + predicted).clamp_min(1.0)).nan_to_num(0.0)
        accuracy = tp.sum() / self.confusion.sum().clamp_min(1.0)
        foreground = torch.arange(self.num_classes) > 0
        return {
            "pixel_acc": float(accuracy.item()),
            "mean_iou": float(iou.mean().item()),
            "mean_dice": float(dice.mean().item()),
            "fg_mean_iou": float(iou[foreground].mean().item()) if foreground.any() else float(iou.mean().item()),
            "fg_mean_dice": float(dice[foreground].mean().item()) if foreground.any() else float(dice.mean().item()),
            **{f"iou_class_{idx}": float(value.item()) for idx, value in enumerate(iou)},
            **{f"dice_class_{idx}": float(value.item()) for idx, value in enumerate(dice)},
        }


class PaperDiceMetrics:
    """Two-lesion Dice used by the baseline paper.

    Masks are stored as integer labels where value 1 marks lesion 1, value 2 marks
    lesion 2, and value 3 marks pixels belonging to both lesion masks.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = ignore_index
        self.intersections = torch.zeros(2, dtype=torch.float64)
        self.predicted = torch.zeros(2, dtype=torch.float64)
        self.targets = torch.zeros(2, dtype=torch.float64)

    def update(self, logits: Tensor, target: Tensor) -> None:
        pred_label = torch.argmax(logits.detach(), dim=1).cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index

        pred_lesion_1 = (pred_label == 1) | (pred_label == 3)
        pred_lesion_2 = (pred_label == 2) | (pred_label == 3)
        target_lesion_1 = (target == 1) | (target == 3)
        target_lesion_2 = (target == 2) | (target == 3)

        for idx, (pred_mask, target_mask) in enumerate(
            (
                (pred_lesion_1, target_lesion_1),
                (pred_lesion_2, target_lesion_2),
            )
        ):
            pred_valid = pred_mask[valid]
            target_valid = target_mask[valid]
            self.intersections[idx] += (pred_valid & target_valid).sum().to(torch.float64)
            self.predicted[idx] += pred_valid.sum().to(torch.float64)
            self.targets[idx] += target_valid.sum().to(torch.float64)

    def compute(self) -> dict[str, float]:
        denominator = self.predicted + self.targets
        dice = ((2.0 * self.intersections) / denominator.clamp_min(1.0)).nan_to_num(0.0)
        return {
            "paper_dice_1": float(dice[0].item()),
            "paper_dice_2": float(dice[1].item()),
            "paper_macro_dice": float(dice.mean().item()),
        }

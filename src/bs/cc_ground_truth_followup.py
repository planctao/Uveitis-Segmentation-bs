"""Independent GT-region follow-up with a matched whole-image DiceCE control.

The original BiCC experiment, source and admission rule remain separate.
Both arms add the same auxiliary coefficient to the unchanged FA objective.
"""

import torch
from torch import Tensor
import torch.nn.functional as F

from bs.bidirectional_component_unet import BidirectionalComponentUNet2D


def whole_image_dicece(logits: Tensor, foreground: Tensor, valid: Tensor) -> Tensor:
    """BiCC's zero-component fallback, applied to every image independently."""
    if logits.ndim != 3 or foreground.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("Require aligned BHW tensors")
    logits = logits.float()
    valid = valid.bool()
    y = foreground.bool() & valid
    p = logits.sigmoid() * valid
    dims = (1, 2)
    dice = 1 - 2 * (p * y).sum(dims) / (p.sum(dims) + y.sum(dims)).clamp_min(1e-8)
    ce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    per_image = dice + (ce * valid).sum(dims) / valid.sum(dims).clamp_min(1)
    per_image = torch.where(valid.flatten(1).any(1), per_image, logits.sum(dims) * 0)
    return per_image.mean()


class CCGroundTruthFollowupUNet(BidirectionalComponentUNet2D):
    def __init__(self, *args, auxiliary_mode="gt_only", **kwargs):
        if auxiliary_mode not in {"gt_only", "global_only"}:
            raise ValueError("Unknown independent CC follow-up mode")
        super().__init__(*args, mode="gt_only", **kwargs)
        self.auxiliary_mode = auxiliary_mode

    def training_loss(self, base_loss, images, masks, criterion):
        if self.auxiliary_mode == "gt_only":
            return super().training_loss(base_loss, images, masks, criterion)
        if not self.training or self._cached_logits is None:
            raise RuntimeError("Require the current training forward")
        term = whole_image_dicece(self._cached_logits[:, 1], (masks == 2) | (masks == 3),
                                 masks != self.ignore_index)
        auxiliary = self.weight * term
        self._batches += 1
        self._sums["instance"] += float(term.detach())
        self._sums["weighted_auxiliary"] += float(auxiliary.detach())
        self._cached_logits = None
        return base_loss + auxiliary

    def instance_statistics(self):
        result = super().instance_statistics()
        result.update(mode=self.auxiliary_mode,
                      experiment="independent_gt_region_vs_whole_image_followup")
        if self.auxiliary_mode == "global_only":
            result["partition_geometry"] = "whole valid image, no component partition"
        return result

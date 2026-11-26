"""BiCC component objective around the unchanged two-output FA nnU-Net."""

from typing import Sequence

from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d
from bs.bidirectional_component_supervision import bidirectional_component_terms


class BidirectionalComponentUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 mode: str = "bidirectional", weight: float = 1., ignore_index: int = 255):
        super().__init__()
        if num_classes != 2 or mode not in {"bidirectional", "gt_only", "smoothed_pred"} or weight < 0:
            raise ValueError("Require two independent outputs, a valid BiCC mode and nonnegative weight")
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        self.mode, self.weight, self.ignore_index = mode, float(weight), int(ignore_index)
        self._cached_logits = None
        self.reset_instance_statistics()

    def reset_instance_statistics(self):
        self._batches = 0
        self._counts = {"gt_components": 0, "pred_components": 0, "GT_empty_pred_cells": 0}
        self._sums = {"gt": 0., "pred": 0., "instance": 0., "weighted_auxiliary": 0.}

    def instance_statistics(self):
        return {"batches": self._batches, **self._counts,
                "mean_terms": {k: v / max(self._batches, 1) for k, v in self._sums.items()},
                "mode": self.mode, "weight": self.weight, "connectivity": "2D8_neighbor",
                "partition_geometry": "isotropic augmented pixel coordinates",
                "prediction_partition_threshold": .5, "auxiliary_target": "lesion2=(palette2 or palette3)"}

    def forward(self, images: Tensor) -> Tensor:
        logits = self.decoder(self.encoder(images))
        self._cached_logits = logits if self.training else None
        return logits

    def training_loss(self, base_loss: Tensor, images: Tensor, masks: Tensor, criterion) -> Tensor:
        if not self.training or self._cached_logits is None:
            raise RuntimeError("BiCC objective requires the current training forward")
        terms = bidirectional_component_terms(self._cached_logits[:, 1], (masks == 2) | (masks == 3),
            valid_mask=masks != self.ignore_index, alpha=0. if self.mode == "gt_only" else .5,
            pred_empty_cell_loss="dice" if self.mode == "smoothed_pred" else "detached",
            pred_smooth=1e-5 if self.mode == "smoothed_pred" else 0.)
        auxiliary = self.weight * terms["instance"]
        self._batches += 1
        for key in self._counts:
            self._counts[key] += terms[key]
        for key in ["gt", "pred", "instance"]:
            self._sums[key] += float(terms[key].detach())
        self._sums["weighted_auxiliary"] += float(auxiliary.detach())
        self._cached_logits = None
        return base_loss + auxiliary

"""Lesion2-only CATMIL transfer retaining the complete original FA nnU-Net."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d
from bs.component_supervision import component_detection_terms


class ComponentSupervisedUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 mode: str = "joint", cat_weight: float = .1, mil_weight: float = .2,
                 cat_warmup_epochs: int = 10, ignore_index: int = 255):
        super().__init__()
        if mode not in {"joint", "cat", "mil"} or num_classes != 2:
            raise ValueError("Require two sigmoid channels and joint/cat/mil mode")
        if min(cat_weight, mil_weight, cat_warmup_epochs) < 0:
            raise ValueError("Weights and warmup must be nonnegative")
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        self.mode, self.cat_weight, self.mil_weight = mode, float(cat_weight), float(mil_weight)
        self.cat_warmup_epochs, self.ignore_index = int(cat_warmup_epochs), int(ignore_index)
        self.epoch = 1
        self._cached_logits = None
        self.reset_component_statistics()

    def reset_component_statistics(self):
        self._batches, self._positive_batches, self._components = 0, 0, 0
        self._sums = {"cat": 0., "mil": 0., "weighted_auxiliary": 0.}

    @property
    def current_cat_weight(self):
        ramp = min(max(self.epoch - 1, 0) / self.cat_warmup_epochs, 1.) if self.cat_warmup_epochs else 1.
        return self.cat_weight * ramp

    def component_statistics(self):
        return {"batches": self._batches, "batches_with_lesion2": self._positive_batches,
                "lesion2_components": self._components,
                "mean_terms": {key: value / max(self._batches, 1) for key, value in self._sums.items()},
                "mode": self.mode, "cat_weight": self.current_cat_weight, "mil_weight": self.mil_weight,
                "cat_warmup_epochs": self.cat_warmup_epochs, "connectivity": "2D4_neighbor",
                "auxiliary_target": "lesion2=(palette2 or palette3)", "full_resolution": True}

    def forward(self, images: Tensor) -> Tensor:
        logits = self.decoder(self.encoder(images))
        self._cached_logits = logits if self.training else None
        return logits

    def training_loss(self, base_loss: Tensor, images: Tensor, masks: Tensor, criterion) -> Tensor:
        if not self.training or self._cached_logits is None:
            raise RuntimeError("Component objective requires the current training forward")
        terms = component_detection_terms(self._cached_logits[:, 1].float().sigmoid(),
            (masks == 2) | (masks == 3), valid_mask=masks != self.ignore_index)
        auxiliary = base_loss * 0.
        if self.mode in {"joint", "cat"}:
            auxiliary = auxiliary + self.current_cat_weight * terms["cat"]
        if self.mode in {"joint", "mil"}:
            auxiliary = auxiliary + self.mil_weight * terms["mil"]
        self._batches += 1
        self._positive_batches += int(terms["component_count"] > 0)
        self._components += terms["component_count"]
        for key in ["cat", "mil"]:
            self._sums[key] += float(terms[key].detach())
        self._sums["weighted_auxiliary"] += float(auxiliary.detach())
        self._cached_logits = None
        return base_loss + auxiliary

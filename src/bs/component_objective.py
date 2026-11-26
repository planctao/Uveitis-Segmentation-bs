"""Training-only CATMIL objective around the unchanged image-aware FA loss."""

from __future__ import annotations

from torch import Tensor, nn

from bs.component_supervision import component_detection_terms


class TrainingComponentObjective(nn.Module):
    def __init__(self, base: nn.Module, *, mode: str = "cat", cat_weight: float = .1,
                 mil_weight: float = .2, cat_warmup_epochs: int = 10, ignore_index: int = 255):
        super().__init__()
        if mode not in {"cat", "mil", "joint"} or min(cat_weight, mil_weight, cat_warmup_epochs) < 0:
            raise ValueError("Require cat/mil/joint and nonnegative weights/warmup")
        self.base = base
        self.mode = mode
        self.cat_weight, self.mil_weight = float(cat_weight), float(mil_weight)
        self.cat_warmup_epochs, self.ignore_index = int(cat_warmup_epochs), int(ignore_index)
        self.epoch, self.enabled = 1, False
        self.reset_statistics()

    @property
    def needs_image(self):
        return bool(getattr(self.base, "needs_image", False))

    @property
    def current_cat_weight(self):
        ramp = min(max(self.epoch - 1, 0) / self.cat_warmup_epochs, 1.) if self.cat_warmup_epochs else 1.
        return self.cat_weight * ramp

    def reset_statistics(self):
        self._batches, self._positive_batches, self._components = 0, 0, 0
        self._sums = {"cat": 0., "mil": 0., "weighted_auxiliary": 0.}

    def statistics(self):
        return {"batches": self._batches, "batches_with_lesion2": self._positive_batches,
                "lesion2_components": self._components,
                "mean_terms": {name: value / max(self._batches, 1) for name, value in self._sums.items()},
                "mode": self.mode, "cat_weight": self.current_cat_weight, "mil_weight": self.mil_weight,
                "cat_warmup_epochs": self.cat_warmup_epochs, "connectivity": "2D4_neighbor",
                "auxiliary_target": "lesion2=(palette2 or palette3)", "full_resolution": True}

    def forward(self, logits: Tensor, masks: Tensor, image: Tensor | None = None) -> Tensor:
        loss = self.base(logits, masks, image=image) if self.needs_image else self.base(logits, masks)
        if not self.enabled:
            return loss
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("Require two independent FA sigmoid channels")
        terms = component_detection_terms(logits[:, 1].float().sigmoid(),
                                          (masks == 2) | (masks == 3), valid_mask=masks != self.ignore_index)
        auxiliary = loss * 0.
        if self.mode in {"joint", "cat"}:
            auxiliary = auxiliary + self.current_cat_weight * terms["cat"]
        if self.mode in {"joint", "mil"}:
            auxiliary = auxiliary + self.mil_weight * terms["mil"]
        self._batches += 1
        self._positive_batches += int(terms["component_count"] > 0)
        self._components += terms["component_count"]
        for name in ["cat", "mil"]:
            self._sums[name] += float(terms[name].detach())
        self._sums["weighted_auxiliary"] += float(auxiliary.detach())
        return loss + auxiliary

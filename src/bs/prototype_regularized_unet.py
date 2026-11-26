"""Teacher-free transfer of the SWARD PDR feature objective to FA nnU-Net.

The printed PDR equations are evaluated on full-resolution prelogit decoder
features to preserve tiny labels. This location, optional unit normalization,
and four palette-state prototypes are explicitly declared FA adaptations.
The network and its two sigmoid outputs are otherwise exactly the baseline.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d
from bs.prototype_regularization import prototype_discriminative_terms


class PrototypeRegularizedUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 mode: str = "joint", weight: float = 1., margin: float = 1.3,
                 normalize_features: bool = True, ignore_index: int = 255):
        super().__init__()
        if mode not in {"joint", "separation", "compactness"}:
            raise ValueError(f"Unknown PDR component mode: {mode}")
        if weight < 0:
            raise ValueError("PDR weight must be nonnegative")
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        self.mode, self.weight, self.margin = mode, float(weight), float(margin)
        self.normalize_features, self.ignore_index = bool(normalize_features), int(ignore_index)
        self._cached_feature = None
        self.decoder.seg_layers[-1].register_forward_pre_hook(self._capture_feature)
        self.reset_pdr_statistics()

    def _capture_feature(self, module, args):
        if self.training:
            self._cached_feature = args[0]

    def reset_pdr_statistics(self):
        self._pdr_batches = 0
        self._pdr_sums = {"separation": 0., "compactness": 0., "weighted_auxiliary": 0.}
        self._pdr_class_batches = {str(label): 0 for label in range(4)}

    def pdr_statistics(self):
        return {"batches": self._pdr_batches,
                "mean_terms": {key: value / max(self._pdr_batches, 1) for key, value in self._pdr_sums.items()},
                "batches_with_eligible_palette_class": dict(self._pdr_class_batches),
                "feature_location": "full_resolution_decoder_prelogit", "mode": self.mode,
                "weight": self.weight, "margin": self.margin,
                "normalize_features": self.normalize_features}

    def forward(self, images: Tensor) -> Tensor:
        self._cached_feature = None
        return self.decoder(self.encoder(images))

    def training_loss(self, base_loss: Tensor, images: Tensor, masks: Tensor, criterion) -> Tensor:
        if not self.training or self._cached_feature is None:
            raise RuntimeError("PDR training loss requires the current training forward")
        terms = prototype_discriminative_terms(self._cached_feature, masks, margin=self.margin,
            normalize_features=self.normalize_features, ignore_index=self.ignore_index)
        if self.mode == "joint":
            auxiliary = terms["separation"] + terms["compactness"]
        elif self.mode == "separation":
            auxiliary = terms["separation"]
        else:
            auxiliary = terms["compactness"]
        auxiliary = self.weight * auxiliary
        self._pdr_batches += 1
        self._pdr_sums["separation"] += float(terms["separation"].detach())
        self._pdr_sums["compactness"] += float(terms["compactness"].detach())
        self._pdr_sums["weighted_auxiliary"] += float(auxiliary.detach())
        for label in terms["eligible_classes"].tolist():
            self._pdr_class_batches[str(label)] += 1
        self._cached_feature = None
        return base_loss + auxiliary

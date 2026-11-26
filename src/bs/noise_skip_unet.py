"""Training-only FA adaptation of NoiseUNet (arXiv2606.04427)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d


class NoiseSkipUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 mode: str = "skip", bound: float = .5, noise_seed: int = 42):
        super().__init__()
        if mode not in {"skip", "encoder"} or not 0 <= bound <= 1:
            raise ValueError("Require skip/encoder and fixed bounded perturbation")
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        self.mode, self.bound, self.noise_seed = mode, float(bound), int(noise_seed)
        self.epoch = 1
        self._generators = {}
        self.injection_count = 0
        if mode == "encoder":
            # The same five skip-producing levels are perturbed. Encoder mode
            # additionally propagates their perturbed outputs into deeper levels.
            for index, stage in enumerate(self.encoder.stages[:-1]):
                stage.register_forward_hook(self._encoder_hook(index))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._generators.clear()
        self.injection_count = 0

    def _perturb(self, feature: Tensor, stage: int) -> Tensor:
        if not self.training or self.bound == 0:
            return feature
        key = (str(feature.device), stage)
        if key not in self._generators:
            generator = torch.Generator(device=feature.device)
            generator.manual_seed(self.noise_seed + 1009 * self.epoch + 100003 * (stage + 1))
            self._generators[key] = generator
        noise = torch.empty_like(feature).uniform_(-self.bound, self.bound, generator=self._generators[key])
        self.injection_count += 1
        # Out-of-place addition keeps skip-only noise out of the encoder path.
        return feature + noise

    def _encoder_hook(self, stage: int):
        def hook(module, inputs, output):
            return self._perturb(output, stage)
        return hook

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        if self.mode == "skip":
            features = [self._perturb(feature, index) if index < len(features) - 1 else feature
                        for index, feature in enumerate(features)]
        return self.decoder(features)

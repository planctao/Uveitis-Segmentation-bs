from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    """Exponential moving average copy of a training model."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    model_value.to(device=ema_value.device, dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value.to(device=ema_value.device))

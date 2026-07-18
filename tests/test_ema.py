from __future__ import annotations

import pytest
import torch
from torch import nn

from bs.ema import ModelEMA


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))


def test_model_ema_blends_float_state_and_copies_integer_buffers() -> None:
    model = TinyModule()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 3.0]]))
        model.steps.fill_(1)

    ema = ModelEMA(model, decay=0.5)

    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[5.0, 5.0]]))
        model.steps.fill_(7)
    ema.update(model)

    assert torch.allclose(ema.module.linear.weight, torch.tensor([[3.0, 4.0]]))
    assert int(ema.module.steps.item()) == 7
    assert all(not parameter.requires_grad for parameter in ema.module.parameters())


def test_model_ema_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        ModelEMA(TinyModule(), decay=1.0)

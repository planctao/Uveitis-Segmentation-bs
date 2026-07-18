import torch
from torch import nn

from bs.augmentations import normalize
from bs.tta import predict_with_tta


class EchoTwoChannelModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :2]


def test_predict_with_tta_can_average_fa_lce_appearance_view() -> None:
    raw = torch.full((1, 3, 9, 9), 0.20, dtype=torch.float32)
    raw[:, :, 4, 4] = 0.55
    images = normalize(raw)
    model = EchoTwoChannelModel()

    plain = predict_with_tta(model, images, {"enabled": True, "flips": [], "scales": [1.0]})
    with_appearance = predict_with_tta(
        model,
        images,
        {
            "enabled": True,
            "flips": [],
            "scales": [1.0],
            "appearance_preprocess": {
                "enabled": True,
                "mode": "fa_lce",
                "kernel_size": 3,
                "strength": 0.5,
                "quantile": 1.0,
                "reference_threshold": 0.01,
            },
        },
    )

    assert torch.all(with_appearance[:, :, 4, 4] > plain[:, :, 4, 4])
    assert torch.allclose(with_appearance[:, :, 0, 0], plain[:, :, 0, 0])

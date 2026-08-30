import torch

from bs.interactive_correction import ClickPCMRefiner, build_clickpcm_features


def test_distance_prompt_layout_has_fifteen_channels():
    image = torch.randn(1, 3, 16, 16)
    logits = torch.randn(1, 2, 16, 16)
    pos = torch.zeros_like(logits)
    neg = torch.zeros_like(logits)
    features = build_clickpcm_features(image, logits, pos, neg, add_distance=True)
    assert features.shape == (1, 15, 16, 16)


def test_fifteen_channel_model_forward():
    model = ClickPCMRefiner(base_channels=4, dropout=0.0, in_channels=15)
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    pos = torch.zeros_like(logits)
    neg = torch.zeros_like(logits)
    features = build_clickpcm_features(image, logits, pos, neg, add_distance=True)
    output = model(features, logits)
    assert output.shape == logits.shape
    assert torch.isfinite(output).all()

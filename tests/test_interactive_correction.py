import torch

from bs.interactive_correction import (
    ClickPCMRefiner,
    build_clickpcm_features,
    correction_step,
    progressive_merge,
)


def test_clickpcm_features_include_previous_mask_and_signed_prompt():
    image = torch.randn(2, 3, 32, 32)
    logits = torch.randn(2, 2, 32, 32)
    pos = torch.zeros_like(logits)
    neg = torch.zeros_like(logits)
    pos[:, :, 10:12, 10:12] = 1
    features = build_clickpcm_features(image, logits, pos, neg)
    assert features.shape == (2, 13, 32, 32)
    assert torch.allclose(features[:, 11:13], pos)


def test_zero_click_step_is_identity():
    model = ClickPCMRefiner(base_channels=4, dropout=0.0)
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    zero = torch.zeros_like(logits)
    result = correction_step(model, image, logits, zero, zero, variant="b3", crop_size=16)
    assert torch.allclose(result.logits, logits)


def test_progressive_merge_bounds_and_preserves_shape():
    previous = torch.zeros(1, 2, 8, 8)
    refined = torch.full_like(previous, 100.0)
    pos = torch.zeros_like(previous)
    neg = torch.zeros_like(previous)
    pos[:, :, 4, 4] = 1
    merged = progressive_merge(previous, refined, pos, neg, max_delta=2.0)
    assert merged.shape == previous.shape
    assert float(merged.max()) <= 2.0 + 1e-5
    assert float(merged.min()) >= 0.0

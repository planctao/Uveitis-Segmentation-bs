import torch

from bs.click_simulator import (
    build_pseudo_sam_candidate,
    build_refiner_features,
    simulate_click_heatmaps,
    simulate_click_sequence,
)


def test_simulate_click_heatmaps_shapes_and_regions():
    target = torch.zeros(1, 2, 16, 16)
    probs = torch.zeros(1, 2, 16, 16)
    target[:, 0, 4:8, 4:8] = 1.0
    probs[:, 1, 10:13, 10:13] = 1.0

    positive, negative = simulate_click_heatmaps(target, probs, num_clicks=2, threshold=0.5, radius=2)

    assert positive.shape == (1, 2, 16, 16)
    assert negative.shape == (1, 2, 16, 16)
    assert positive[:, 0].sum() > 0
    assert negative[:, 1].sum() > 0


def test_build_refiner_features_has_expected_channels():
    image = torch.randn(1, 3, 16, 16)
    logits = torch.randn(1, 2, 16, 16)
    positive = torch.zeros(1, 2, 16, 16)
    negative = torch.zeros(1, 2, 16, 16)
    candidate = build_pseudo_sam_candidate(torch.sigmoid(logits), positive, negative)

    features = build_refiner_features(image, logits, candidate, positive, negative)

    assert features.shape == (1, 13, 16, 16)
    assert torch.isfinite(features).all()


def test_cumulative_click_sequence_moves_to_new_residual_pixels():
    target = torch.zeros(1, 1, 12, 12)
    target[:, :, 2:4, 2:4] = 1.0
    target[:, :, 8:10, 8:10] = 1.0
    prediction = torch.zeros_like(target)

    positive, negative = simulate_click_sequence(
        target, prediction, num_clicks=2, strategy="farthest"
    )

    assert positive.shape == (1, 1, 2, 2)
    assert negative.eq(-1).all()
    assert not torch.equal(positive[:, :, 0], positive[:, :, 1])

import torch

from bs.interactive_boundary import (
    BoundaryAwareInteractiveRefiner,
    boundary_click_priority_scores,
    boundary_f1_score,
    binary_entropy,
    build_fa_aware_image,
    build_boundary_refiner_features,
    make_boundary_target,
    make_uncertainty_target,
    recommend_boundary_click_points,
    refine_boundary_with_clicks,
    should_stop_boundary_interaction,
    boundary_uncertainty_loss,
)


def test_binary_entropy_has_expected_extrema():
    certain = torch.tensor([[[[0.0, 1.0]]]])
    ambiguous = torch.full_like(certain, 0.5)
    assert float(binary_entropy(certain).max()) < 1e-4
    assert torch.allclose(binary_entropy(ambiguous), torch.ones_like(ambiguous), atol=1e-5)


def test_fa_aware_image_has_three_bounded_channels():
    image = torch.randn(2, 3, 16, 16)
    output = build_fa_aware_image(image)
    assert output.shape == (2, 3, 16, 16)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_boundary_and_uncertainty_targets_are_bounded():
    target = torch.zeros(1, 2, 16, 16)
    target[:, 0, 4:12, 4:12] = 1.0
    valid = torch.ones_like(target)
    logits = torch.zeros_like(target)
    boundary = make_boundary_target(target, valid, kernel_size=3, soft=True)
    uncertainty = make_uncertainty_target(logits, target, valid, boundary_kernel=3)
    assert boundary.shape == target.shape
    assert uncertainty.shape == target.shape
    assert 0.0 <= float(boundary.min()) <= float(boundary.max()) <= 1.0
    assert 0.0 <= float(uncertainty.min()) <= float(uncertainty.max()) <= 1.0


def test_feature_layout_is_19_channels():
    image = torch.randn(1, 3, 16, 16)
    logits = torch.randn(1, 2, 16, 16)
    uncertainty = torch.rand_like(logits)
    boundary = torch.rand_like(logits)
    positive = torch.zeros_like(logits)
    negative = torch.zeros_like(logits)
    positive[:, :, 4:6, 4:6] = 1.0
    features = build_boundary_refiner_features(
        image, logits, uncertainty, boundary, positive, negative
    )
    assert features.shape == (1, 19, 16, 16)
    assert torch.isfinite(features).all()


def test_zero_click_refiner_preserves_coarse_logits():
    model = BoundaryAwareInteractiveRefiner(base_channels=4, dropout=0.0)
    model.eval()
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    output = model(image, logits)
    assert torch.allclose(output, logits, atol=1e-6)


def test_boundary_policy_returns_one_polarity_and_is_target_free():
    probabilities = torch.full((1, 1, 32, 32), 0.5)
    uncertainty = torch.ones_like(probabilities)
    boundary = torch.zeros_like(probabilities)
    boundary[:, :, 8:12, 8:12] = 1.0
    pos, neg = recommend_boundary_click_points(
        probabilities, uncertainty, boundary, num_clicks=2, min_separation=4
    )
    assert pos.shape == (1, 1, 2, 2)
    assert neg.shape == pos.shape
    for step in range(2):
        has_pos = bool((pos[:, :, step] >= 0).all())
        has_neg = bool((neg[:, :, step] >= 0).all())
        assert has_pos != has_neg


def test_boundary_stop_policy_uses_uncertainty_and_boundary():
    probabilities = torch.full((1, 1, 16, 16), 0.5)
    low_uncertainty = torch.zeros_like(probabilities)
    boundary = torch.ones_like(probabilities)
    assert should_stop_boundary_interaction(
        probabilities, low_uncertainty, boundary, min_priority=0.05
    ).all()
    assert not should_stop_boundary_interaction(
        probabilities, torch.ones_like(probabilities), boundary, min_priority=0.05
    ).all()


def test_policy_loop_runs_with_roi_and_has_expected_history():
    model = BoundaryAwareInteractiveRefiner(base_channels=4, dropout=0.0)
    model.eval()
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    result = refine_boundary_with_clicks(
        model, image, logits, num_clicks=2, roi_size=16, min_separation=4
    )
    assert len(result.history) == 3
    assert result.logits.shape == logits.shape
    assert result.positive_clicks.shape == logits.shape
    assert torch.isfinite(result.uncertainty).all()


def test_auxiliary_loss_backpropagates():
    dino_logits = torch.randn(1, 2, 16, 16)
    boundary_logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    uncertainty_logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    target = torch.zeros(1, 2, 16, 16)
    target[:, 0, 4:12, 4:12] = 1.0
    total, values = boundary_uncertainty_loss(
        boundary_logits, uncertainty_logits, dino_logits, target,
        boundary_kernel=3, pos_weight=[2.0, 4.0]
    )
    total.backward()
    assert torch.isfinite(total)
    assert boundary_logits.grad is not None and torch.isfinite(boundary_logits.grad).all()
    assert uncertainty_logits.grad is not None and torch.isfinite(uncertainty_logits.grad).all()
    assert set(values) >= {"boundary_loss", "uncertainty_loss"}


def test_boundary_f1_is_high_for_aligned_prediction():
    target = torch.zeros(1, 2, 16, 16)
    target[:, 0, 4:12, 4:12] = 1.0
    logits = torch.full_like(target, -8.0)
    logits[:, 0, 4:12, 4:12] = 8.0
    score = boundary_f1_score(logits, target, kernel_size=3)
    assert float(score) > 0.9

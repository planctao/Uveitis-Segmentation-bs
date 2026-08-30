import torch

from bs.click_simulator import (
    build_pseudo_sam_candidate,
    build_soft_prompt_features,
    should_stop_interaction,
    perturb_click_points,
    recommend_click_points,
)
from bs.interactive_refiner import (
    UncertaintyGatedResidualRefiner,
    _controlled_residual_update,
    _teacher_force_state,
    click_consistency_loss,
    policy_refine_with_clicks,
)


def test_soft_prompt_features_add_signed_and_reliability_channels():
    image = torch.randn(1, 3, 16, 16)
    logits = torch.randn(1, 2, 16, 16)
    positive = torch.zeros(1, 2, 16, 16)
    negative = torch.zeros_like(positive)
    positive[:, 0, 4:8, 4:8] = 1.0
    negative[:, 1, 9:12, 9:12] = 1.0
    candidate = build_pseudo_sam_candidate(torch.sigmoid(logits), positive, negative)

    features = build_soft_prompt_features(image, logits, candidate, positive, negative)

    assert features.shape == (1, 17, 16, 16)
    assert torch.allclose(features[:, 13], positive[:, 0] - negative[:, 0])
    assert torch.isfinite(features).all()


def test_uag_refiner_preserves_dino_at_initialization():
    model = UncertaintyGatedResidualRefiner(in_channels=17, out_channels=2, base_channels=4)
    features = torch.randn(1, 17, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    output = model(features, logits)
    assert torch.allclose(output, logits, atol=1e-6)


def test_recommend_click_points_is_target_free_and_spatially_separated():
    probabilities = torch.full((1, 1, 32, 32), 0.5)
    probabilities[:, :, 8:12, 8:12] = 0.25
    probabilities[:, :, 22:26, 22:26] = 0.75
    positive, negative = recommend_click_points(probabilities, num_clicks=2, min_separation=4)
    assert positive.shape == (1, 1, 2, 2)
    assert negative.shape == (1, 1, 2, 2)
    assert (positive >= -1).all() and (negative >= -1).all()
    assert not torch.equal(positive[:, :, 0], positive[:, :, 1])
    assert not torch.equal(negative[:, :, 0], negative[:, :, 1])


def test_stop_policy_detects_ambiguous_regions_but_not_certain_regions():
    ambiguous = torch.full((1, 1, 16, 16), 0.5)
    certain = torch.zeros_like(ambiguous)
    assert should_stop_interaction(ambiguous, min_priority=0.05).all()
    assert should_stop_interaction(certain, min_priority=0.05).all()
    certain[:, :, 4:8, 4:8] = 0.25
    assert not should_stop_interaction(certain, min_priority=0.05).all()


def test_policy_refinement_runs_without_target():
    model = UncertaintyGatedResidualRefiner(in_channels=17, out_channels=2, base_channels=4)
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    result = policy_refine_with_clicks(model, image, logits, num_clicks=2)
    assert len(result.history) == 3
    assert result.logits.shape == logits.shape
    assert torch.isfinite(result.logits).all()


def test_click_noise_keeps_missing_coordinates_missing():
    points = torch.tensor([[[[4, 5], [-1, -1]]]])
    noisy = perturb_click_points(points, 16, 16, jitter=3.0, dropout=0.0)
    assert noisy[0, 0, 1].tolist() == [-1, -1]
    assert ((noisy[0, 0, 0] >= 0) & (noisy[0, 0, 0] < 16)).all()


def test_click_consistency_loss_has_positive_gradient_signal():
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    positive = torch.zeros_like(logits)
    negative = torch.zeros_like(logits)
    positive[:, :, 3, 3] = 1.0
    negative[:, :, 4, 4] = 1.0
    loss = click_consistency_loss(logits, positive, negative, margin=2.0)
    loss.backward()
    assert float(loss) > 0.0
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 3, 3] < 0
    assert logits.grad[0, 0, 4, 4] > 0


def test_residual_step_limit_bounds_each_logit_update():
    class FixedDelta(torch.nn.Module):
        def forward(self, features, state):
            return state + torch.full_like(state, 10.0)

    state = torch.zeros(1, 1, 4, 4)
    updated = _controlled_residual_update(
        FixedDelta(), torch.zeros_like(state), state, residual_step_limit=0.5
    )
    assert float(updated.abs().max()) <= 0.5 + 1e-6


def test_stop_gradient_detaches_rollout_state_and_features():
    class IdentityDelta(torch.nn.Module):
        def forward(self, features, state):
            return state + features

    state = torch.zeros(1, 1, 4, 4, requires_grad=True)
    features = state.clone()
    updated = _controlled_residual_update(
        IdentityDelta(), features, state, stop_gradient=True
    )
    assert updated.requires_grad is False


def test_teacher_forcing_blends_only_with_bounded_teacher_logits():
    state = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(state)
    forced = _teacher_force_state(state, target, ratio=0.5, logit_scale=4.0)
    assert torch.allclose(forced, torch.full_like(state, 2.0))

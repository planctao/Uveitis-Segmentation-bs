import torch

from bs.click_simulator import click_points_to_heatmaps
from bs.interactive_refiner import (
    InteractiveResidualRefiner,
    ReliabilityAwareRefiner,
    _controlled_residual_update,
    enforce_click_constraints,
    refine_with_clicks,
)


def test_click_heatmaps_support_per_lesion_radii():
    points = torch.tensor([[[[16, 16]], [[16, 16]]]])
    heatmaps = click_points_to_heatmaps(points, 33, 33, radius=[6, 2], mode="disk")
    assert heatmaps.shape == (1, 2, 33, 33)
    assert int(heatmaps[0, 0].sum()) > int(heatmaps[0, 1].sum())
    assert heatmaps[0, 0, 16, 22] == 1
    assert heatmaps[0, 1, 16, 22] == 0


def test_interactive_refiner_forward_shape():
    model = InteractiveResidualRefiner(in_channels=13, out_channels=2, base_channels=8)
    features = torch.randn(2, 13, 64, 64)
    dino_logits = torch.randn(2, 2, 64, 64)

    output = model(features, dino_logits)

    assert output.shape == (2, 2, 64, 64)
    assert torch.isfinite(output).all()


def test_interactive_refiner_initially_preserves_dino_logits():
    model = ReliabilityAwareRefiner(in_channels=13, out_channels=2, base_channels=8)
    features = torch.randn(1, 13, 32, 32)
    dino_logits = torch.randn(1, 2, 32, 32)

    output = model(features, dino_logits)

    assert torch.allclose(output, dino_logits, atol=1e-6)


def test_refine_with_clicks_returns_one_output_per_round():
    model = InteractiveResidualRefiner(in_channels=13, out_channels=2, base_channels=4)
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    points = torch.tensor([[[[8, 8], [20, 20]], [[10, 10], [22, 22]]]])

    result = refine_with_clicks(model, image, logits, points, points * 0 - 1)

    assert len(result.history) == 2
    assert result.logits.shape == logits.shape
    assert torch.isfinite(result.logits).all()


def test_uncertainty_click_gate_limits_updates_outside_prompt_region():
    model = InteractiveResidualRefiner(
        in_channels=13,
        out_channels=2,
        base_channels=4,
        residual_gate="uncertainty_click",
        residual_gate_floor=0.25,
        residual_gate_click_gain=0.75,
    )
    # Make the residual non-zero so the parameter-free gate can be observed.
    with torch.no_grad():
        model.delta_head.bias.fill_(1.0)
    features = torch.zeros(1, 13, 32, 32)
    features[:, 5:7] = 0.0  # confident background -> floor gate
    features[:, 9:13, 12:16, 12:16] = 1.0  # clicked patch -> full gate
    logits = torch.zeros(1, 2, 32, 32)
    output = model(features, logits)
    outside = output[:, :, 0, 0]
    inside = output[:, :, 13, 13]
    assert torch.all(inside > outside * 2.0)
    assert torch.isfinite(output).all()


def test_channelwise_gate_does_not_cross_amplify_other_lesion():
    model = InteractiveResidualRefiner(
        in_channels=13,
        out_channels=2,
        base_channels=4,
        residual_gate="uncertainty_click_channelwise",
        residual_gate_floor=0.25,
        residual_gate_click_gain=0.75,
    )
    with torch.no_grad():
        model.delta_head.bias.copy_(torch.tensor([1.0, 1.0]))
    features = torch.zeros(1, 13, 32, 32)
    features[:, 5:7] = 0.0
    # Click only lesion_2.  Channel 1 should stay at the floor gate while
    # channel 2 receives the full click-aware update.
    features[:, 12:13, 12:16, 12:16] = 1.0
    output = model(features, torch.zeros(1, 2, 32, 32))
    outside = output[:, :, 0, 0]
    inside = output[:, :, 13, 13]
    assert torch.allclose(inside[:, 0], outside[:, 0], atol=1e-5)
    assert torch.all(inside[:, 1] > outside[:, 1] * 2.0)


def test_click_constraint_projection_honors_polarity_only_on_prompt():
    logits = torch.zeros(1, 2, 8, 8)
    positive = torch.zeros_like(logits)
    negative = torch.zeros_like(logits)
    positive[:, 0, 2, 3] = 1
    negative[:, 1, 5, 6] = 1
    projected = enforce_click_constraints(logits, positive, negative, strength=6.0)
    assert projected[0, 0, 2, 3] == 6.0
    assert projected[0, 1, 5, 6] == -6.0
    assert projected[0, 0, 0, 0] == 0.0


def test_total_residual_limit_anchors_rollout_drift():
    class ConstantDelta(torch.nn.Module):
        def forward(self, features, state):  # noqa: D401
            return state + 10.0

    model = ConstantDelta()
    features = torch.zeros(1, 1, 4, 4)
    anchor = torch.zeros(1, 1, 4, 4)
    state = anchor.clone()
    for _ in range(4):
        state = _controlled_residual_update(
            model,
            features,
            state,
            residual_step_limit=0.5,
            anchor_state=anchor,
            total_residual_limit=1.0,
        )
    assert float(state.max()) <= 1.0 + 1e-6
    assert float(state.min()) >= -1e-6


def test_residual_channel_scale_is_applied_independently():
    class ConstantDelta(torch.nn.Module):
        def forward(self, features, state):  # noqa: D401
            return state + 1.0

    model = ConstantDelta()
    features = torch.zeros(1, 2, 4, 4)
    state = torch.zeros(1, 2, 4, 4)
    updated = _controlled_residual_update(
        model,
        features,
        state,
        residual_channel_scale=[1.0, 0.25],
    )
    assert torch.allclose(updated[:, 0], torch.ones_like(updated[:, 0]))
    assert torch.allclose(updated[:, 1], torch.full_like(updated[:, 1], 0.25))

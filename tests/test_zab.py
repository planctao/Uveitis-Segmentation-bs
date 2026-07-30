from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.multilabel import masks_to_paper_targets
from bs.zab import ZABLeakageHead, ZABLoss, calibrate_logits_to_area


class _PaperBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target, _ = masks_to_paper_targets(mask)
        return F.binary_cross_entropy_with_logits(logits, target.to(logits))


def test_area_calibration_matches_mass_and_preserves_ranking() -> None:
    logits = torch.randn(3, 1, 24, 20)
    area = torch.tensor([[0.001], [0.01], [0.08]])

    calibrated = calibrate_logits_to_area(logits, area, iterations=4)
    calibrated_area = torch.sigmoid(calibrated).mean(dim=(-2, -1))

    assert torch.allclose(calibrated_area, area, atol=2e-4, rtol=0.03)
    assert torch.equal(logits.flatten(2).argsort(dim=-1), calibrated.flatten(2).argsort(dim=-1))


def test_area_calibration_is_finite_at_fp16_extremes() -> None:
    logits = torch.tensor([[[[-100.0, 100.0]]]], dtype=torch.float16)
    area = torch.tensor([[1e-6]], dtype=torch.float16)

    calibrated = calibrate_logits_to_area(logits, area, iterations=4)

    assert calibrated.dtype == torch.float16
    assert torch.isfinite(calibrated).all()


def test_zab_head_outputs_zero_inflated_burden() -> None:
    head = ZABLeakageHead(
        8,
        topk_fraction=0.25,
        presence_prior=0.2,
        area_prior=0.01,
        max_area_fraction=0.05,
    )
    features = torch.randn(2, 8, 12, 10)

    logits, auxiliary = head(features)

    assert logits.shape == (2, 2, 12, 10)
    assert auxiliary["anatomy_logits"].shape == (2, 1, 12, 10)
    assert auxiliary["presence_logits"].shape == (2, 1)
    assert torch.all(auxiliary["conditional_area_fraction"] > 0.0)
    assert torch.all(auxiliary["conditional_area_fraction"] < 0.05)
    assert torch.allclose(
        auxiliary["expected_area_fraction"],
        torch.sigmoid(auxiliary["presence_logits"]) * auxiliary["conditional_area_fraction"],
    )


def test_zab_hierarchy_suppresses_macular_logits_only_with_negative_retinal_evidence() -> None:
    base = ZABLeakageHead(8, topk_fraction=0.25, hierarchy_strength=0.0, calibration_iterations=0)
    coupled = ZABLeakageHead(8, topk_fraction=0.25, hierarchy_strength=0.75, calibration_iterations=0)
    coupled.load_state_dict(base.state_dict())
    base.eval()
    coupled.eval()
    features = torch.randn(2, 8, 12, 10)

    base_logits, _ = base(features)
    coupled_logits, auxiliary = coupled(features)
    negative_retinal = coupled_logits[:, 0] < 0.0
    positive_retinal = ~negative_retinal

    assert torch.all(coupled_logits[:, 1][negative_retinal] <= base_logits[:, 1][negative_retinal] + 1e-6)
    assert torch.allclose(
        coupled_logits[:, 1][positive_retinal],
        base_logits[:, 1][positive_retinal],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(auxiliary["hierarchy_evidence"] <= 0.0)


def test_zab_bidirectional_coupling_is_bounded() -> None:
    head = ZABLeakageHead(
        8,
        topk_fraction=0.25,
        bidirectional_strength=0.25,
        calibration_iterations=0,
    )
    features = torch.randn(2, 8, 12, 10)
    logits, _ = head(features)

    assert torch.isfinite(logits).all()
    assert torch.all(torch.sigmoid(logits) >= 0.0)
    assert torch.all(torch.sigmoid(logits) <= 1.0)


def test_zab_weak_anatomy_target_tracks_positive_region() -> None:
    criterion = ZABLoss(_PaperBCELoss(), anatomy_min_pixels=4, anatomy_sigma=0.08)
    macular = torch.zeros(2, 16, 16, dtype=torch.bool)
    valid = torch.ones_like(macular)
    macular[0, 10:14, 2:6] = True

    target, selected = criterion._anatomy_targets(macular, valid, (9, 9))
    peak = torch.nonzero(target[0, 0] == target[0, 0].max(), as_tuple=False)[0]

    assert selected.tolist() == [True, False]
    assert int(peak[0]) >= 6
    assert int(peak[1]) <= 3
    assert target[1].sum() == 0.0


def test_zab_loss_backpropagates_through_all_branches() -> None:
    head = ZABLeakageHead(8, topk_fraction=0.25, max_area_fraction=0.05)
    features = torch.randn(2, 8, 8, 8, requires_grad=True)
    logits, auxiliary = head(features)
    logits = F.interpolate(logits, size=(16, 16), mode="bilinear", align_corners=False)
    mask = torch.zeros(2, 16, 16, dtype=torch.long)
    mask[0, 2:12, 2:12] = 1
    mask[0, 7:12, 5:10] = 3
    mask[1, 1:8, 8:15] = 1
    criterion = ZABLoss(
        _PaperBCELoss(),
        anatomy_min_pixels=4,
        min_confidence_pixels=4,
    )

    loss = criterion(logits, mask, auxiliary)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert head.retinal_head.weight.grad is not None
    assert head.macular_rank_head.weight.grad is not None
    assert head.anatomy_head[-1].weight.grad is not None
    assert head.presence_head.weight.grad is not None
    assert head.area_head.weight.grad is not None


def test_zab_loss_is_finite_for_macular_negative_batch() -> None:
    head = ZABLeakageHead(8, topk_fraction=0.25)
    features = torch.randn(2, 8, 8, 8)
    logits, auxiliary = head(features)
    mask = torch.ones(2, 8, 8, dtype=torch.long)
    criterion = ZABLoss(_PaperBCELoss(), anatomy_min_pixels=4)

    loss = criterion(logits, mask, auxiliary)

    assert torch.isfinite(loss)


def test_convnext_zab_decoder_returns_auxiliary_only_while_training() -> None:
    decoder = ConvNeXtFPNDecoder(
        in_channels=[4, 8, 16, 32],
        decoder_channels=16,
        head_type="zab",
        zab_topk_fraction=0.25,
    )
    features = [
        torch.randn(2, 4, 32, 32),
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    decoder.train()

    train_logits, auxiliary = decoder(features, output_size=(64, 64))

    assert train_logits.shape == (2, 2, 64, 64)
    assert {"anatomy_logits", "presence_logits", "conditional_area_fraction"}.issubset(auxiliary)

    decoder.eval()
    eval_logits = decoder(features, output_size=(64, 64))
    assert isinstance(eval_logits, torch.Tensor)
    assert eval_logits.shape == (2, 2, 64, 64)

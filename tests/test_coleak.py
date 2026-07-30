from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bs.coleak import CoLeakLoss, CoupledLeakageHead, TopKPresencePool
from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.multilabel import masks_to_paper_targets


class _PaperBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target, _ = masks_to_paper_targets(mask)
        return F.binary_cross_entropy_with_logits(logits, target.to(logits))


def test_topk_presence_pool_keeps_sparse_signal() -> None:
    features = torch.zeros(1, 1, 4, 4)
    features[0, 0, 0, 0] = 8.0

    pooled = TopKPresencePool(1.0 / 16.0)(features)

    assert torch.allclose(pooled, torch.tensor([[8.0]]))


def test_coleak_head_recovers_conditional_marginal_without_global_prior() -> None:
    head = CoupledLeakageHead(8, topk_fraction=0.25, prior_strength=0.0)
    features = torch.randn(2, 8, 12, 10)

    logits, auxiliary = head(features)

    retinal = torch.sigmoid(logits[:, :1])
    expected_macular = (
        retinal * torch.sigmoid(auxiliary["inside_logits"])
        + (1.0 - retinal) * torch.sigmoid(auxiliary["outside_logits"])
    )
    assert logits.shape == (2, 2, 12, 10)
    assert auxiliary["presence_logits"].shape == (2, 1)
    assert torch.allclose(torch.sigmoid(logits[:, 1:]), expected_macular, atol=1e-5)


def test_coleak_logit_is_finite_at_fp16_probability_endpoints() -> None:
    probability = torch.tensor([0.0, 1.0], dtype=torch.float16)

    logits = CoupledLeakageHead._logit(probability)

    assert logits.dtype == torch.float16
    assert torch.isfinite(logits).all()


def test_coleak_loss_uses_all_four_palette_states_and_backpropagates() -> None:
    head = CoupledLeakageHead(8, topk_fraction=0.25)
    features = torch.randn(2, 8, 8, 8, requires_grad=True)
    logits, auxiliary = head(features)
    mask = torch.zeros(2, 16, 16, dtype=torch.long)
    mask[0, 1:9, 1:9] = 1
    mask[0, 3:6, 3:6] = 3
    mask[1, 10:14, 10:14] = 2
    criterion = CoLeakLoss(_PaperBCELoss(), hard_negative_min_pixels=8)

    loss = criterion(F.interpolate(logits, size=(16, 16)), mask, auxiliary)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert head.retinal_head.weight.grad is not None
    assert head.inside_head.weight.grad is not None
    assert head.outside_delta.weight.grad is not None
    assert head.presence_head[-1].weight.grad is not None


def test_coleak_loss_is_finite_for_child_negative_batch() -> None:
    head = CoupledLeakageHead(8, topk_fraction=0.25)
    features = torch.randn(2, 8, 8, 8)
    logits, auxiliary = head(features)
    mask = torch.ones(2, 8, 8, dtype=torch.long)
    criterion = CoLeakLoss(_PaperBCELoss(), hard_negative_min_pixels=8)

    loss = criterion(logits, mask, auxiliary)

    assert torch.isfinite(loss)


def test_convnext_coleak_decoder_returns_auxiliary_only_while_training() -> None:
    decoder = ConvNeXtFPNDecoder(
        in_channels=[4, 8, 16, 32],
        decoder_channels=16,
        head_type="coleak",
        coleak_topk_fraction=0.25,
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
    assert {"inside_logits", "outside_logits", "presence_logits"}.issubset(auxiliary)

    decoder.eval()
    eval_logits = decoder(features, output_size=(64, 64))
    assert isinstance(eval_logits, torch.Tensor)
    assert eval_logits.shape == (2, 2, 64, 64)

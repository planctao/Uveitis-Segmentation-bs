from __future__ import annotations

import torch

from bs.multilabel import AsymmetricFocalTverskyBCE, masks_to_paper_targets


def _make_batch():
    mask = torch.zeros(1, 32, 32, dtype=torch.long)
    mask[0, 8:24, 8:24] = 1  # lesion_1
    mask[0, 4:12, 20:28] = 2  # lesion_2
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 32, 32)
    return logits, mask


def test_soft_boundary_disabled_matches_default():
    logits, mask = _make_batch()
    explicit_off = AsymmetricFocalTverskyBCE(soft_boundary_sigma=0.0)
    default = AsymmetricFocalTverskyBCE()
    assert torch.allclose(explicit_off(logits, mask), default(logits, mask))


def test_soft_boundary_target_has_intermediate_values():
    _, mask = _make_batch()
    loss = AsymmetricFocalTverskyBCE(soft_boundary_sigma=2.0, soft_boundary_band=7, soft_boundary_weight=1.0)
    target, valid = masks_to_paper_targets(mask)
    target = target.float()
    valid = valid.expand_as(target).float()
    soft = loss._soft_boundary_target(target, valid)
    assert float(soft.min()) >= 0.0 and float(soft.max()) <= 1.0
    intermediate = ((soft > 0.01) & (soft < 0.99)).float().mean()
    assert float(intermediate) > 0.0
    # 病灶核心 (远离边界) 仍保持接近 1
    assert float(soft[0, 0, 15, 15]) > 0.9


def test_soft_boundary_loss_is_differentiable():
    logits, mask = _make_batch()
    logits = logits.clone().requires_grad_(True)
    loss = AsymmetricFocalTverskyBCE(soft_boundary_sigma=2.0)
    value = loss(logits, mask)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

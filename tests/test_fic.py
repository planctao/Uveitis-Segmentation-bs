from __future__ import annotations

import torch

from bs.multilabel import (
    AsymmetricFocalTverskyBCE,
    EOCLoss,
    FICLoss,
    edge_orientation_consistency,
)


def _make_batch(size: int = 32):
    logits = torch.randn(2, 2, size, size, requires_grad=True)
    mask = torch.zeros(2, size, size, dtype=torch.long)
    mask[:, size // 4 : size // 2, size // 4 : size // 2] = 1
    mask[0, size // 2 : 3 * size // 4, size // 2 : 3 * size // 4] = 2
    image = torch.rand(2, 3, size, size)
    return logits, mask, image


def test_fic_declares_needs_image():
    assert getattr(FICLoss, "needs_image", False) is True


def test_fic_weight_zero_returns_base_loss():
    base = AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0))
    fic = FICLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.0)
    logits, mask, image = _make_batch()
    expected = base(logits, mask)
    got = fic(logits, mask, image)
    assert torch.allclose(got, expected, atol=1e-6)


def test_fic_image_none_returns_base_loss():
    base = AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0))
    fic = FICLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.2)
    logits, mask, _ = _make_batch()
    expected = base(logits, mask)
    got = fic(logits, mask, None)
    assert torch.allclose(got, expected, atol=1e-6)


def test_fic_adds_finite_positive_term_and_is_differentiable():
    base = AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0))
    fic = FICLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.2, pool=2)
    logits, mask, image = _make_batch()
    base_value = base(logits.detach(), mask)
    total = fic(logits, mask, image)
    assert torch.isfinite(total).all()
    # FIC 项非负 (1 - corr >= 0 的加权平均), 故总损失不小于基础损失
    assert float(total.detach()) >= float(base_value.detach()) - 1e-5
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_fic_amp_fp16_is_finite():
    if not torch.cuda.is_available():
        return
    fic = FICLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.2).cuda()
    logits = torch.randn(2, 2, 64, 64, device="cuda", requires_grad=True)
    mask = torch.zeros(2, 64, 64, dtype=torch.long, device="cuda")
    mask[:, 16:32, 16:32] = 1
    image = torch.rand(2, 3, 64, 64, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total = fic(logits, mask, image)
    assert torch.isfinite(total).all()


def test_eoc_declares_needs_image():
    assert getattr(EOCLoss, "needs_image", False) is True


def test_eoc_weight_zero_and_none_image_return_base():
    base = AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0))
    eoc0 = EOCLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.0)
    eocN = EOCLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.3)
    logits, mask, image = _make_batch()
    assert torch.allclose(eoc0(logits, mask, image), base(logits, mask), atol=1e-6)
    assert torch.allclose(eocN(logits, mask, None), base(logits, mask), atol=1e-6)


def test_eoc_adds_finite_term_and_differentiable():
    base = AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0))
    eoc = EOCLoss(AsymmetricFocalTverskyBCE(pos_weight=(3.0, 60.0)), weight=0.3, pool=2)
    logits, mask, image = _make_batch()
    base_value = base(logits.detach(), mask)
    total = eoc(logits, mask, image)
    assert torch.isfinite(total).all()
    assert float(total.detach()) >= float(base_value.detach()) - 1e-5
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_eoc_term_zero_when_aligned_and_larger_when_orthogonal():
    # 构造一个水平梯度场作为图像强度; logits 与其成比例 -> 梯度方向完全对齐 -> EOC≈0
    ramp_x = torch.linspace(0, 1, 32).view(1, 1, 1, 32).repeat(1, 1, 32, 1)
    ramp_y = torch.linspace(0, 1, 32).view(1, 1, 32, 1).repeat(1, 1, 1, 32)
    image_x = ramp_x.repeat(1, 3, 1, 1)
    logits_aligned = (ramp_x * 6.0 - 3.0).repeat(1, 2, 1, 1)   # ∇p ∥ ∇I (都沿 x)
    logits_orth = (ramp_y * 6.0 - 3.0).repeat(1, 2, 1, 1)      # ∇p ⊥ ∇I (沿 y)
    aligned = float(edge_orientation_consistency(logits_aligned, image_x, pool=1))
    orthogonal = float(edge_orientation_consistency(logits_orth, image_x, pool=1))
    assert aligned < 1e-2
    assert orthogonal > aligned + 0.3

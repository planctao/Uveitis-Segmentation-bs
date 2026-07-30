import torch

from bs.multilabel import AsymmetricFocalTverskyBCE


def test_boundary_dou_weight_zero_matches_default_loss() -> None:
    logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(31))
    mask = torch.zeros((2, 16, 16), dtype=torch.long)
    mask[:, 4:12, 4:12] = 1

    default_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    dou_off_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dou_weight=0.0)

    assert torch.allclose(default_loss(logits, mask), dou_off_loss(logits, mask))


def test_boundary_dou_changes_loss_and_stays_finite() -> None:
    logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(37))
    mask = torch.zeros((2, 16, 16), dtype=torch.long)
    mask[:, 4:12, 4:12] = 1

    plain_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dou_weight=0.0)
    dou_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dou_weight=1.0)

    value = dou_loss(logits, mask)
    assert torch.isfinite(value)
    assert not torch.allclose(plain_loss(logits, mask), value)


def test_boundary_dou_is_zero_for_perfect_prediction() -> None:
    loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dou_weight=1.0)
    target = torch.zeros((1, 2, 16, 16))
    target[:, :, 4:12, 4:12] = 1.0
    valid = torch.ones_like(target)

    dou = loss._boundary_dou_loss(target.clone(), target, valid)
    assert float(dou) < 1e-4


def test_boundary_dou_alpha_cap_respected() -> None:
    # 大目标 (几乎整幅前景) 时 alpha 应被 cap 限制, 损失保持有限
    loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dou_weight=1.0, boundary_dou_alpha_cap=0.8)
    target = torch.ones((1, 2, 16, 16))
    valid = torch.ones_like(target)
    probs = torch.full_like(target, 0.6)

    dou = loss._boundary_dou_loss(probs, target, valid)
    assert torch.isfinite(dou)
    assert float(dou) >= 0.0

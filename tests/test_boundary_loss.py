import torch

from bs.multilabel import AsymmetricFocalTverskyBCE


def test_boundary_weight_zero_matches_default_loss() -> None:
    logits = torch.randn((2, 2, 8, 8), generator=torch.Generator().manual_seed(7))
    mask = torch.zeros((2, 8, 8), dtype=torch.long)
    mask[:, 2:6, 2:6] = 1

    default_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    boundary_off_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_weight=0.0, boundary_kernel=5)

    assert torch.allclose(default_loss(logits, mask), boundary_off_loss(logits, mask))


def test_hard_negative_ratio_zero_matches_default_loss() -> None:
    logits = torch.randn((2, 2, 8, 8), generator=torch.Generator().manual_seed(17))
    mask = torch.zeros((2, 8, 8), dtype=torch.long)
    mask[:, 2:6, 2:6] = 1

    default_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    hard_negative_off_loss = AsymmetricFocalTverskyBCE(
        pos_weight=(1.0, 1.0),
        hard_negative_ratio=0.0,
        hard_negative_min_pixels=8,
    )

    assert torch.allclose(default_loss(logits, mask), hard_negative_off_loss(logits, mask))


def test_boundary_dice_weight_zero_matches_default_loss() -> None:
    logits = torch.randn((2, 2, 8, 8), generator=torch.Generator().manual_seed(23))
    mask = torch.zeros((2, 8, 8), dtype=torch.long)
    mask[:, 2:6, 2:6] = 1

    default_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    boundary_dice_off_loss = AsymmetricFocalTverskyBCE(
        pos_weight=(1.0, 1.0),
        boundary_dice_weight=0.0,
        boundary_dice_kernel=5,
    )

    assert torch.allclose(default_loss(logits, mask), boundary_dice_off_loss(logits, mask))


def test_boundary_band_highlights_edges() -> None:
    loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    target = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 1.0
    valid = torch.ones_like(target)

    boundary = loss._boundary_band(target, valid, kernel_size=3)

    assert boundary.shape == target.shape
    assert float(boundary.max()) <= 1.0
    assert float(boundary.min()) >= 0.0
    assert bool(boundary[0, 0, 2, 2] > 0)
    assert bool(boundary[0, 0, 4, 4] == 0)


def test_hard_negative_weight_map_keeps_positives_and_hardest_negatives() -> None:
    loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), hard_negative_ratio=0.25)
    bce = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
    target = torch.zeros_like(bce)
    target[0, 0, 0, 0] = 1.0
    valid = torch.ones_like(bce)

    keep = loss._hard_negative_weight_map(bce, target, valid).bool()

    assert bool(keep[0, 0, 0, 0])
    assert int(keep[0, 0].sum()) == 3
    assert bool(keep[0, 0, 1, 2])
    assert bool(keep[0, 0, 1, 3])
    assert int(keep[0, 1].sum()) == 2
    assert bool(keep[0, 1, 1, 2])
    assert bool(keep[0, 1, 1, 3])


def test_hard_negative_weight_map_supports_per_channel_ratios() -> None:
    loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), hard_negative_ratio=(0.0, 0.25))
    bce = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
    target = torch.zeros_like(bce)
    target[0, 0, 0, 0] = 1.0
    valid = torch.ones_like(bce)

    keep = loss._hard_negative_weight_map(bce, target, valid).bool()

    assert int(keep[0, 0].sum()) == 8
    assert int(keep[0, 1].sum()) == 2
    assert bool(keep[0, 1, 1, 2])
    assert bool(keep[0, 1, 1, 3])


def test_hard_negative_ratio_rejects_invalid_value() -> None:
    try:
        AsymmetricFocalTverskyBCE(hard_negative_ratio=1.5)
    except ValueError as error:
        assert "hard_negative_ratio" in str(error)
    else:
        raise AssertionError("Expected invalid hard_negative_ratio to raise ValueError")

    try:
        AsymmetricFocalTverskyBCE(hard_negative_ratio=(0.2, -0.1))
    except ValueError as error:
        assert "hard_negative_ratio" in str(error)
    else:
        raise AssertionError("Expected invalid per-channel hard_negative_ratio to raise ValueError")


def test_boundary_weight_changes_loss_and_stays_finite() -> None:
    logits = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(11))
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    mask[:, 2:6, 2:6] = 1

    plain_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_weight=0.0)
    boundary_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_weight=2.0, boundary_kernel=3)

    loss_value = boundary_loss(logits, mask)

    assert torch.isfinite(loss_value)
    assert not torch.allclose(plain_loss(logits, mask), loss_value)


def test_hard_negative_loss_changes_loss_and_stays_finite() -> None:
    logits = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(19))
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    mask[:, 2:5, 2:5] = 1

    plain_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), hard_negative_ratio=0.0)
    hard_negative_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), hard_negative_ratio=0.25)

    loss_value = hard_negative_loss(logits, mask)

    assert torch.isfinite(loss_value)
    assert not torch.allclose(plain_loss(logits, mask), loss_value)


def test_boundary_dice_loss_changes_loss_and_stays_finite() -> None:
    logits = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(29))
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    mask[:, 2:6, 2:6] = 1

    plain_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0), boundary_dice_weight=0.0)
    boundary_dice_loss = AsymmetricFocalTverskyBCE(
        pos_weight=(1.0, 1.0),
        boundary_dice_weight=0.2,
        boundary_dice_kernel=3,
    )

    loss_value = boundary_dice_loss(logits, mask)

    assert torch.isfinite(loss_value)
    assert not torch.allclose(plain_loss(logits, mask), loss_value)

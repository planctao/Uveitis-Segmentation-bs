import torch

from bs.edge import (
    AngularDifferenceConv2d,
    CentralDifferenceConv2d,
    EdgeGuidedHead,
    GeodesicActiveContourHead,
    PDCBank,
    RadialDifferenceConv2d,
    boundary_band,
    lesion_edge_target,
    make_pdc_conv,
)
from bs.multilabel import AsymmetricFocalTverskyBCE, EdgeGuidedLoss


def test_cpdc_zero_response_on_constant_interior() -> None:
    conv = CentralDifferenceConv2d(3, 4, kernel_size=3, padding=1, bias=False)
    x = torch.full((1, 3, 8, 8), 0.37)
    out = conv(x)
    # 中心差分卷积对常数区域响应为 0 (内部像素, 避开零填充边界)
    assert torch.allclose(out[..., 1:-1, 1:-1], torch.zeros_like(out[..., 1:-1, 1:-1]), atol=1e-5)


def test_cpdc_responds_to_edges() -> None:
    conv = CentralDifferenceConv2d(1, 1, kernel_size=3, padding=1, bias=False)
    x = torch.zeros((1, 1, 8, 8))
    x[..., 4:] = 1.0  # 阶跃边缘
    out = conv(x)
    assert float(out.abs().max()) > 1e-3


def test_edge_head_returns_seg_and_edge_shapes() -> None:
    head = EdgeGuidedHead(in_channels=32, out_channels=2, edge_channels=0)
    feat = torch.randn((2, 32, 12, 12), generator=torch.Generator().manual_seed(3))
    seg_logits, edge_logits = head(feat)
    assert seg_logits.shape == (2, 2, 12, 12)
    assert edge_logits.shape == (2, 2, 12, 12)


def test_edge_gate_zero_initialised_is_identity() -> None:
    head = EdgeGuidedHead(in_channels=16, out_channels=2)
    edge_feat = torch.randn((2, head.edge_gate.in_channels, 10, 10))
    # gate conv 零初始化 -> 输出恒为 0 -> refined == feat, 初始退化为普通卷积头
    assert torch.allclose(head.edge_gate(edge_feat), torch.zeros_like(head.edge_gate(edge_feat)))


def test_boundary_band_highlights_edges() -> None:
    target = torch.zeros((1, 2, 16, 16))
    target[:, :, 4:12, 4:12] = 1.0
    valid = torch.ones_like(target)
    band = boundary_band(target, valid, kernel_size=3)
    assert band.shape == target.shape
    assert 0.0 <= float(band.min()) and float(band.max()) <= 1.0
    assert bool(band[0, 0, 4, 4] > 0)   # 病灶边界处
    assert bool(band[0, 0, 8, 8] == 0)  # 病灶内部深处


def test_lesion_edge_target_soft_widens_band() -> None:
    target = torch.zeros((1, 2, 16, 16))
    target[:, :, 4:12, 4:12] = 1.0
    valid = torch.ones_like(target)
    hard = lesion_edge_target(target, valid, band=3, soft=False)
    soft = lesion_edge_target(target, valid, band=3, soft=True, sigma=1.5)
    assert 0.0 <= float(soft.min()) and float(soft.max()) <= 1.0
    assert torch.isfinite(soft).all()
    assert not torch.allclose(hard, soft)
    # 软边缘经高斯扩散, 非零像素数不少于硬边缘
    assert int((soft > 1e-4).sum()) >= int((hard > 1e-4).sum())


def _square_mask() -> torch.Tensor:
    mask = torch.zeros((2, 16, 16), dtype=torch.long)
    mask[:, 4:12, 4:12] = 1
    return mask


def test_edge_guided_loss_weight_zero_matches_segmentation_loss() -> None:
    logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(5))
    edge_logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(6))
    mask = _square_mask()

    seg_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    edge_loss = EdgeGuidedLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)), edge_weight=0.0)

    reference = seg_loss(logits, mask)
    value = edge_loss(logits, mask, {"edge_logits": edge_logits})
    assert torch.allclose(reference, value)


def test_edge_guided_loss_without_auxiliary_matches_segmentation_loss() -> None:
    logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(7))
    mask = _square_mask()

    seg_loss = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    edge_loss = EdgeGuidedLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)), edge_weight=0.5)

    assert torch.allclose(seg_loss(logits, mask), edge_loss(logits, mask, None))


def test_edge_guided_loss_changes_loss_and_stays_finite() -> None:
    logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(8))
    edge_logits = torch.randn((2, 2, 16, 16), generator=torch.Generator().manual_seed(9))
    mask = _square_mask()

    plain = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    edge_loss = EdgeGuidedLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)), edge_weight=0.5, edge_soft=True)

    value = edge_loss(logits, mask, {"edge_logits": edge_logits})
    assert torch.isfinite(value)
    assert not torch.allclose(plain(logits, mask), value)


def test_edge_guided_loss_supports_scalar_pos_weight() -> None:
    logits = torch.randn((1, 2, 16, 16), generator=torch.Generator().manual_seed(10))
    edge_logits = torch.randn((1, 2, 16, 16), generator=torch.Generator().manual_seed(11))
    mask = _square_mask()[:1]

    edge_loss = EdgeGuidedLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)), edge_weight=0.5, edge_pos_weight=8.0)
    value = edge_loss(logits, mask, {"edge_logits": edge_logits})
    assert torch.isfinite(value)


def test_apdc_rpdc_zero_on_constant_and_respond_to_edges() -> None:
    x_const = torch.full((1, 1, 10, 10), 0.42)
    x_edge = torch.zeros((1, 1, 10, 10))
    x_edge[..., 5:] = 1.0
    for conv in (AngularDifferenceConv2d(1, 1), RadialDifferenceConv2d(1, 1)):
        out_const = conv(x_const)
        out_edge = conv(x_edge)
        assert torch.allclose(out_const[..., 2:-2, 2:-2], torch.zeros_like(out_const[..., 2:-2, 2:-2]), atol=1e-4)
        assert float(out_edge.abs().max()) > 1e-3


def test_rpdc_preserves_spatial_size() -> None:
    conv = RadialDifferenceConv2d(3, 5)
    out = conv(torch.randn(2, 3, 12, 12))
    assert out.shape == (2, 5, 12, 12)


def test_make_pdc_conv_dispatch() -> None:
    assert isinstance(make_pdc_conv("cpdc", 4, 8), CentralDifferenceConv2d)
    assert isinstance(make_pdc_conv("apdc", 4, 8), AngularDifferenceConv2d)
    assert isinstance(make_pdc_conv("rpdc", 4, 8), RadialDifferenceConv2d)
    assert isinstance(make_pdc_conv("vanilla", 4, 8), torch.nn.Conv2d)
    try:
        make_pdc_conv("bogus", 4, 8)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pdc_bank_and_edge_head_bank_shapes() -> None:
    bank = PDCBank(16, 24, ["cpdc", "apdc", "rpdc"])
    assert bank(torch.randn(2, 16, 12, 12)).shape == (2, 24, 12, 12)
    head = EdgeGuidedHead(in_channels=32, out_channels=2, edge_pdc_types=["cpdc", "apdc", "rpdc"])
    feat = torch.randn(2, 32, 12, 12)
    seg_logits, edge_logits = head(feat)
    assert seg_logits.shape == (2, 2, 12, 12) and edge_logits.shape == (2, 2, 12, 12)
    seg_logits.sum().backward()


def test_gac_head_shapes_and_degenerate() -> None:
    head = GeodesicActiveContourHead(in_channels=32, out_channels=2, iters=6)
    feat = torch.randn(2, 32, 24, 24)
    guide = torch.rand(2, 3, 24, 24)
    seg_logits, edge_logits = head(feat, guide)
    assert seg_logits.shape == (2, 2, 24, 24) and edge_logits.shape == (2, 2, 24, 24)
    assert torch.isfinite(seg_logits).all() and torch.isfinite(edge_logits).all()
    # guide=None -> 退化为普通分割头输出 (seg_out(seg_fuse(feat)))
    seg_none, _ = head(feat, None)
    expected = head.seg_out(head.seg_fuse(feat))
    assert torch.allclose(seg_none, expected, atol=1e-5)


def test_gac_head_is_differentiable_including_pde_params() -> None:
    head = GeodesicActiveContourHead(in_channels=16, out_channels=2, iters=8)
    feat = torch.randn(2, 16, 24, 24, requires_grad=True)
    guide = torch.rand(2, 3, 24, 24)
    seg_logits, edge_logits = head(feat, guide)
    target = (torch.rand(2, 2, 24, 24) > 0.5).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_logits, target)
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, target)
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    for param in (head.dt, head.beta, head.log_kappa):
        assert param.grad is not None and torch.isfinite(param.grad).all()

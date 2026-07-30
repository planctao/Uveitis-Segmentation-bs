from __future__ import annotations

import torch

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.dual_branch import CoreContourDualBranchHead, DualBranchLoss, source_core_target
from bs.multilabel import AsymmetricFocalTverskyBCE, masks_to_paper_targets


def _mask(batch: int = 2, size: int = 32) -> torch.Tensor:
    mask = torch.zeros((batch, size, size), dtype=torch.long)
    mask[:, 8:24, 8:24] = 1
    mask[0, 14:18, 14:18] = 2
    if batch > 1:
        mask[1, 15:17, 15:17] = 2
    return mask


def test_source_core_target_shape_range_and_fallback() -> None:
    target, valid = masks_to_paper_targets(_mask(batch=2, size=32))
    source = source_core_target(target, valid.expand_as(target), erosion_kernel=9, soft_sigma=1.5)
    assert source.shape == target.shape
    assert 0.0 <= float(source.min()) <= float(source.max()) <= 1.0
    # 极小 lesion_2 腐蚀后应回退，不能整通道消失
    assert bool((source[:, 1].flatten(1).sum(dim=1) > 0).all())


def test_source_core_target_hard_core_is_subset_for_large_regions() -> None:
    target, valid = masks_to_paper_targets(_mask(batch=1, size=32))
    source = source_core_target(target, valid.expand_as(target), erosion_kernel=5, soft_sigma=0.0)
    assert torch.all(source <= target + 1e-6)
    assert float(source[:, 0].sum()) < float(target[:, 0].sum())


def test_dual_branch_head_outputs_shapes_and_finite() -> None:
    head = CoreContourDualBranchHead(32, out_channels=2, edge_pdc_types=["cpdc", "apdc", "rpdc"])
    feat = torch.randn(2, 32, 16, 16)
    seg_logits, aux = head(feat)
    assert seg_logits.shape == (2, 2, 16, 16)
    assert aux["source_logits"].shape == (2, 2, 16, 16)
    assert aux["edge_logits"].shape == (2, 2, 16, 16)
    assert torch.isfinite(seg_logits).all()
    assert torch.isfinite(aux["source_logits"]).all()
    assert torch.isfinite(aux["edge_logits"]).all()


def test_dual_branch_gates_zero_initialized() -> None:
    head = CoreContourDualBranchHead(16, out_channels=2)
    source = torch.rand(2, 2, 8, 8)
    edge_feat = torch.rand(2, head.edge_gate.in_channels, 8, 8)
    assert torch.allclose(head.source_gate(source), torch.zeros(2, 16, 8, 8))
    assert torch.allclose(head.edge_gate(edge_feat), torch.zeros(2, 16, 8, 8))


def test_dual_branch_loss_without_aux_matches_segmentation_loss() -> None:
    logits = torch.randn(2, 2, 32, 32)
    mask = _mask()
    base = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    loss = DualBranchLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)))
    assert torch.allclose(base(logits, mask), loss(logits, mask, None))


def test_dual_branch_loss_weight_zero_matches_segmentation_loss() -> None:
    logits = torch.randn(2, 2, 32, 32)
    aux = {
        "source_logits": torch.randn(2, 2, 32, 32),
        "edge_logits": torch.randn(2, 2, 32, 32),
    }
    mask = _mask()
    base = AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0))
    loss = DualBranchLoss(
        AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)),
        edge_weight=0.0,
        source_weight=0.0,
        consistency_weight=0.0,
    )
    assert torch.allclose(base(logits, mask), loss(logits, mask, aux))


def test_dual_branch_loss_finite_and_differentiable() -> None:
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    aux = {
        "source_logits": torch.randn(2, 2, 32, 32, requires_grad=True),
        "edge_logits": torch.randn(2, 2, 32, 32, requires_grad=True),
    }
    mask = _mask()
    loss_fn = DualBranchLoss(AsymmetricFocalTverskyBCE(pos_weight=(1.0, 1.0)))
    value = loss_fn(logits, mask, aux)
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert aux["source_logits"].grad is not None and torch.isfinite(aux["source_logits"].grad).all()
    assert aux["edge_logits"].grad is not None and torch.isfinite(aux["edge_logits"].grad).all()


def test_convnext_decoder_dual_branch_train_and_eval() -> None:
    decoder = ConvNeXtFPNDecoder(
        in_channels=[96, 192, 384, 768],
        head_type="dual_branch",
        edge_pdc_types=["cpdc", "apdc", "rpdc"],
    )
    features = [
        torch.randn(1, 96, 48, 48),
        torch.randn(1, 192, 24, 24),
        torch.randn(1, 384, 12, 12),
        torch.randn(1, 768, 6, 6),
    ]
    decoder.train()
    logits, aux = decoder(features, (192, 192))
    assert logits.shape == (1, 2, 192, 192)
    assert aux["source_logits"].shape == (1, 2, 192, 192)
    assert aux["edge_logits"].shape == (1, 2, 192, 192)
    decoder.eval()
    out = decoder(features, (192, 192))
    assert out.shape == (1, 2, 192, 192)


from bs.dual_branch import RdhDualBranchFusionHead


def test_rdh_dual_branch_fusion_head_zero_residual_matches_rdh() -> None:
    head = RdhDualBranchFusionHead(
        16,
        out_channels=2,
        edge_pdc_types=["cpdc", "apdc", "rpdc"],
        rdh_kwargs={"iters": 4, "use_image_conductance": False},
    )
    feat = torch.randn(2, 16, 12, 12)
    logits, aux = head(feat, None)
    reference = head.rdh_head(feat, None)
    assert torch.allclose(logits, reference, atol=1e-6)
    assert aux["source_logits"].shape == (2, 2, 12, 12)
    assert aux["edge_logits"].shape == (2, 2, 12, 12)
    assert aux["dual_logits"].shape == (2, 2, 12, 12)
    assert aux["rdh_logits"].shape == (2, 2, 12, 12)


def test_rdh_dual_branch_fusion_head_residual_learns() -> None:
    head = RdhDualBranchFusionHead(8, out_channels=2, rdh_kwargs={"iters": 2, "use_image_conductance": False})
    feat = torch.randn(2, 8, 10, 10)
    logits, _ = head(feat, None)
    target = (torch.rand_like(logits) > 0.5).float()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
    assert head.residual_scale.grad is not None and torch.isfinite(head.residual_scale.grad).all()


def test_convnext_decoder_rdh_dual_branch_train_and_eval() -> None:
    decoder = ConvNeXtFPNDecoder(
        in_channels=[96, 192, 384, 768],
        head_type="rdh_dual_branch",
        edge_pdc_types=["cpdc", "apdc", "rpdc"],
        rdh_iters=3,
        rdh_use_image_conductance=False,
    )
    features = [
        torch.randn(1, 96, 48, 48),
        torch.randn(1, 192, 24, 24),
        torch.randn(1, 384, 12, 12),
        torch.randn(1, 768, 6, 6),
    ]
    decoder.train()
    logits, aux = decoder(features, (192, 192))
    assert logits.shape == (1, 2, 192, 192)
    assert aux["source_logits"].shape == (1, 2, 192, 192)
    assert aux["edge_logits"].shape == (1, 2, 192, 192)
    decoder.eval()
    out = decoder(features, (192, 192))
    assert out.shape == (1, 2, 192, 192)

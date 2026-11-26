import pytest
import torch

from bs.paper_methods_unet import (
    ConDSegUNet2D,
    MedCAGDUNet2D,
    MKUNet2D,
    PBEUNet2D,
    PUUNet2D,
    SpectralMambaUNet2D,
    UDMambaUNet2D,
)
from bs.recent_unet_methods import DifferenceGatedUNet2D, EALiteUNet2D, ProSMAUNet2D


@pytest.mark.parametrize(
    "model_cls",
    [ConDSegUNet2D, PBEUNet2D, MedCAGDUNet2D, UDMambaUNet2D, SpectralMambaUNet2D, PUUNet2D, MKUNet2D],
)
def test_recent_paper_unet_forward_and_auxiliary_supervision(model_cls):
    model = model_cls(features_per_stage=(8, 16, 24, 32, 40, 40))
    images = torch.randn(1, 3, 128, 128)
    masks = torch.zeros(1, 128, 128, dtype=torch.long)
    masks[:, 40:80, 40:80] = 1
    logits = model(images)
    assert logits.shape == (1, 2, 128, 128)
    loss = model.auxiliary_loss(
        masks,
        segmentation_weight=0.1,
        boundary_weight=0.1,
        uncertainty_weight=0.1,
    )
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (ProSMAUNet2D, {}),
        (DifferenceGatedUNet2D, {"mode": "fdg"}),
        (DifferenceGatedUNet2D, {"mode": "edg"}),
    ],
)
def test_next_priority_paper_unets_forward_backward(model_cls, kwargs):
    model = model_cls(features_per_stage=(8, 16, 24, 32, 40, 40), **kwargs)
    images = torch.randn(1, 3, 128, 128, requires_grad=True)
    logits = model(images)
    assert logits.shape == (1, 2, 128, 128)
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_prosma_has_exact_sparse_selection_signal():
    model = ProSMAUNet2D(features_per_stage=(8, 16, 24, 32, 40, 40))
    with torch.no_grad():
        model(torch.randn(1, 3, 128, 128))
    assert len(model.last_sparsity) == 5
    assert all(0.0 <= float(value) <= 1.0 for value in model.last_sparsity)
    assert any(float(value) > 0.0 for value in model.last_sparsity)


def test_ea_liteunet_boundary_modules_forward_backward():
    model = EALiteUNet2D(channels=(8, 12, 16, 20, 24, 32))
    # The paper applies AA-Down only on E1->E2 and E2->E3; deeper stages use
    # max-pooling for semantic abstraction.
    assert [index for index, block in enumerate(model.downsample) if block.__class__.__name__ == "_AntiAliasingDown"] == [1, 2]
    images = torch.randn(1, 3, 128, 128, requires_grad=True)
    masks = torch.zeros(1, 128, 128, dtype=torch.long)
    masks[:, 32:92, 36:84] = 1
    masks[:, 60:104, 68:112] = 2
    from bs.paper_reproduction import PaperSegmentationLoss

    logits = model(images)
    criterion = PaperSegmentationLoss(smooth=1e-5)
    loss = model.training_loss(criterion(logits, masks), images, masks, criterion)
    loss.backward()
    assert logits.shape == (1, 2, 128, 128)
    assert len(model.last_boundary_logits) == 5
    assert torch.isfinite(loss)

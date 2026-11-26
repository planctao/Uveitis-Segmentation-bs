import math

import pytest
import torch

from bs.bidirectional_component_supervision import bidirectional_component_terms


def test_probability_mass_score_has_detached_weights_in_logit_gradient():
    p = torch.tensor([[[.8, .6]]])
    logits = torch.logit(p).requires_grad_()
    terms = bidirectional_component_terms(logits, torch.zeros_like(p, dtype=torch.bool), alpha=1)
    loss = terms["instance"]
    expected = (p.square().sum() / p.sum()) - (1 - p).log().mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    expected_gradient = (p / p.sum()) * p * (1 - p) + p / p.numel()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert terms["gt_components"] == 0 and terms["pred_components"] == 1
    assert terms["GT_empty_pred_cells"] == 1


def test_smoothed_dice_control_keeps_ce_and_loses_the_extra_FP_gradient():
    logits = torch.full((1, 1, 20), torch.logit(torch.tensor(.5)).item() + .1, requires_grad=True)
    target = torch.zeros_like(logits, dtype=torch.bool)
    a = bidirectional_component_terms(logits, target, alpha=1)["instance"]
    b = bidirectional_component_terms(logits, target, alpha=1, pred_empty_cell_loss="dice", pred_smooth=1e-5)["instance"]
    ga = torch.autograd.grad(a, logits, retain_graph=True)[0]
    gb = torch.autograd.grad(b, logits)[0]
    assert ga.sum() - gb.sum() > .2
    assert gb.sum() == pytest.approx(logits.detach().sigmoid().mean().item(), abs=1e-6)


def test_empty_prediction_and_GT_preserve_author_whole_patch_fallback():
    logits = torch.full((2, 3, 4), torch.logit(torch.tensor(.1)).item(), requires_grad=True)
    terms = bidirectional_component_terms(logits, torch.zeros_like(logits, dtype=torch.bool))
    assert terms["gt_components"] == terms["pred_components"] == 0
    assert terms["instance"].item() == pytest.approx(1 - math.log(.9))
    terms["instance"].backward()
    torch.testing.assert_close(logits.grad, torch.full_like(logits, .1 / logits.numel()))


def test_batch_instances_are_independent_and_diagonal_pixels_use_eight_connectivity():
    logits = torch.full((2, 5, 5), -2.)
    target = torch.zeros_like(logits, dtype=torch.bool)
    target[:, 0, 0] = target[:, 1, 1] = target[:, 4, 4] = True
    terms = bidirectional_component_terms(logits, target)
    assert terms["gt_components_per_image"] == [2, 2]
    assert terms["gt_components"] == 4


def test_ignore_excludes_geometry_loss_and_gradients_and_fully_unknown_images():
    logits = torch.tensor([[[1., -1., 20.]]], requires_grad=True)
    foreground = torch.tensor([[[1, 0, 1]]], dtype=torch.bool)
    valid = torch.tensor([[[1, 1, 0]]], dtype=torch.bool)
    terms = bidirectional_component_terms(logits, foreground, valid_mask=valid)
    assert terms["gt_components"] == terms["pred_components"] == 1
    terms["instance"].backward()
    assert logits.grad[0, 0, 2] == 0
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 0, 1] > 0
    logits = logits.detach().requires_grad_()
    loss = bidirectional_component_terms(logits, foreground, valid_mask=torch.zeros_like(valid))["instance"]
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_no_added_network_parameters_and_exact_baseline_initial_inference():
    from bs.aaag_unet import build_nnunet_2d
    from bs.bidirectional_component_unet import BidirectionalComponentUNet2D
    widths = (4, 8, 12, 16, 20, 20)
    torch.manual_seed(42)
    base = build_nnunet_2d(3, 2, widths, deep_supervision=False).eval()
    torch.manual_seed(42)
    model = BidirectionalComponentUNet2D(features_per_stage=widths).eval()
    assert model.state_dict().keys() == base.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        assert torch.equal(model(images), base(images))
    assert model._cached_logits is None


def test_overlap_target_reaches_only_lesion2_output_directly():
    from bs.bidirectional_component_unet import BidirectionalComponentUNet2D
    model = BidirectionalComponentUNet2D(features_per_stage=(4, 8, 12, 16, 20, 20)).train()
    images = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[0, 4:7, 4:7] = 3
    target[1, 4:7, 4:7] = 2
    logits = model(images)
    model.training_loss(logits.sum() * 0, images, target, None).backward()
    grad = model.decoder.seg_layers[-1].weight.grad
    assert grad[0].abs().sum() == 0 and grad[1].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.instance_statistics()["gt_components"] == 2
    assert model._cached_logits is None

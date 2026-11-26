import math

import pytest
import torch

from bs.component_supervision import component_detection_terms


def test_per_image_components_do_not_connect_across_batch_and_face_connectivity():
    probabilities = torch.full((2, 3, 3), .5, requires_grad=True)
    foreground = torch.zeros_like(probabilities, dtype=torch.bool)
    foreground[:, 0, 0] = True
    foreground[:, 1, 1] = True
    face = component_detection_terms(probabilities, foreground, connectivity=1)
    diagonal = component_detection_terms(probabilities, foreground, connectivity=2)
    assert face["component_count"] == 4
    assert face["component_sizes"] == [1, 1, 1, 1]
    assert diagonal["component_count"] == 2
    assert diagonal["component_sizes"] == [2, 2]


def test_cat_matches_explicit_component_weighted_tversky():
    p = torch.tensor([[[.2, .6, .1, .3, .8, .9]]], requires_grad=True)
    y = torch.tensor([[[1, 1, 0, 0, 1, 1]]], dtype=torch.bool)
    terms = component_detection_terms(p, y)
    weights = torch.tensor([[[1 / 7, 1 / 7, .01, .01, 1 / 7, 1 / 7]]])
    tp = (weights * p * y).sum()
    fp = (weights * p * ~y).sum()
    fn = (weights * (1 - p) * y).sum()
    expected = 1 - (tp + 1e-5) / (tp + .3 * fp + .7 * fn + 1e-5)
    torch.testing.assert_close(terms["cat"], expected)
    assert terms["component_sizes"] == [2, 2]


def test_mil_max_score_and_gradient_are_equal_per_bag_independent_of_size():
    p = torch.tensor([[[.8, .1, .1, .2, .3, .8, .5]]], requires_grad=True)
    y = torch.tensor([[[1, 0, 0, 1, 1, 1, 1]]], dtype=torch.bool)
    terms = component_detection_terms(p, y)
    assert terms["component_sizes"] == [1, 4]
    assert terms["mil"].item() == pytest.approx(-math.log(.8))
    terms["mil"].backward()
    assert p.grad[0, 0, 0].item() == pytest.approx(-1 / (2 * .8))
    assert p.grad[0, 0, 5].item() == pytest.approx(-1 / (2 * .8))
    assert torch.count_nonzero(p.grad).item() == 2


def test_ignored_pixels_have_zero_auxiliary_gradient():
    p = torch.tensor([[[.6, .2, .9, .4]]], requires_grad=True)
    y = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool)
    valid = torch.tensor([[[1, 1, 0, 1]]], dtype=torch.bool)
    terms = component_detection_terms(p, y, valid_mask=valid)
    (terms["cat"] + terms["mil"]).backward()
    assert p.grad[0, 0, 2].item() == 0
    assert p.grad[0, 0, 0].item() < 0
    assert p.grad[0, 0, 1].item() > 0


def test_empty_foreground_is_finite_zero_and_mil_clipping_is_explicit():
    p = torch.full((2, 4, 4), .2, requires_grad=True)
    terms = component_detection_terms(p, torch.zeros_like(p, dtype=torch.bool))
    (terms["cat"] + terms["mil"]).backward()
    assert terms["component_count"] == 0
    assert torch.equal(p.grad, torch.zeros_like(p))
    p = torch.tensor([[[1e-8, .2]]], requires_grad=True)
    terms = component_detection_terms(p, torch.tensor([[[1, 0]]], dtype=torch.bool))
    terms["mil"].backward()
    assert terms["mil"].item() == pytest.approx(-math.log(1e-6))
    assert torch.equal(p.grad, torch.zeros_like(p))


def test_model_preserves_initial_state_inference_and_zero_based_warmup():
    from bs.aaag_unet import build_nnunet_2d
    from bs.component_supervised_unet import ComponentSupervisedUNet2D
    widths = (4, 8, 12, 16, 20, 20)
    torch.manual_seed(42)
    reference = build_nnunet_2d(3, 2, widths, deep_supervision=False).eval()
    torch.manual_seed(42)
    model = ComponentSupervisedUNet2D(features_per_stage=widths).eval()
    assert reference.state_dict().keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in reference.state_dict().items())
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        assert torch.equal(reference(images), model(images))
    assert model._cached_logits is None
    assert model.current_cat_weight == 0
    model.epoch = 6
    assert model.current_cat_weight == pytest.approx(.05)
    model.epoch = 11
    assert model.current_cat_weight == pytest.approx(.1)


def test_auxiliary_preserves_overlap_and_only_directly_supervises_lesion2_head():
    from bs.component_supervised_unet import ComponentSupervisedUNet2D
    model = ComponentSupervisedUNet2D(features_per_stage=(4, 8, 12, 16, 20, 20), mode="mil").train()
    images = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 64, 64, dtype=torch.long)
    mask[0, 4:7, 4:7] = 3
    mask[1, 4:7, 4:7] = 2
    logits = model(images)
    loss = model.training_loss(logits.sum() * 0., images, mask, None)
    loss.backward()
    gradient = model.decoder.seg_layers[-1].weight.grad
    assert gradient[0].abs().sum() == 0
    assert gradient[1].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.component_statistics()["lesion2_components"] == 2
    assert model._cached_logits is None

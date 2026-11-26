import pytest
import torch

from bs.prototype_regularization import prototype_discriminative_terms


def test_printed_equations_match_explicit_centroids_and_covariance_trace():
    features = torch.tensor([[[[0., 2., 4., 6.]], [[1., 3., 4., 8.]]]], requires_grad=True)
    mask = torch.tensor([[[0, 0, 2, 2]]])
    terms = prototype_discriminative_terms(features, mask, margin=10.)
    a = torch.tensor([[0., 1.], [2., 3.]])
    b = torch.tensor([[4., 4.], [6., 8.]])
    expected_sep = 10. - torch.linalg.vector_norm(a.mean(0) - b.mean(0))
    expected_comp = (torch.cov(a.T).trace() + torch.cov(b.T).trace()) / 2
    torch.testing.assert_close(terms["separation"], expected_sep)
    torch.testing.assert_close(terms["compactness"], expected_comp)
    (terms["separation"] + terms["compactness"]).backward()
    assert torch.isfinite(features.grad).all()


def test_overlap_is_a_separate_palette_state_and_ignore_is_excluded():
    features = torch.randn(1, 3, 1, 9, requires_grad=True)
    mask = torch.tensor([[[0, 0, 1, 1, 2, 2, 3, 3, 255]]])
    terms = prototype_discriminative_terms(features, mask)
    assert terms["eligible_classes"].tolist() == [0, 1, 2, 3]
    assert terms["eligible_pixel_counts"].tolist() == [2, 2, 2, 2]
    (terms["separation"] + terms["compactness"]).backward()
    assert features.grad[..., -1].abs().sum() == 0


@pytest.mark.parametrize("labels", [[255, 255], [1, 255]])
def test_empty_or_single_pixel_classes_have_finite_zero_loss(labels):
    feature = torch.randn(1, 4, 1, 2, requires_grad=True)
    terms = prototype_discriminative_terms(feature, torch.tensor([[labels]]))
    total = terms["separation"] + terms["compactness"]
    assert total.item() == 0
    total.backward()
    assert torch.equal(feature.grad, torch.zeros_like(feature))


def test_unit_normalization_adaptation_is_invariant_to_positive_feature_scaling():
    feature = torch.randn(1, 4, 2, 3)
    mask = torch.tensor([[[0, 0, 1], [1, 2, 2]]])
    a = prototype_discriminative_terms(feature, mask, normalize_features=True)
    b = prototype_discriminative_terms(feature * 17., mask, normalize_features=True)
    torch.testing.assert_close(a["separation"], b["separation"])
    torch.testing.assert_close(a["compactness"], b["compactness"])


def test_grid_mismatch_requires_explicit_target_strategy():
    with pytest.raises(ValueError, match="grids must match"):
        prototype_discriminative_terms(torch.randn(1, 4, 2, 2), torch.zeros(1, 4, 4, dtype=torch.long))


def test_model_preserves_all_baseline_state_and_exact_inference():
    from bs.aaag_unet import build_nnunet_2d
    from bs.prototype_regularized_unet import PrototypeRegularizedUNet2D
    features = (4, 8, 12, 16, 20, 20)
    torch.manual_seed(42)
    baseline = build_nnunet_2d(3, 2, features, deep_supervision=False).eval()
    torch.manual_seed(42)
    model = PrototypeRegularizedUNet2D(features_per_stage=features).eval()
    assert baseline.state_dict().keys() == model.state_dict().keys()
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name])
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(image), baseline(image))
    assert model._cached_feature is None


def test_model_auxiliary_loss_has_finite_encoder_and_head_gradients():
    from bs.prototype_regularized_unet import PrototypeRegularizedUNet2D
    model = PrototypeRegularizedUNet2D(features_per_stage=(4, 8, 12, 16, 20, 20)).train()
    image = torch.randn(2, 3, 64, 64)
    mask = torch.randint(0, 4, (2, 64, 64))
    logits = model(image)
    loss = model.training_loss(logits.square().mean(), image, mask, None)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.decoder.seg_layers[-1].weight.grad.abs().sum() > 0
    assert model._cached_feature is None
    assert model.pdr_statistics()["batches_with_eligible_palette_class"] == {"0": 1, "1": 1, "2": 1, "3": 1}

import pytest
import torch

from bs.component_objective import TrainingComponentObjective
from bs.component_supervised_unet import ComponentSupervisedUNet2D
from bs.multilabel import AsymmetricFocalTverskyBCE


def image_aware_loss():
    return AsymmetricFocalTverskyBCE(pos_weight=[3., 60.], vsubr_weight=.1, vsubr_kernel=5)


def test_image_aware_main_loss_is_preserved_at_cat_epoch1_and_during_validation():
    torch.manual_seed(42)
    logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    images = torch.rand(2, 3, 16, 16)
    masks = torch.zeros(2, 16, 16, dtype=torch.long)
    masks[0, 4:8, 4:8] = 3
    base = image_aware_loss()
    objective = TrainingComponentObjective(base)
    assert objective.needs_image
    expected = base(logits, masks, image=images)
    objective.enabled = True
    assert torch.equal(objective(logits, masks, image=images), expected)
    objective.epoch = 11
    trained = objective(logits, masks, image=images)
    assert trained > expected
    statistics = objective.statistics()
    objective.enabled = False
    validated = objective(logits, masks, image=images)
    assert torch.equal(validated, expected)
    assert objective.statistics() == statistics
    torch.testing.assert_close(torch.autograd.grad(validated, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["cat", "mil", "joint"])
def test_loss_wrapper_matches_existing_unet_objective_and_overlap_ignore_gradients(mode):
    model = ComponentSupervisedUNet2D(features_per_stage=(4, 8, 12, 16, 20, 20), mode=mode)
    model.epoch = 11
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    images = torch.rand(2, 3, 8, 8)
    masks = torch.zeros(2, 8, 8, dtype=torch.long)
    masks[0, 1:4, 1:4] = 3
    masks[1, 1, 1] = 2
    masks[:, 0] = 255
    base = image_aware_loss()
    objective = TrainingComponentObjective(base, mode=mode)
    objective.enabled, objective.epoch = True, 11
    base_loss = base(logits, masks, image=images)
    model._cached_logits = logits
    reference = model.training_loss(base_loss, images, masks, base)
    actual = objective(logits, masks, image=images)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    main_grad = torch.autograd.grad(base_loss, logits)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
    auxiliary_grad = actual_grad - main_grad
    assert torch.equal(auxiliary_grad[:, 0], torch.zeros_like(auxiliary_grad[:, 0]))
    assert torch.equal(auxiliary_grad[:, :, 0], torch.zeros_like(auxiliary_grad[:, :, 0]))
    assert auxiliary_grad[0, 1, 1:4, 1:4].abs().sum() > 0
    assert objective.statistics() == model.component_statistics()

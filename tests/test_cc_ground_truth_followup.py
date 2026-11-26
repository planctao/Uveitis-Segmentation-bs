import torch

from bs.bidirectional_component_supervision import bidirectional_component_terms
from bs.cc_ground_truth_followup import CCGroundTruthFollowupUNet, whole_image_dicece
from bs.bidirectional_component_unet import BidirectionalComponentUNet2D


def test_single_component_and_empty_images_match_regional_fallback_gradients():
    torch.manual_seed(5)
    logits = torch.randn(3, 7, 9, requires_grad=True)
    target = torch.zeros_like(logits, dtype=torch.bool)
    target[0, 2:4, 3:5] = True
    target[1, 1:5, 2:6] = True
    valid = torch.ones_like(target)
    valid[:, 0] = False
    regional = bidirectional_component_terms(logits, target, valid_mask=valid, alpha=0)["gt"]
    global_loss = whole_image_dicece(logits, target, valid)
    torch.testing.assert_close(global_loss, regional, rtol=1e-6, atol=1e-7)
    a = torch.autograd.grad(regional, logits, retain_graph=True)[0]
    b = torch.autograd.grad(global_loss, logits)[0]
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(b[:, 0]) == 0


def test_multiple_components_change_equal_region_weighting():
    logits = torch.full((1, 11, 19), -3., requires_grad=True)
    target = torch.zeros_like(logits, dtype=torch.bool)
    target[0, 2:7, 2:7] = True
    target[0, 5, 15] = True
    with torch.no_grad():
        logits[0, 2:7, 2:7] = 3
    valid = torch.ones_like(target)
    regional = bidirectional_component_terms(logits, target, valid_mask=valid, alpha=0)["gt"]
    global_loss = whole_image_dicece(logits, target, valid)
    assert regional > global_loss
    regional_gradient = torch.autograd.grad(regional, logits, retain_graph=True)[0]
    global_gradient = torch.autograd.grad(global_loss, logits)[0]
    assert regional_gradient[0, 5, 15].abs() > global_gradient[0, 5, 15].abs()


def test_all_ignored_loss_and_gradients_zero():
    logits = torch.randn(2, 4, 5, requires_grad=True)
    loss = whole_image_dicece(logits, torch.ones_like(logits), torch.zeros_like(logits))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_gt_adapter_preserves_original_forward_loss_and_parameter_gradients():
    torch.set_num_threads(2)
    options = dict(features_per_stage=(4, 8, 16), weight=.25)
    torch.manual_seed(8)
    original = BidirectionalComponentUNet2D(**options, mode="gt_only").train()
    followup = CCGroundTruthFollowupUNet(**options, auxiliary_mode="gt_only").train()
    followup.load_state_dict(original.state_dict(), strict=True)
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    masks[0, 5:9, 6:10] = 3
    masks[0, 20:23, 20:23] = 2
    masks[:, :2] = 255
    losses = []
    for model in [original, followup]:
        logits = model(images)
        losses.append(model.training_loss(logits.square().mean(), images, masks, None))
        losses[-1].backward()
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for a, b in zip(original.parameters(), followup.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_checkpoint_replay_uses_followup_factory_and_unchanged_clean_inference():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.fusion_evaluation_components import evaluation_components
    config = {"model": {"name": "cc_ground_truth_followup_controlled",
                        "features_per_stage": [4, 8, 16],
                        "cc_ground_truth_followup": {"mode": "global_only", "weight": .25}},
              "data": {"ignore_index": 255}}
    factory = evaluation_components(config)[1]
    replay_model = factory(config).eval()
    assert isinstance(replay_model, CCGroundTruthFollowupUNet)
    original = BidirectionalComponentUNet2D(features_per_stage=(4, 8, 16)).eval()
    original.load_state_dict(replay_model.state_dict(), strict=True)
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(original(images), replay_model(images), rtol=0, atol=0)

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from bs.aaag_unet import build_nnunet_2d
from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints, seed_paired_epoch
from bs.recent_unet_methods import DifferenceGatedFusion, DifferenceGatedUNet2D


def test_equal_control_has_identical_branches_but_only_content_affects_output():
    fusion = DifferenceGatedFusion(8, "equal").eval()
    global_feature, local_feature = torch.randn(2, 8, 12, 12), torch.randn(2, 8, 12, 12)
    expected = (fusion.global_features(global_feature)[2] + fusion.local_features(local_feature)[2]) * 0.5
    actual = fusion(global_feature, local_feature)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert fusion.global_features.norm[2][0].weight.grad is not None
    assert fusion.global_features.norm[0][0].weight.grad is None
    with torch.no_grad():
        fusion.global_features.norm[0][0].weight.add_(100)
        fusion.local_features.norm[1][0].weight.mul_(-30)
    torch.testing.assert_close(fusion(global_feature, local_feature), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["fdg", "edg"])
def test_diagnostic_capture_does_not_change_logits_and_gates_are_complementary(mode):
    fusion = DifferenceGatedFusion(8, mode).eval()
    global_feature, local_feature = torch.randn(2, 8, 8, 8), torch.randn(2, 8, 16, 16)
    without_capture = fusion(global_feature, local_feature)
    assert not fusion.last_diagnostics
    fusion.capture_diagnostics = True
    with_capture = fusion(global_feature, local_feature)
    torch.testing.assert_close(with_capture, without_capture, rtol=0, atol=0)
    captured = fusion.last_diagnostics
    assert captured["global_weight"].shape == (2, 1, 16, 16)
    torch.testing.assert_close(captured["global_weight"] + captured["local_weight"], torch.ones(2, 1, 16, 16))
    assert all(not tensor.requires_grad for tensor in captured.values())


def test_paired_initialization_matches_all_fusion_weights_and_shared_concat_weights():
    features = (8, 16, 24, 32, 40, 40)
    config = {"runtime": {"paired_model_seed": 3407}}
    models = [initialize_paired_model(lambda _: DifferenceGatedUNet2D(features_per_stage=features, mode=mode), config)
              for mode in ("equal", "fdg", "edg")]
    fingerprints = [initial_weight_fingerprints(model) for model in models]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    concat = initialize_paired_model(lambda _: build_nnunet_2d(features_per_stage=features), config)
    concat_fingerprints = initial_weight_fingerprints(concat)
    paired = 0
    for name, fingerprint in concat_fingerprints.items():
        if name in fingerprints[0] and fingerprint["shape"] == fingerprints[0][name]["shape"]:
            assert fingerprint == fingerprints[0][name], name
            paired += 1
    assert paired > 100


def test_model_initialization_does_not_consume_callers_random_stream():
    config = {"runtime": {"paired_model_seed": 42}}
    before = torch.random.get_rng_state().clone()
    initialize_paired_model(lambda _: DifferenceGatedFusion(8), config)
    assert torch.equal(before, torch.random.get_rng_state())


def test_inactive_placement_uses_equal_fusion_with_same_parameterization():
    all_equal = DifferenceGatedUNet2D(features_per_stage=(8, 16, 24, 32, 40, 40), mode="equal").eval()
    placement = DifferenceGatedUNet2D(features_per_stage=(8, 16, 24, 32, 40, 40), mode="edg", gated_stages=[]).eval()
    placement.load_state_dict(all_equal.state_dict(), strict=True)
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(placement(images), all_equal(images), rtol=0, atol=0)
    with pytest.raises(ValueError, match="deep to shallow"):
        DifferenceGatedUNet2D(gated_stages=[5])


class RandomDataset(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), float(np.random.rand()), float(torch.rand(()))])


def paired_loader():
    sampler = WeightedRandomSampler(torch.ones(24), 24, replacement=True, generator=torch.Generator())
    return DataLoader(RandomDataset(), batch_size=4, sampler=sampler, num_workers=2,
                      generator=torch.Generator(), persistent_workers=False)


def test_paired_epoch_matches_sample_order_and_all_worker_rngs_after_resume():
    config = {"runtime": {"paired_data_seed": 2026}}
    first, first_val = paired_loader(), paired_loader()
    seed_paired_epoch(config, 7, first, first_val)
    expected = torch.cat(list(first))
    # A restarted model can consume arbitrarily many random numbers.
    torch.randn(1000)
    restarted, restarted_val = paired_loader(), paired_loader()
    seed_paired_epoch(config, 7, restarted, restarted_val)
    torch.testing.assert_close(torch.cat(list(restarted)), expected, rtol=0, atol=0)
    seed_paired_epoch(config, 8, restarted, restarted_val)
    assert not torch.equal(torch.cat(list(restarted)), expected)

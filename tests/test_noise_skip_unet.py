import torch

from bs.aaag_unet import build_nnunet_2d
from bs.noise_skip_unet import NoiseSkipUNet2D


FEATURES = (4, 8, 12, 16, 20, 20)


def models(mode="skip", bound=.5):
    torch.manual_seed(42)
    baseline = build_nnunet_2d(3, 2, FEATURES, deep_supervision=False)
    torch.manual_seed(42)
    candidate = NoiseSkipUNet2D(3, 2, FEATURES, mode=mode, bound=bound)
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in baseline.state_dict().items())
    return baseline, candidate


def test_clean_evaluation_original_logits_and_gradients_both_arms():
    for mode in ["skip", "encoder"]:
        baseline, candidate = models(mode)
        baseline.eval()
        candidate.eval()
        x = torch.randn(2, 3, 64, 64)
        a = x.clone().requires_grad_(True)
        b = x.clone().requires_grad_(True)
        y, z = baseline(a), candidate(b)
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        y.square().mean().backward()
        z.square().mean().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        assert candidate.injection_count == 0


def test_skip_noise_does_not_change_deep_encoder_features():
    baseline, candidate = models("skip")
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        before = baseline.encoder(x)
        recorded = []
        handle = candidate.encoder.register_forward_hook(lambda m, args, output: recorded.append(output))
        baseline_output = baseline(x)
        output = candidate(x)
        handle.remove()
    assert all(torch.equal(a, b) for a, b in zip(before, recorded[0]))
    assert not torch.equal(baseline_output, output)
    assert candidate.injection_count == 5


def test_encoder_noise_propagates_to_bottleneck():
    baseline, candidate = models("encoder")
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a, b = baseline.encoder(x), candidate.encoder(x)
    assert not torch.equal(a[-1], b[-1])
    assert candidate.injection_count == 5


def test_noise_is_bounded_zero_mean_uses_independent_rng_and_reseeds_epoch():
    _, candidate = models("skip")
    x = torch.zeros(1, 1, 512, 512)
    state = torch.random.get_rng_state().clone()
    first = candidate._perturb(x, 0)
    assert torch.equal(state, torch.random.get_rng_state())
    assert float(first.abs().max()) <= .5
    assert abs(float(first.mean())) < .002
    assert abs(float(first.var()) - 1 / 12) < .001
    second = candidate._perturb(x, 0)
    assert not torch.equal(first, second)
    candidate.set_epoch(1)
    assert torch.equal(first, candidate._perturb(x, 0))
    candidate.set_epoch(2)
    assert not torch.equal(first, candidate._perturb(x, 0))


def test_zero_bound_preserves_training_logits_and_gradient():
    baseline, candidate = models("encoder", bound=0.)
    x = torch.randn(2, 3, 64, 64)
    y, z = baseline(x), candidate(x)
    torch.testing.assert_close(y, z, rtol=0, atol=0)
    y.mean().backward()
    z.mean().backward()
    for (name, a), (_, b) in zip(baseline.named_parameters(), candidate.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

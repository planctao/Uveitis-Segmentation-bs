import torch

from bs.aaag_unet import build_nnunet_2d
from bs.depth_space_unet import DepthSpaceUNet2D, restore_subpixels


def test_subpixel_phase_information_survives_only_rearrangement():
    a = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1)
    b = torch.tensor([4., 3., 2., 1.]).reshape(1, 4, 1, 1)
    torch.testing.assert_close(restore_subpixels(a, "phase_mean_bilinear"),
                               restore_subpixels(b, "phase_mean_bilinear"), rtol=0, atol=0)
    assert not torch.equal(restore_subpixels(a, "depth_to_space"), restore_subpixels(b, "depth_to_space"))
    torch.testing.assert_close(torch.nn.functional.pixel_unshuffle(restore_subpixels(a, "depth_to_space"), 2), a)


def test_controlled_models_preserve_initialization_and_receive_gradients():
    features = (8, 16, 24, 32, 40, 40)
    torch.manual_seed(42)
    baseline = build_nnunet_2d(features_per_stage=features)
    models = []
    for mode in ("phase_mean_bilinear", "depth_to_space"):
        torch.manual_seed(42)
        models.append(DepthSpaceUNet2D(features_per_stage=features, restoration=mode))
    state = baseline.state_dict()
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(tensor, models[1].state_dict()[name], rtol=0, atol=0)
        if name in state:
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    images = torch.randn(2, 3, 64, 64)
    for model in models:
        output = model(images)
        assert output.shape == (2, 2, 64, 64)
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.transpconvs[0].parameters())

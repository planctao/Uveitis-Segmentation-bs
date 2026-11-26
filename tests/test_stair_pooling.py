import pytest
import torch

from bs.aaag_unet import build_nnunet_2d
from bs.stair_pooling import StairPool2D, StairPoolingUNet2D


def test_intermediate_stair_processing_distinguishes_information_lost_by_full_pooling():
    first = torch.tensor([[[[1., 0.], [0., 1.]]]])
    second = torch.tensor([[[[1., 1.], [0., 0.]]]])
    torch.testing.assert_close(torch.nn.functional.max_pool2d(first, 2), torch.nn.functional.max_pool2d(second, 2))
    stair, capacity = StairPool2D(1), StairPool2D(1, progressive=False)
    with torch.no_grad():
        for parameter in stair.parameters():
            parameter.fill_(1)
        capacity.load_state_dict(stair.state_dict(), strict=True)
        assert not torch.equal(stair(first), stair(second))
        torch.testing.assert_close(capacity(first), capacity(second), rtol=0, atol=0)


def test_stair_and_capacity_controls_pair_every_parameter_and_train_at_same_shapes():
    torch.manual_seed(42)
    stair = StairPoolingUNet2D(features_per_stage=(8, 16, 24, 32, 40, 40), downsampling="stair")
    torch.manual_seed(42)
    capacity = StairPoolingUNet2D(features_per_stage=(8, 16, 24, 32, 40, 40), downsampling="capacity")
    for name, parameter in stair.state_dict().items():
        torch.testing.assert_close(parameter, capacity.state_dict()[name], rtol=0, atol=0)
    image = torch.randn(2, 3, 64, 64)
    for model in (stair, capacity):
        logits = model(image)
        assert logits.shape == (2, 2, 64, 64)
        logits.square().mean().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.downsamplers.parameters())


def test_strided_control_preserves_existing_nnunet_checkpoint_and_forward():
    features = (8, 16, 24, 32, 40, 40)
    original = build_nnunet_2d(features_per_stage=features).eval()
    control = StairPoolingUNet2D(features_per_stage=features, downsampling="strided").eval()
    control.load_state_dict(original.state_dict(), strict=True)
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(original(image), control(image), rtol=0, atol=0)
    with pytest.raises(ValueError, match="encoder stage"):
        StairPoolingUNet2D(pooling_stages=[0])

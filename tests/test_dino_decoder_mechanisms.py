import copy

import torch

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.dino_decoder_mechanisms import MechanismFPNDecoder


def test_bilinear_transfer_preserves_original_fpn_predictions_and_checkpoint():
    original = ConvNeXtFPNDecoder([8, 16, 24, 32], decoder_channels=8).eval()
    transferred = MechanismFPNDecoder(copy.deepcopy(original)).eval()
    transferred.load_state_dict(original.state_dict(), strict=True)
    features = [torch.randn(2, channels, size, size) for channels, size in zip([8, 16, 24, 32], [16, 8, 4, 2])]
    with torch.no_grad():
        torch.testing.assert_close(original(features, (64, 64)), transferred(features, (64, 64)), rtol=0, atol=0)


def test_transferred_mechanisms_preserve_common_tensors_and_receive_gradients():
    original = ConvNeXtFPNDecoder([8, 16, 24, 32], decoder_channels=8)
    features = [torch.randn(2, channels, size, size) for channels, size in zip([8, 16, 24, 32], [16, 8, 4, 2])]
    for settings in ({"restoration": "depth_to_space"}, {"refinement_basis": "haar"}):
        head = MechanismFPNDecoder(copy.deepcopy(original), **settings)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, head.state_dict()[name], rtol=0, atol=0)
        output = head(features, (64, 64))
        output.square().mean().backward()
        active = head.upsamplers if head.upsamplers else head.refiners
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.parameters())

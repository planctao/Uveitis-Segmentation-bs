import pytest
import torch
from torch import nn

from bs.convnext_seg import MambaVisionFusion
from bs.model import TokenFPNHead


def _features() -> list[torch.Tensor]:
    return [torch.randn(2, 16, 8, 8) for _ in range(4)]


def test_token_fpn_defaults_to_no_decoder_attention() -> None:
    head = TokenFPNHead(
        in_channels=16,
        num_inputs=4,
        decoder_channels=8,
        num_classes=2,
    )

    assert isinstance(head.attention, nn.Identity)
    assert head(_features(), output_size=(32, 32)).shape == (2, 2, 32, 32)


def test_token_fpn_supports_hmvf_and_backpropagates() -> None:
    head = TokenFPNHead(
        in_channels=16,
        num_inputs=4,
        decoder_channels=8,
        num_classes=2,
        decoder_attention="mambavision",
    )

    output = head(_features(), output_size=(32, 32))
    output.mean().backward()

    assert isinstance(head.attention, MambaVisionFusion)
    assert output.shape == (2, 2, 32, 32)
    assert head.attention.out_proj.weight.grad is not None


def test_token_fpn_rejects_unsupported_decoder_attention() -> None:
    with pytest.raises(ValueError, match="Unsupported ViT decoder attention"):
        TokenFPNHead(
            in_channels=16,
            num_inputs=4,
            decoder_channels=8,
            num_classes=2,
            decoder_attention="unknown",
        )

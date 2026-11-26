import pytest
import torch

from bs.convnext_seg import (
    ChannelSpatialAttention,
    ConvNeXtFPNDecoder,
    MCADSRestorationHead,
    MambaVisionFusion,
    PEMambaVisionFusion,
    ProgressiveExpansionLite,
    ResidualLinearAttention2D,
    build_attention,
)


def _features() -> list[torch.Tensor]:
    return [
        torch.randn(2, 4, 32, 32),
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]


def test_convnext_decoder_default_attention_keeps_checkpoint_keys_clean() -> None:
    decoder = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2)

    logits = decoder(_features(), output_size=(64, 64))

    assert tuple(logits.shape) == (2, 2, 64, 64)
    assert not any(key.startswith("attention.") for key in decoder.state_dict())


def test_convnext_decoder_cbam_attention_preserves_output_shape() -> None:
    decoder = ConvNeXtFPNDecoder(
        [4, 8, 16, 32],
        decoder_channels=16,
        out_channels=2,
        attention="cbam",
        attention_reduction=4,
    )

    logits = decoder(_features(), output_size=(64, 64))

    assert tuple(logits.shape) == (2, 2, 64, 64)
    assert any(key.startswith("attention.") for key in decoder.state_dict())


def test_convnext_decoder_deep_supervision_returns_aux_logits_only_in_train() -> None:
    decoder = ConvNeXtFPNDecoder(
        [4, 8, 16, 32],
        decoder_channels=16,
        out_channels=2,
        deep_supervision=True,
    )
    decoder.train()

    main_logits, aux_logits = decoder(_features(), output_size=(64, 64))

    assert tuple(main_logits.shape) == (2, 2, 64, 64)
    assert len(aux_logits) == 3
    assert all(tuple(item.shape) == (2, 2, 64, 64) for item in aux_logits)

    decoder.eval()
    eval_logits = decoder(_features(), output_size=(64, 64))
    assert isinstance(eval_logits, torch.Tensor)
    assert tuple(eval_logits.shape) == (2, 2, 64, 64)


def test_channel_spatial_attention_returns_same_shape() -> None:
    attention = ChannelSpatialAttention(channels=12, reduction=4)
    x = torch.randn(2, 12, 8, 8)

    y = attention(x)

    assert tuple(y.shape) == tuple(x.shape)


def test_mambavision_fusion_is_initially_noop_and_backpropagates() -> None:
    module = MambaVisionFusion(channels=32, inner_channels=16, d_state=4, max_spatial_size=8)
    features = torch.randn(1, 32, 15, 13, requires_grad=True)

    output = module(features)

    assert output.shape == features.shape
    assert torch.allclose(output, features, atol=1e-6)
    output.square().mean().backward()
    assert module.out_proj.weight.grad is not None


def test_progressive_expansion_lite_is_initially_noop_and_backpropagates() -> None:
    module = ProgressiveExpansionLite(channels=32, bottleneck=32)
    features = torch.randn(1, 32, 15, 13, requires_grad=True)

    output = module(features)

    assert output.shape == features.shape
    assert torch.allclose(output, features, atol=1e-6)
    output.square().mean().backward()
    assert module.up.weight.grad is not None


def test_pe_hmvf_fusion_is_initially_noop() -> None:
    module = PEMambaVisionFusion(channels=32, bottleneck=32)
    features = torch.randn(1, 32, 15, 13)

    output = module(features)

    assert torch.allclose(output, features, atol=1e-6)


def test_residual_linear_attention_is_initially_noop() -> None:
    module = ResidualLinearAttention2D(channels=32, key_channels=8)
    features = torch.randn(2, 32, 11, 13, requires_grad=True)

    output = module(features)

    assert torch.allclose(output, features, atol=1e-6)
    output.square().mean().backward()
    assert module.output.weight.grad is not None


def test_mcads_restoration_head_uses_subpixel_output() -> None:
    module = MCADSRestorationHead(in_channels=64, out_channels=2)
    output = module(torch.randn(1, 64, 16, 16), output_size=(64, 64))
    assert output.shape == (1, 2, 64, 64)


def test_convnext_decoder_mcads_head_restores_requested_size() -> None:
    decoder = ConvNeXtFPNDecoder(
        [4, 8, 16, 32], decoder_channels=16, out_channels=2, head_type="mcads"
    )
    logits = decoder(_features(), output_size=(128, 128))
    assert logits.shape == (2, 2, 128, 128)


def test_build_attention_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported ConvNeXt decoder attention"):
        build_attention("unknown", channels=16)

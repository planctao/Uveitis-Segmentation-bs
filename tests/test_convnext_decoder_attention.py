import pytest
import torch

from bs.convnext_seg import ChannelSpatialAttention, ConvNeXtFPNDecoder, build_attention


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


def test_build_attention_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported ConvNeXt decoder attention"):
        build_attention("unknown", channels=16)

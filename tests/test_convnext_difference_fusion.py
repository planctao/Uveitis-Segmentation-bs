import torch

from bs.convnext_seg import ConvNeXtFPNDecoder


def test_fpn_migration_pairs_every_existing_weight_and_all_fusion_control_weights():
    models = {}
    for mode in ("sum", "equal", "fdg", "edg"):
        torch.manual_seed(2026)
        models[mode] = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, fusion_mode=mode).eval()
    default_state = models["sum"].state_dict()
    assert not any(name.startswith("top_down_fusions.") for name in default_state)
    for mode in ("equal", "fdg", "edg"):
        state = models[mode].state_dict()
        for name, weight in default_state.items():
            torch.testing.assert_close(weight, state[name], rtol=0, atol=0)
        for name, weight in models["equal"].state_dict().items():
            torch.testing.assert_close(weight, state[name], rtol=0, atol=0)


def test_edg_migration_trains_both_statistics_branches_and_content_at_each_scale():
    decoder = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, fusion_mode="edg")
    features = [torch.randn(2, c, s, s) for c, s in zip((4, 8, 16, 32), (32, 16, 8, 4))]
    logits = decoder(features, (64, 64))
    logits.square().mean().backward()
    for fusion in decoder.top_down_fusions:
        for stream in (fusion.global_features, fusion.local_features):
            for branch in stream.norm:
                gradient = branch[0].weight.grad
                assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

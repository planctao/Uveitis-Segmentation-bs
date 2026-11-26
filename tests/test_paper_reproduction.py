from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.fft import dctn

from bs.paper_reproduction import ConDSegAuthor, MKStructureLoss, PUFocalTverskyBCE, PaperSegmentationLoss, boundary_targets
from bs.pu_unet import PUUNet, ResidualBlock, StableProductConv2d
from bs.spectralmamba_unet import dct_2d, idct_2d, split_spectrum


def test_dct_matches_independent_scipy_and_inverse_preserves_energy():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 7, 10, dtype=torch.float64, requires_grad=True)
    actual = dct_2d(x)
    expected = dctn(x.detach().numpy(), axes=(-2, -1), norm="ortho")
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12, rtol=1e-12)
    restored = idct_2d(actual)
    torch.testing.assert_close(restored, x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.square().sum(), x.square().sum())
    restored.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_spectral_masks_are_complementary_and_low_band_contains_dc():
    f = torch.ones(1, 3, 64, 64)
    low, high = split_spectrum(f)
    torch.testing.assert_close(low + high, f)
    assert (low * high).count_nonzero() == 0
    assert low[0, 0, 0, 0] == 1
    assert abs(float(low[0, 0].mean()) - 0.125) < 0.005
    dc = dct_2d(torch.ones_like(f))
    low, high = split_spectrum(dc)
    assert high.abs().max() < 1e-5
    torch.testing.assert_close(idct_2d(low), f, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("loss", [PaperSegmentationLoss(), PUFocalTverskyBCE()])
def test_losses_ignore_pixels_and_all_ignored_gradients(loss):
    masks = torch.zeros(2, 9, 9, dtype=torch.long)
    masks[:, 2:4, 2:4] = 3
    masks[:, 0, :] = 255
    logits = torch.randn(2, 2, 9, 9, requires_grad=True)
    modified = logits.detach().clone()
    modified[:, :, 0, :] = 50
    torch.testing.assert_close(loss(logits, masks), loss(modified, masks))
    loss(logits, masks).backward()
    assert logits.grad[:, :, 0, :].count_nonzero() == 0
    assert logits.grad[:, :, 2:4, 2:4].abs().sum() > 0
    ignored = torch.full_like(masks, 255)
    zero_loss = loss(logits, ignored)
    assert zero_loss.item() == 0
    assert torch.isfinite(zero_loss)


def test_boundary_preserves_overlap_interface_and_excludes_invalid_neighbors():
    masks = torch.ones(1, 9, 9, dtype=torch.long)
    masks[:, 3:6, 3:6] = 3
    masks[:, 0, 0] = 255
    edges, valid = boundary_targets(masks)
    assert edges[0, 0, 3, 3] == 1  # boundary inside the foreground union
    assert edges[0, 0, 4, 4] == 0
    assert valid[0, 0, 0:2, 0:2].count_nonzero() == 0


def test_med_edges_match_author_zero_padded_erosion_at_frame_boundary():
    masks = torch.ones(1, 16, 16, dtype=torch.long)
    edges, valid = boundary_targets(masks, radius=3, zero_padded_erosion=True)
    assert edges[0, 0, :3].all()
    assert edges[0, 0, -3:].all()
    assert edges[0, 0, 3:-3, 3:-3].count_nonzero() == 0
    assert valid.all()


def test_pbe_ukan_per_image_dice_differs_from_pooling_entire_batch():
    logits = torch.zeros(2, 2, 2, 2)
    masks = torch.zeros(2, 2, 2, dtype=torch.long)
    masks[1] = 3
    actual = PaperSegmentationLoss(bce_weight=.5, per_image=True, smooth=1e-5)(logits, masks)
    expected_dice = 1 - (1e-5 / (2+1e-5) + (4+1e-5) / (6+1e-5)) / 2
    expected = torch.tensor(.5 * np.log(2) + expected_dice, dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
    pooled = PaperSegmentationLoss(bce_weight=.5, smooth=1e-5)(logits, masks)
    assert abs(float(actual-pooled)) > .1


def test_udmamba_configured_squared_dice_matches_author_sparse_target_ratio():
    from pathlib import Path

    import yaml
    from scripts.train_unet_multilabel import build_loss

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/paper_reproduction/udmamba.yaml").read_text())
    torch.manual_seed(22)
    logits = (torch.randn(2, 2, 8, 8) - 2).requires_grad_()
    masks = torch.zeros(2, 8, 8, dtype=torch.long)
    masks[:, 2:4, 2:4] = 1
    masks[0, 5, 5] = 3
    targets = torch.stack([masks.eq(1) | masks.eq(3), masks.eq(2) | masks.eq(3)], dim=1).float()
    # The author DiceLoss._dice_loss uses squared sums and smooth=1e-5.
    # Sparse targets expose a default-smooth regression that large masks hide.
    probabilities = logits.sigmoid()
    ratios = [(2 * (probabilities[:, c] * targets[:, c]).sum() + 1e-5)
              / (probabilities[:, c].square().sum() + targets[:, c].square().sum() + 1e-5)
              for c in range(2)]
    dice = 1 - torch.stack(ratios).mean()
    bce = -(targets * torch.nn.functional.logsigmoid(logits)
            + (1 - targets) * torch.nn.functional.logsigmoid(-logits)).mean()
    expected = .5 * (dice + bce)
    actual = build_loss(config)(logits, masks)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)


def test_mk_structure_loss_matches_author_binary_formula_without_ignore():
    import torch.nn.functional as F
    torch.manual_seed(10)
    masks = torch.zeros(2, 40, 40, dtype=torch.long)
    masks[:, 8:20, 8:20] = 3
    target = masks.eq(3).float().unsqueeze(1)
    single_logits = torch.randn(2, 1, 40, 40)
    weit = 1 + 5 * (F.avg_pool2d(target, 31, 1, 15) - target).abs()
    wbce = (weit * F.binary_cross_entropy_with_logits(single_logits, target, reduction="none")).sum((2,3)) / weit.sum((2,3))
    pred = single_logits.sigmoid()
    inter = (pred * target * weit).sum((2,3))
    union = ((pred + target) * weit).sum((2,3))
    expected = (wbce + 1 - (inter+1)/(union-inter+1)).mean()
    torch.testing.assert_close(MKStructureLoss()(single_logits.repeat(1,2,1,1), masks), expected)


def test_ukan_optimizer_keeps_author_spline_learning_rate():
    from bs.paper_reproduction import UKANAuthor
    from scripts.train_unet_multilabel import build_optimizer
    model = UKANAuthor()
    config = {"train": {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 1e-4,
                         "kan_learning_rate": .01, "kan_weight_decay": 1e-4}}
    optimizer = build_optimizer(model, config)
    groups = optimizer.param_groups
    assert [g["lr"] for g in groups] == [1e-4, .01]
    kan_ids = {id(p) for p in groups[1]["params"]}
    expected = {id(p) for name, p in model.named_parameters() if "layer" in name.lower() and "fc" in name.lower()}
    assert kan_ids == expected and kan_ids
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_ukan_formal_config_and_fallback_use_paper_base_not_large_constructor():
    from pathlib import Path

    import yaml
    from bs.paper_reproduction import UKANAuthor
    from scripts.train_unet_multilabel import build_model

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/paper_reproduction/ukan.yaml").read_text())
    assert config["model"]["embed_dims"] == [128, 160, 256]
    config["model"].pop("embed_dims")
    fallback = build_model(config)
    direct = UKANAuthor()
    for model in (fallback, direct):
        assert model.core.patch_embed3.proj.in_channels == 128
        assert model.core.patch_embed3.proj.out_channels == 160
        assert model.core.patch_embed4.proj.out_channels == 256
        assert model.core.final.out_channels == 2


def test_condseg_fine_tuning_preserves_encoder_lr_and_updates_both_groups():
    from pathlib import Path

    import yaml
    from scripts.train_unet_multilabel import build_optimizer, build_scheduler

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/paper_reproduction/condseg.yaml").read_text())
    model = ConDSegAuthor(pretrained=False, stage=2)
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.Adam)
    encoder_ids = {id(p) for i in range(4) for p in getattr(model.core, f"layer{i}").parameters()}
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == encoder_ids
    remaining_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert not encoder_ids & remaining_ids
    assert encoder_ids | remaining_ids == {id(p) for p in model.parameters()}
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(group["weight_decay"] == 0 for group in optimizer.param_groups)

    encoder_parameter = model.core.layer0[0].weight
    decoder_parameter = model.core.output_block.c3.weight
    encoder_before, decoder_before = encoder_parameter.detach().clone(), decoder_parameter.detach().clone()
    encoder_parameter.grad = torch.ones_like(encoder_parameter)
    decoder_parameter.grad = torch.ones_like(decoder_parameter)
    scheduler = build_scheduler(optimizer, config)
    optimizer.step()
    torch.testing.assert_close(encoder_before - encoder_parameter, torch.full_like(encoder_parameter, 1e-5), atol=3e-8, rtol=1e-3)
    torch.testing.assert_close(decoder_before - decoder_parameter, torch.full_like(decoder_parameter, 1e-4), atol=3e-8, rtol=1e-3)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(config["train"]["epochs"]):
        optimizer.step()
        scheduler.step()
        encoder_lr, decoder_lr = [group["lr"] for group in optimizer.param_groups]
        assert encoder_lr / decoder_lr == pytest.approx(.1)
    assert encoder_lr == pytest.approx(1e-7)
    assert decoder_lr == pytest.approx(1e-6)


def test_condseg_encoder_lr_cannot_be_applied_to_other_models():
    from scripts.train_unet_multilabel import build_optimizer

    config = {"train": {"optimizer": "adam", "learning_rate": 1e-4,
                        "encoder_learning_rate": 1e-5, "weight_decay": 0}}
    with pytest.raises(ValueError, match="ConDSeg stage 2"):
        build_optimizer(torch.nn.Linear(2, 2), config)


def test_product_convolution_matches_equation_and_clips_exponent():
    module = StableProductConv2d(1, 1)
    with torch.no_grad():
        module.conv.weight.zero_()
        module.conv.weight[0, 0, 1, 1] = 2
        module.conv.bias.fill_(0.3)
    x = torch.linspace(-3, 3, 25).reshape(1, 1, 5, 5).requires_grad_()
    expected = ((torch.nn.functional.softplus(x) + torch.nn.functional.softplus(module.theta_raw) + 1e-7).log() * 2 + 0.3).clamp(-10, 10).exp()
    torch.testing.assert_close(module(x), expected)
    module(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert module.theta_raw.grad.abs() > 0
    with torch.no_grad():
        module.conv.bias.fill_(1e4)
    torch.testing.assert_close(module(x), torch.full_like(x, np.exp(10)), rtol=1e-6, atol=0.01)


def test_pu_architecture_has_every_residual_block_in_paper_figure():
    model = PUUNet(out_channels=1)
    assert [sum(isinstance(m, ResidualBlock) for m in stage.children()) for stage in model.encoder] == [2, 2, 2, 1]
    assert sum(isinstance(m, ResidualBlock) for m in model.modules()) == 13
    product_names = [n for n, m in model.named_modules() if isinstance(m, StableProductConv2d)]
    assert product_names == ["encoder.3.1.second.0", "bottleneck.second.0", "decoder.0.second.0", "decoder.1.second.0"]
    # The paper rounds to 14.393M but omits bias/shortcut details; ensure the
    # figure topology accounts for its parameter scale, unlike the old 12.8M.
    assert sum(p.numel() for p in model.parameters()) == pytest.approx(14.393e6, rel=.001)


def test_spectral_mamba_layer_figure_residual_preserves_zero_mixer_input():
    pytest.importorskip("mamba_ssm")
    from bs.spectralmamba_unet import SpectralDecompositionModeling
    class ZeroMixer(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    module = SpectralDecompositionModeling(8, use_checkpoint=False)
    band = torch.randn(2, 8, 7, 8, requires_grad=True)
    output = module._model_band(band, ZeroMixer())
    torch.testing.assert_close(output, band)
    output.square().sum().backward()
    torch.testing.assert_close(band.grad, 2 * band.detach())


def test_spectral_figure_concat_block_keeps_input_feature_path():
    from bs.spectralmamba_unet import ConvBlock
    block = ConvBlock(8, 16).eval()
    assert block.second[0].in_channels == 24
    assert block.project.in_channels == 40
    with torch.no_grad():
        for stage in [block.first, block.second]:
            stage[0].weight.zero_(); stage[0].bias.zero_()
        block.project.weight.zero_(); block.project.bias.zero_()
        block.project.weight[:8, :8, 0, 0] = torch.eye(8)
    x = torch.randn(2, 8, 8, 8)
    y = block(x)
    torch.testing.assert_close(y[:, :8], x)
    assert y[:, 8:].count_nonzero() == 0


def test_condseg_cr_transfer_rejects_other_fold_and_requires_complete_encoder():
    model = ConDSegAuthor(pretrained=False)
    state = {k: v for k, v in model.state_dict().items() if k.startswith(tuple(f"core.layer{i}." for i in range(4)))}
    checkpoint = {"config": {"train": {"folds_to_run": ["f2"]}}, "model": state}
    with patch("torch.load", return_value=checkpoint), pytest.raises(ValueError, match="different validation fold"):
        model.initialize_backbone("unused.pt", "f1")
    checkpoint["config"]["train"]["folds_to_run"] = ["f1"]
    key = next(k for k in state if k.endswith("weight"))
    state[key] = torch.full_like(state[key], 0.123)
    with patch("torch.load", return_value=checkpoint):
        model.initialize_backbone("unused.pt", "f1")
    torch.testing.assert_close(model.state_dict()[key], state[key])
    state.pop(key)
    with patch("torch.load", return_value=checkpoint), pytest.raises(ValueError, match="complete compatible encoder"):
        model.initialize_backbone("unused.pt", "f1")


def test_udmamba_uncertainty_sort_and_restore_are_exact_and_differentiable():
    pytest.importorskip("mamba_ssm")
    from bs.author_models.udmamba import restore_original_order, sort_by_entropy
    x = torch.tensor([[[[1., 0., 2., 1.]], [[1., 4., 3., 1.]], [[1., 2., 4., 1.]]]], requires_grad=True)
    sorted_x, indices = sort_by_entropy(x)
    assert indices[0, 0].item() == 1  # largest mean absolute channel deviation
    restored = restore_original_order(sorted_x.flatten(2), indices, x.shape).reshape_as(x)
    torch.testing.assert_close(restored, x)
    restored.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())

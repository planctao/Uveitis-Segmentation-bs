"""Four pretrained register positions at three frozen DINO-L depths."""
import numpy as np
import torch
from torch.nn import functional as F

TOKEN_KEYS = ['early_registers', 'middle_registers', 'final_registers']


def forward_depth_features(model, images, captured):
    assert model.num_prefix_tokens == 5 and model.num_features == 1024
    grid = images.shape[-1]//16
    with torch.autocast('cuda', dtype=torch.float16):
        final = model.forward_features(images); early = captured['early']; middle = captured['middle']
        assert early.shape == middle.shape == final.shape == (len(images), grid*grid+5, 1024)
        values = {'early_registers': early[:, 1:5], 'middle_registers': middle[:, 1:5], 'final_registers': final[:, 1:5]}
        patch = final[:, 5:]
        regions = F.adaptive_avg_pool2d(patch.transpose(1, 2).reshape(len(images), 1024, grid, grid), (3, 3)).flatten(2).transpose(1, 2)
        legacy = {'global': final[:, 0], 'regions': regions, 'mid': middle[:, 5:].mean(1), 'dispersion': patch.float().std(1)}
    depth = {key: value.float().cpu().numpy().astype('float16' if key == 'final_registers' else 'float32') for key, value in values.items()}
    legacy = {key: value.float().cpu().numpy().astype('float16') for key, value in legacy.items()}
    return depth, legacy


def install_depth_hooks(model, captured):
    return [model.blocks[7].register_forward_hook(lambda module, args, output: captured.update(early=output)),
        model.blocks[15].register_forward_hook(lambda module, args, output: captured.update(middle=output))]

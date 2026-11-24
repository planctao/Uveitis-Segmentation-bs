"""Pixel-occupancy-weighted DINO pooling with explicit empty-region fallback."""
import torch
import torch.nn.functional as F


def foreground_fraction(normalized_images, patch_grid):
    mean = normalized_images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = normalized_images.new_tensor([.229, .224, .225])[None, :, None, None]
    pixels = (normalized_images * std + mean).clamp(0, 1)
    foreground = (pixels.amax(1, keepdim=True) > 20/255) & (pixels[:, 1:2] > 10/255)
    return F.adaptive_avg_pool2d(foreground.float(), (patch_grid, patch_grid))


def region_pool(patch_tokens, occupancy, grid):
    assert patch_tokens.ndim == 4 and occupancy.shape == (len(patch_tokens), 1, *patch_tokens.shape[-2:])
    tokens = patch_tokens.float()
    ordinary = F.adaptive_avg_pool2d(tokens, (grid, grid))
    pooled_weight = F.adaptive_avg_pool2d(occupancy.float(), (grid, grid))
    weighted = F.adaptive_avg_pool2d(tokens * occupancy, (grid, grid)) / pooled_weight.clamp(min=1e-6)
    weighted = torch.where(pooled_weight > 1e-6, weighted, ordinary)
    flatten = lambda value: value.flatten(2).transpose(1, 2)
    return flatten(ordinary), flatten(weighted), pooled_weight.flatten(1)


def valid_regions(occupancy):
    valid = occupancy > .2
    empty = ~valid.any(1)
    if empty.any():
        valid = valid.clone()
        valid[empty, occupancy[empty].argmax(1)] = True
    return valid

"""Intermediate regions and final registers with matched late spatial controls."""
import numpy as np
import torch
from torch.nn import functional as F

TOKEN_KEYS = ['mid_regions', 'late_regions_control', 'registers', 'quadrants_control']


def forward_token_features(model, images, intermediate):
    size = images.shape[-1]; grid = size//16
    assert model.num_prefix_tokens == 5 and model.num_features == 1024
    with torch.autocast('cuda', dtype=torch.float16):
        tokens = model.forward_features(images); mid = intermediate['tokens']; patch = tokens[:, 5:]; middle = mid[:, 5:]
        assert tokens.shape[1] == mid.shape[1] == grid*grid+5
        def pool(value, cells):
            return F.adaptive_avg_pool2d(value.transpose(1, 2).reshape(len(images), 1024, grid, grid), (cells, cells)).flatten(2).transpose(1, 2)
        values = {'mid_regions': pool(middle, 3), 'late_regions_control': pool(patch, 3),
            'registers': tokens[:, 1:5], 'quadrants_control': pool(patch, 2)}
        legacy = {'global': tokens[:, 0], 'regions': values['late_regions_control'],
            'mid': middle.mean(1), 'dispersion': patch.float().std(1)}
    convert = lambda x: {key: value.float().cpu().numpy().astype('float16') for key, value in x.items()}
    return convert(values), convert(legacy)


def independent_pool(patches, grid, cells):
    array = np.asarray(patches, dtype='float32').reshape(len(patches), grid, grid, 1024); values = []
    for row in range(cells):
        for col in range(cells):
            a, b = row*grid//cells, ((row+1)*grid+cells-1)//cells
            c, d = col*grid//cells, ((col+1)*grid+cells-1)//cells
            values.append(array[:, a:b, c:d].mean((1, 2)))
    return np.stack(values, 1).astype('float16')

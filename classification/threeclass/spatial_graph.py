"""Zero-initialized regional neighbor contrast before attention pooling."""
import torch
from torch import nn


class SpatialGraph(nn.Module):
    def __init__(self, dimension=128, rank=32, steps=1):
        super().__init__()
        if steps not in [1, 2]:raise ValueError('Only predeclared one/two graph steps supported')
        self.steps = steps
        self.incoming = nn.Linear(2 * dimension, rank)
        self.outgoing = nn.Linear(rank, dimension)
        nn.init.zeros_(self.outgoing.weight);nn.init.zeros_(self.outgoing.bias)

    @staticmethod
    def neighbor_difference(tokens, mask=None):
        n, count, dimension = tokens.shape
        grid = int(count ** .5)
        if grid * grid != count:raise ValueError('Region graph requires a square grid')
        valid = torch.ones((n, count), device=tokens.device, dtype=torch.bool) if mask is None else mask > .5
        x = tokens.reshape(n, grid, grid, dimension)
        valid = valid.reshape(n, grid, grid)
        neighbor_sum = torch.zeros_like(x)
        neighbor_count = tokens.new_zeros((n, grid, grid, 1))
        for row_shift, column_shift in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = torch.roll(x, shifts=(row_shift, column_shift), dims=(1, 2))
            allowed = torch.roll(valid, shifts=(row_shift, column_shift), dims=(1, 2)).clone()
            if row_shift == 1:allowed[:, 0] = False
            if row_shift == -1:allowed[:, -1] = False
            if column_shift == 1:allowed[:, :, 0] = False
            if column_shift == -1:allowed[:, :, -1] = False
            neighbor_sum = neighbor_sum + shifted * allowed[:, :, :, None]
            neighbor_count = neighbor_count + allowed[:, :, :, None]
        difference = neighbor_sum / neighbor_count.clamp(min=1) - x
        difference = torch.where((neighbor_count > 0) & valid[:, :, :, None], difference, torch.zeros_like(difference))
        return difference.reshape(n, count, dimension)

    def forward(self, tokens, mask=None):
        for _ in range(self.steps):
            difference = self.neighbor_difference(tokens, mask)
            delta = self.outgoing(torch.nn.functional.gelu(self.incoming(torch.cat([tokens, difference], -1))))
            if mask is not None:delta = delta * (mask > .5)[:, :, None]
            tokens = tokens + .1 * delta
        return tokens

"""Compact second-order regional evidence with a neutral residual readout."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class RegionalSecondOrder(nn.Module):
    def __init__(self, dimension, diseases, cfg):
        super().__init__()
        rank = int(cfg.get('covariance_rank', 16))
        assert rank in [16, 32]
        self.rank = rank
        self.mode = cfg.get('covariance_mode', 'full')
        assert self.mode in ['diagonal', 'full', 'uncentered']
        self.projection = nn.Linear(dimension, rank, bias=False)
        indices = torch.triu_indices(rank, rank)
        self.register_buffer('triangle_rows', indices[0])
        self.register_buffer('triangle_cols', indices[1])
        self.register_buffer('triangle_scale', torch.where(indices[0] == indices[1], 1., math.sqrt(2.)))
        size = rank if self.mode == 'diagonal' else rank * (rank + 1) // 2
        self.dropout = nn.Dropout(cfg.get('covariance_dropout', .2))
        self.readout = nn.Linear(size, diseases)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def embedding(self, regions, mask=None):
        projected = F.gelu(self.projection(regions))
        valid = torch.ones_like(projected[:, :, 0]) if mask is None else (mask > .5).to(projected.dtype)
        valid = torch.where(valid.any(1, keepdim=True), valid, torch.ones_like(valid))
        total = valid.sum(1).clamp(min=1)
        if self.mode != 'uncentered':
            mean = (projected * valid[:, :, None]).sum(1) / total[:, None]
            projected = projected - mean[:, None]
        moment = torch.einsum('btr,bts->brs', projected * valid[:, :, None], projected) / total[:, None, None]
        if self.mode == 'diagonal':
            value = moment.diagonal(dim1=-2, dim2=-1)
        else:
            value = moment[:, self.triangle_rows, self.triangle_cols] * self.triangle_scale
        # This signed-root form stays differentiable and finite at exactly zero.
        rooted = value / (value.abs() + 1e-5).sqrt()
        return F.normalize(rooted, dim=-1, eps=1e-6)

    def forward(self, regions, mask=None):
        embedding = self.embedding(regions, mask)
        return self.readout(self.dropout(embedding)), embedding

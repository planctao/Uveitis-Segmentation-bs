"""Shared regional pooling with exactly symmetric paired-view evidence."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class MirrorConsensusPool(nn.Module):
    dimension = 640

    def __init__(self, dims, cfg):
        super().__init__()
        self.region_proj = nn.Linear(dims['regions'], 128)
        self.region_gate = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 2))
        self.global_proj = nn.Linear(dims['global'], 256)
        self.mid_proj = nn.Linear(dims['mid'], 64)
        self.std_proj = nn.Linear(dims['dispersion'], 64)
        initial = cfg.get('mirror_agreement', 1.)
        self.use_agreement = initial > 0
        if initial > 0:
            self.log_penalty = nn.Parameter(torch.full((2,), math.log(math.expm1(initial))))
        else:
            self.register_buffer('log_penalty', torch.zeros(2))

    def forward(self, x):
        assert x['regions'].ndim == 4 and x['regions'].shape[1] == 2
        regions = F.gelu(self.region_proj(x['regions']))
        center = regions.mean(1)
        normalized = F.normalize(regions, dim=-1)
        disagreement = (normalized[:, 0] - normalized[:, 1]).square().sum(-1)
        penalty = F.softplus(self.log_penalty) if self.use_agreement else torch.zeros_like(self.log_penalty)
        scores = self.region_gate(center) - disagreement[:, :, None] * penalty
        attention = scores.softmax(1)
        pooled = torch.einsum('btk,bvtd->bvkd', attention, regions).flatten(2)
        views = torch.cat([F.gelu(self.global_proj(x['global'])), pooled,
                           F.gelu(self.mid_proj(x['mid'])), F.gelu(self.std_proj(x['dispersion']))], -1)
        return views.mean(1), views, attention, disagreement

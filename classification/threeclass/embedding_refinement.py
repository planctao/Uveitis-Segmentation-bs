"""Shared low-rank residual refinement after the original MLP encoder."""
import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingRefinement(nn.Module):
    def __init__(self,dimension,rank,steps,scale=.1):
        super().__init__()
        if rank<=0 or steps not in [1,2] or not 0<scale<=1:
            raise ValueError('Positive rank, one/two steps and bounded positive scale required')
        self.steps=steps;self.scale=float(scale)
        self.norm=nn.LayerNorm(dimension)
        self.down=nn.Linear(dimension,rank)
        self.up=nn.Linear(rank,dimension)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self,embedding):
        for _ in range(self.steps):
            embedding=embedding+self.scale*self.up(F.gelu(self.down(self.norm(embedding))))
        return embedding

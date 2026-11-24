"""Bounded global-context residual queries for the existing regional attention."""
import math
import torch
from torch import nn


class ContextQueries(nn.Module):
    def __init__(self, context_dim, token_dim, heads, rank=16, strength=1.):
        super().__init__()
        if rank <= 0 or not 0 < strength <= 1:
            raise ValueError('Positive query rank and strength in (0,1] required')
        self.heads, self.rank, self.strength = heads, rank, float(strength)
        self.key = nn.Linear(token_dim, rank, bias=False)
        self.query = nn.Linear(context_dim, heads * rank)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def forward(self, tokens, context):
        query = self.query(context).reshape(len(context), self.heads, self.rank)
        key = self.key(tokens)
        scores = torch.einsum('btd,bhd->bth', key, query) / math.sqrt(self.rank)
        return self.strength * scores.tanh()

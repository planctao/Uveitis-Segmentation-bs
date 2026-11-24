"""Explicit coarse family probabilities and within-family disease conditionals."""
import math
import torch
from torch import nn


class FactorizedCoarse(nn.Module):
    def __init__(self, dimension, order):
        super().__init__()
        if order not in ['flat', 'cascade_i', 'cascade_n']:raise ValueError('Unknown coarse factorization')
        self.order = order
        self.readout = nn.Linear(dimension, 3 if order == 'flat' else 2)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        if order != 'flat':
            with torch.no_grad():self.readout.bias[0] = -math.log(2) if order == 'cascade_i' else math.log(2)

    def forward(self, embedding):
        logits = self.readout(embedding)
        if self.order == 'flat':return logits.softmax(-1), logits.new_empty((len(logits), 0))
        gates = logits.sigmoid();a, b = gates.unbind(1)
        probability = torch.stack([a, (1-a)*b, (1-a)*(1-b)], 1) if self.order == 'cascade_i' else torch.stack([a*b, a*(1-b), 1-a], 1)
        return probability, gates


def conditional_disease_logp(logits, mapping):
    mapping = mapping.argmax(1) if mapping.ndim == 2 else mapping
    if set(mapping.tolist()) != {0, 1, 2}:raise ValueError('All three disease families required')
    normalization = torch.stack([torch.logsumexp(logits[:, mapping == label], 1) for label in range(3)], 1)
    return logits - normalization[:, mapping]

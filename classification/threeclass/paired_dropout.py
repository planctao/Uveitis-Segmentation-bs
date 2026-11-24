"""Two stochastic training passes with optional probability consistency.

This is an R-Drop-style training ablation; deployed architecture is unchanged.
"""
import torch.nn.functional as F
from experiment import Model, loss_fn


class PairedDropoutModel(Model):
    def forward(self, x):
        first = super().forward(x)
        if self.training:
            first['second_view'] = super().forward(x)
        return first


def symmetric_kl(a, b):
    log_a = a.clamp(min=1e-7).log(); log_b = b.clamp(min=1e-7).log()
    return .5 * ((a * (log_a - log_b)).sum(-1) + (b * (log_b - log_a)).sum(-1))


def paired_loss(out, y, disease, weights, cfg):
    second = out['second_view']
    task = .5 * (loss_fn(out, y, disease, weights, cfg)
                  + loss_fn(second, y, disease, weights, cfg))
    strength = cfg['paired_consistency_weight']
    if strength == 0:
        return task
    level = cfg['paired_consistency_level']
    if level == 'coarse':
        a, b = out['p'], second['p']
    elif level == 'disease':
        a, b = F.softmax(out['subtype'], -1), F.softmax(second['subtype'], -1)
    else:
        raise ValueError('Consistency level must be coarse or disease')
    return task + strength * (symmetric_kl(a, b) * weights['leaf']).mean()

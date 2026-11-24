"""Normalized subtype prototypes with optional multiple centers and training margins."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class AngularTaxonomy(nn.Module):
    def __init__(self, dimension, diseases, cfg):
        super().__init__()
        self.centers = int(cfg.get('angular_centers', 1))
        assert self.centers in [1, 2]
        self.prototypes = nn.Parameter(torch.randn(diseases, self.centers, dimension) * .05)
        self.log_scale = nn.Parameter(torch.tensor(math.log(cfg.get('angular_scale', 10.))))

    def similarities(self, embedding):
        values = torch.einsum('...d,ckd->...ck', F.normalize(embedding, dim=-1),
                              F.normalize(self.prototypes, dim=-1))
        # The most relevant center represents each disease at prediction time.
        return values.max(-1).values

    def scale(self):
        return self.log_scale.exp().clamp(4., 32.)

    def forward(self, embedding):
        return self.similarities(embedding) * self.scale()


def margin_logits(cosines, scale, target, kind, margin):
    if kind == 'none' or margin == 0:
        return cosines * scale
    assert kind in ['cosface', 'arcface'] and 0 <= margin < .5
    valid = target >= 0
    safe = target.clamp(min=0)
    positive = cosines.gather(1, safe[:, None]).squeeze(1)
    if kind == 'cosface':
        corrected = positive - margin
    else:
        cosine = positive.clamp(-1 + 1e-6, 1 - 1e-6)
        sine = (1 - cosine.square()).clamp(min=1e-6).sqrt()
        rotated = cosine * math.cos(margin) - sine * math.sin(margin)
        corrected = torch.where(cosine > -math.cos(margin), rotated,
                                cosine - math.sin(margin) * margin)
    selector = F.one_hot(safe, num_classes=cosines.shape[1]).to(cosines.dtype) * valid[:, None]
    return (cosines + selector * (corrected - positive)[:, None]) * scale

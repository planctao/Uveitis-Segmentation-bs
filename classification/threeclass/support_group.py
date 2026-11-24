"""Fit-support-defined sibling groups, preserving all original disease logits."""
import torch
from torch import nn


class SupportGroups(nn.Module):
    def __init__(self, mapping, counts, threshold):
        super().__init__()
        if threshold <= 0 or len(mapping) != len(counts):raise ValueError('Positive support threshold and aligned counts required')
        identities = []
        for disease, (family, count) in enumerate(zip(mapping, counts)):
            if count <= 0:raise ValueError('Fitting disease support must be positive')
            identities.append((int(family), 'tail') if count < threshold else (int(family), str(disease)))
        groups = []
        for identity in identities:
            if identity not in groups:groups.append(identity)
        labels = torch.tensor([groups.index(identity) for identity in identities], dtype=torch.long)
        self.register_buffer('disease_to_group', labels)
        self.group_count = len(groups)

    def forward(self, disease_logits):
        return torch.stack([torch.logsumexp(disease_logits[:, self.disease_to_group == group], 1)
            for group in range(self.group_count)], 1)

    def targets(self, disease):
        known = disease >= 0
        return torch.where(known, self.disease_to_group[disease.clamp(min=0)], torch.full_like(disease, -1))

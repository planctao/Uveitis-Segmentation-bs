"""Complete patient/disease bags for training-only auxiliary supervision."""
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model, sample_weights, loss_fn


def make_bags(frame):
    keys = list(zip(frame.patient_group.astype(str), frame.disease.astype(str), frame.label.astype(int)))
    unique = sorted(set(keys)); mapping = {key: i for i, key in enumerate(unique)}
    ids = np.array([mapping[key] for key in keys], dtype='int64')
    support = pd.DataFrame(unique, columns=['patient_group', 'disease', 'label'])
    support['bag_id'] = np.arange(len(support))
    support['images'] = np.bincount(ids, minlength=len(support))
    support['weight'] = sample_weights(support, support.label)
    members = [np.flatnonzero(ids == i) for i in range(len(support))]
    assert sum(map(len, members)) == len(frame) and min(map(len, members)) > 0
    return ids, members, support


def complete_batches(members, order, target_images=256):
    assert target_images > 0 and sorted(map(int, order)) == list(range(len(members)))
    current = []; size = 0
    for token in order:
        rows = members[int(token)]
        if current and size + len(rows) > target_images:
            yield np.concatenate(current); current = []; size = 0
        current.append(rows); size += len(rows)
        if size >= target_images:
            yield np.concatenate(current); current = []; size = 0
    if current:
        yield np.concatenate(current)


class PatientBagModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        assert cfg['hidden'] == 256
        self.encoder[2].register_forward_pre_hook(self._capture_hidden)
        with torch.random.fork_rng(devices=[]):
            self.bag_quality = nn.Linear(256, 1)
            nn.init.zeros_(self.bag_quality.weight); nn.init.zeros_(self.bag_quality.bias)

    def _capture_hidden(self, module, args):
        self._bag_hidden = args[0]

    def forward(self, x):
        out = super().forward(x)
        out['bag_quality'] = self.bag_quality(self._bag_hidden).squeeze(-1)
        return out


def bag_distributions(out, bag_ids, pooling):
    tokens, inverse = torch.unique(bag_ids, sorted=True, return_inverse=True)
    count = len(tokens); quality = out['bag_quality']
    if pooling == 'mean':
        quality = torch.zeros_like(quality)
    elif pooling != 'attention':
        raise ValueError('Bag pooling must be mean or attention')
    maximum = quality.new_full((count,), -torch.inf).scatter_reduce(0, inverse, quality.detach(), reduce='amax')
    unnormalized = (quality - maximum[inverse]).exp()
    denominator = quality.new_zeros(count).scatter_add(0, inverse, unnormalized)
    attention = unnormalized / denominator[inverse]
    coarse = out['p'].new_zeros((count, 3)).index_add(0, inverse, out['p'] * attention[:, None])
    fine = out['subtype'].softmax(-1)
    disease = fine.new_zeros((count, fine.shape[1])).index_add(0, inverse, fine * attention[:, None])
    return {'tokens': tokens, 'inverse': inverse, 'attention': attention, 'coarse': coarse, 'disease': disease}


def bag_auxiliary(out, y, disease, weights, cfg):
    pooled = bag_distributions(out, weights['bag_ids'], cfg['patient_bag_pool'])
    inverse = pooled['inverse']; count = len(pooled['tokens'])
    target = y.new_full((count,), -1).scatter_reduce(0, inverse, y, reduce='amax')
    fine_target = disease.new_full((count,), -1).scatter_reduce(0, inverse, disease, reduce='amax')
    bag_weights = weights['bag_weight'].new_zeros(count).scatter_reduce(0, inverse, weights['bag_weight'], reduce='amax')
    logp = pooled['coarse'].clamp(min=1e-7).log()
    nll = F.nll_loss(logp, target, reduction='none')
    smoothing = cfg.get('label_smoothing', 0.)
    nll = (1 - smoothing) * nll - smoothing * logp.mean(1)
    fine_loss = F.nll_loss(pooled['disease'].clamp(min=1e-7).log(), fine_target, reduction='none')
    return ((nll + cfg['aux'] * fine_loss) * bag_weights).mean()


def patient_bag_loss(out, y, disease, weights, cfg):
    original = loss_fn(out, y, disease, weights, cfg)
    strength = cfg['patient_bag_weight']
    if strength == 0:
        return original
    return original + strength * bag_auxiliary(out, y, disease, weights, cfg)

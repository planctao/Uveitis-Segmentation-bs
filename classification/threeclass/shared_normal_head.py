"""One attention MLP shared by raw and normal-reference evidence views."""
import torch.nn.functional as F
from experiment import Model, loss_fn
from normal_subspace import NormalSuppressedModel


class SharedNormalModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['normal_rank'] == 32 and cfg['normal_strength'] == 1.
        assert cfg['shared_normal_level'] in ['coarse', 'disease']
        assert cfg['shared_normal_consistency'] >= 0
        assert cfg['shared_normal_weight'] == .75

    def forward(self, x):
        raw = Model.forward(self, x)
        normal = Model.forward(self, self.transform(x))
        weight = self.cfg['shared_normal_weight']
        return {'p': (1 - weight) * raw['p'] + weight * normal['p'],
                'raw_view': raw, 'normal_view': normal}


def shared_normal_loss(out, y, disease, weights, cfg):
    a, b = out['raw_view'], out['normal_view']
    weight = cfg['shared_normal_weight']
    task = (1 - weight) * loss_fn(a, y, disease, weights, cfg) + weight * loss_fn(b, y, disease, weights, cfg)
    strength = cfg['shared_normal_consistency']
    if not strength:
        return task
    if cfg['shared_normal_level'] == 'coarse':
        p, q = a['p'], b['p']
    else:
        p, q = F.softmax(a['subtype'], -1), F.softmax(b['subtype'], -1)
    logp, logq = p.clamp(min=1e-7).log(), q.clamp(min=1e-7).log()
    discrepancy = .5 * ((p * (logp - logq)).sum(-1) + (q * (logq - logp)).sum(-1))
    return task + strength * (discrepancy * weights['leaf']).mean()

"""Normal-B binary residual specialists with complete-graph versus node coupling."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import loss_fn
from normal_subspace import NormalSuppressedModel

PAIRS = [(0, 1), (0, 2), (1, 2)]
# Columns below are (0,1), (0,2), (1,2); rows are classes.
INCIDENCE = [[1., 1., 0.], [-1., 0., 1.], [0., -1., -1.]]
DETAILS = ['pairwise_base_probability', 'pairwise_residual', 'pairwise_class_shift', 'pairwise_binary_logits']


class BinaryResidualExperts(nn.Module):
    def __init__(self, rank):
        super().__init__(); assert rank in [16, 32]
        self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(256), nn.Linear(256, rank), nn.GELU(), nn.Linear(rank, 2)) for _ in PAIRS])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight); nn.init.zeros_(expert[-1].bias)

    def forward(self, hidden):
        values = torch.stack([expert(hidden) for expert in self.experts], 1)
        return .1*(values[:, :, 0]-values[:, :, 1])


class NormalPairwiseGraphModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['hidden'] == 256 and cfg['taxonomy_only'] and cfg['pairwise_kind'] in ['node', 'edge']
        with torch.random.fork_rng(devices=[]): self.pairwise = BinaryResidualExperts(cfg['pairwise_rank'])
        self.witness_path = None if witness_path is None else Path(witness_path); self.witness_written = False

    def forward(self, x):
        context = None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device = x['global'].device
            context = {'model': self, 'raw_global_all': x['global'], 'rng_cpu': torch.random.get_rng_state().clone(),
                'rng_cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                'initial_parameter_sha256': {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name, p in self.named_parameters()}}
        out = super().forward(x); base = out['p']; residual = self.pairwise(out['embedding'])
        if self.cfg['pairwise_kind'] == 'edge':
            shift = residual@residual.new_tensor(INCIDENCE).T/3
            binary_delta = residual
        else:
            shift = residual-residual.mean(-1, keepdim=True)
            binary_delta = torch.stack([shift[:, a]-shift[:, b] for a, b in PAIRS], -1)
        original_logits = out['subtype']
        original_coarse = torch.stack([torch.logsumexp(original_logits[:, self.subtype_to_class[:, c].bool()], -1) for c in range(3)], -1).detach()
        binary = torch.stack([original_coarse[:, a]-original_coarse[:, b] for a, b in PAIRS], -1)+binary_delta
        logits = original_logits+shift@self.subtype_to_class.T
        p = logits.softmax(-1)@self.subtype_to_class
        out.update(p=p, flat=p.clamp(min=1e-7).log(), subtype=logits, subtype_probability=p,
            pairwise_base_probability=base, pairwise_residual=residual, pairwise_class_shift=shift, pairwise_binary_logits=binary)
        if context is not None: out['_pairwise_witness_context'] = context
        return out


def pairwise_penalty(binary, y, weight, smoothing=.03):
    values = []
    for i, (a, b) in enumerate(PAIRS):
        mask = ((y == a)|(y == b)).to(weight.dtype); mass = (weight*mask).sum()
        target = (y == a).to(binary.dtype)*(1-smoothing)+smoothing/2
        per_image = F.binary_cross_entropy_with_logits(binary[:, i], target, reduction='none')
        values.append((per_image*weight*mask).sum()/mass.clamp(min=1e-12))
    return torch.stack(values).mean()


def pairwise_graph_loss(out, y, disease, weights, cfg):
    original = loss_fn(out, y, disease, weights, cfg)
    penalty = pairwise_penalty(out['pairwise_binary_logits'], y, weights['leaf'], cfg['label_smoothing'])
    loss = original+cfg['pairwise_aux']*penalty
    context = out.get('_pairwise_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'rank': np.array(cfg['pairwise_rank']),
            'kind': np.array(cfg['pairwise_kind']), 'original_loss': np.array(float(original.detach())),
            'penalty': np.array(float(penalty.detach())), 'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'y': y, 'disease': disease, 'probability': out['p'],
            'subtype': out['subtype'], 'parameter_gradient': torch.cat([g.flatten() for g in gradients]),
            **{key: out[key] for key in DETAILS}, **{'weight_'+key: v for key, v in weights.items()}}
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss

"""Aligned normal-B/L regional products versus matched additive residuals."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS, loss_fn
from normal_large_residual import NormalLargeResidualModel
from normal_subspace import independent_transform

DETAILS = ['large_residual_input', 'large_residual_hidden', 'large_residual_delta',
           'interaction_base_regions', 'interaction_large_regions', 'interaction_joint_regions']


class LargeInteraction(nn.Module):
    def __init__(self, rank, kind):
        super().__init__()
        assert rank in [16, 32] and kind in ['add', 'product']
        self.kind = kind
        self.base_norm = nn.LayerNorm(1536)
        self.large_norm = nn.LayerNorm(2048)
        self.base_down = nn.Linear(1536, rank)
        self.large_down = nn.Linear(2048, rank)
        self.joint_norm = nn.LayerNorm(rank)
        self.readout = nn.Linear(4*rank, 256)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, base, large):
        a = {k: torch.tanh(self.base_down(self.base_norm(v))) for k, v in base.items()}
        b = {k: torch.tanh(self.large_down(self.large_norm(v))) for k, v in large.items()}
        joined = {k: a[k]*b[k] if self.kind == 'product' else (a[k]+b[k])/(2**.5) for k in KEYS}
        pooled = {k: v.mean(1) if k == 'regions' else v for k, v in joined.items()}
        summary = torch.cat([self.joint_norm(pooled[k]) for k in ['global', 'regions', 'mid', 'dispersion']], dim=-1)
        return .1*self.readout(summary), summary, a['regions'], b['regions'], joined['regions']


class NormalLargeInteractionModel(NormalLargeResidualModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]):
            self.large_residual = LargeInteraction(cfg['large_interaction_rank'], cfg['large_interaction_kind'])

    def transform(self, x):
        self._normal_base = super().transform(x)
        return self._normal_base

    def _add(self, module, args, hidden):
        self._base_hidden = hidden
        self._delta, self._residual_input, self._interaction_base, self._interaction_large, self._interaction_joint = self.large_residual(self._normal_base, self._large)
        return hidden+self._delta

    def forward(self, x):
        out = super().forward(x)
        out.update(interaction_base_regions=self._interaction_base, interaction_large_regions=self._interaction_large,
                   interaction_joint_regions=self._interaction_joint)
        return out


def interaction_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg)
    context = out.get('_large_residual_witness_context')
    if context is not None:
        model = context['model']
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)),
            'rank': np.array(cfg['large_interaction_rank']), 'kind': np.array(cfg['large_interaction_kind']),
            'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'], **{k: out[k] for k in DETAILS}}
        tensors.update({'weight_'+k: v for k, v in weights.items()})
        for k, v in tensors.items():
            proof[k] = v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:
            proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for k, v in context['initial_parameter_sha256'].items():
            proof['initial_parameter_sha256_'+k] = np.array(v)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof)
        model.witness_written = True
    return loss


def independent_forward(plain, raw, state, cfg):
    device = next(plain.parameters()).device
    base, large = {}, {}
    for key in KEYS:
        for output, section, prefix in [(base, slice(None, 1536), 'normal_'), (large, slice(1536, None), 'large_normal_')]:
            output[key] = torch.from_numpy(independent_transform(raw[key][..., section],
                state[prefix+'center_'+key].detach().cpu().numpy(),
                state[prefix+'basis_'+key].detach().cpu().numpy(), 1.)).to(device)
    get = lambda name: state['large_residual.'+name].to(device)
    def norm(value, name):
        centered = value-value.sum(-1, keepdim=True)/value.shape[-1]
        return centered*torch.rsqrt(centered.square().sum(-1, keepdim=True)/value.shape[-1]+1e-5)*get(name+'.weight')+get(name+'.bias')
    a = {k: F.linear(norm(v, 'base_norm'), get('base_down.weight'), get('base_down.bias')).tanh() for k, v in base.items()}
    b = {k: F.linear(norm(v, 'large_norm'), get('large_down.weight'), get('large_down.bias')).tanh() for k, v in large.items()}
    joined = {}
    for k in KEYS:
        product = a[k]*b[k] if cfg['large_interaction_kind'] == 'product' else torch.add(a[k], b[k])*(.5**.5)
        joined[k] = product
    summary = torch.cat([norm(joined[k].sum(1)/9 if k == 'regions' else joined[k], 'joint_norm')
                         for k in ['global', 'regions', 'mid', 'dispersion']], dim=-1)
    delta = .1*F.linear(summary, get('readout.weight'), get('readout.bias'))
    captured = {}
    def add(module, args, hidden):
        captured['hidden'] = hidden
        return hidden+delta
    handle = plain.encoder[1].register_forward_hook(add)
    try:
        out = plain(base)
    finally:
        handle.remove()
    out.update(large_residual_input=summary, large_residual_hidden=captured['hidden'], large_residual_delta=delta,
        interaction_base_regions=a['regions'], interaction_large_regions=b['regions'], interaction_joint_regions=joined['regions'])
    return out

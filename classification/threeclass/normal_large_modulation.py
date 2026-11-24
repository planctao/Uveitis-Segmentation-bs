"""Normal-L-conditioned B hidden channels against an equal-parameter additive control."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS, loss_fn
from normal_large_residual import NormalLargeResidualModel
from normal_subspace import independent_transform

DETAILS = ['large_residual_input', 'large_residual_hidden', 'large_residual_delta',
    'modulation_scale', 'modulation_shift', 'modulation_normalized_hidden']


class LargeModulation(nn.Module):
    def __init__(self, rank, kind):
        super().__init__(); assert rank in [16, 32] and kind in ['add', 'film']
        self.kind = kind; self.norm = nn.LayerNorm(8192)
        self.down = nn.Linear(8192, rank); self.readout = nn.Linear(rank, 512)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, large, hidden):
        fused = torch.cat([large[k].mean(1) if k == 'regions' else large[k] for k in KEYS], dim=-1)
        scale, shift = self.readout(F.gelu(self.down(self.norm(fused)))).tanh().chunk(2, dim=-1)
        normalized = F.layer_norm(hidden, (256,))
        delta = .1*(scale*normalized+shift) if self.kind == 'film' else .1*(scale+shift)/(2**.5)
        return delta, fused, scale, shift, normalized


class NormalLargeModulationModel(NormalLargeResidualModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        with torch.random.fork_rng(devices=[]):
            self.large_residual = LargeModulation(cfg['large_modulation_rank'], cfg['large_modulation_kind'])

    def _add(self, module, args, hidden):
        self._base_hidden = hidden
        self._delta, self._residual_input, self._scale, self._shift, self._normalized_hidden = self.large_residual(self._large, hidden)
        return hidden+self._delta

    def forward(self, x):
        out = super().forward(x)
        out.update(modulation_scale=self._scale, modulation_shift=self._shift, modulation_normalized_hidden=self._normalized_hidden)
        return out


def modulation_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_large_residual_witness_context')
    if context is not None:
        model = context['model']
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'rank': np.array(cfg['large_modulation_rank']),
            'kind': np.array(cfg['large_modulation_kind']), 'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'], 'parameter_gradient': torch.cat([g.flatten() for g in gradients]),
            **{k: out[k] for k in DETAILS}, **{'weight_'+k: v for k, v in weights.items()}}
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for key, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+key] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss


def independent_forward(plain, raw, state, cfg):
    device = next(plain.parameters()).device; base = {}; large = {}; details = {}
    for key in KEYS:
        for target, section, prefix in [(base, slice(None, 1536), 'normal_'), (large, slice(1536, None), 'large_normal_')]:
            target[key] = torch.from_numpy(independent_transform(raw[key][..., section],
                state[prefix+'center_'+key].detach().cpu().numpy(), state[prefix+'basis_'+key].detach().cpu().numpy(), 1.)).to(device)
    get = lambda name: state['large_residual.'+name].to(device)
    fused = torch.cat([large[k].sum(1)/9 if k == 'regions' else large[k] for k in KEYS], dim=-1)
    centered = fused-fused.sum(-1, keepdim=True)/8192
    normalized = centered*torch.rsqrt(centered.square().sum(-1, keepdim=True)/8192+1e-5)*get('norm.weight')+get('norm.bias')
    code = F.gelu(F.linear(normalized, get('down.weight'), get('down.bias')))
    gates = torch.tanh(F.linear(code, get('readout.weight'), get('readout.bias')))
    scale, shift = gates[:, :256], gates[:, 256:]
    def add(module, args, hidden):
        center = hidden-hidden.sum(-1, keepdim=True)/256
        normalized_hidden = center*torch.rsqrt(center.square().sum(-1, keepdim=True)/256+1e-5)
        delta = .1*(torch.mul(scale, normalized_hidden)+shift) if cfg['large_modulation_kind'] == 'film' else .1*(scale+shift)*(2**-.5)
        details.update(large_residual_input=fused, large_residual_hidden=hidden, large_residual_delta=delta,
            modulation_scale=scale, modulation_shift=shift, modulation_normalized_hidden=normalized_hidden)
        return hidden+delta
    handle = plain.encoder[1].register_forward_hook(add)
    try: out = plain(base)
    finally: handle.remove()
    out.update(details); return out

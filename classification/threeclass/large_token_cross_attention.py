"""Bidirectional crossdepth token attention versus equalparameter selfattention."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS, loss_fn
from normal_subspace import NormalSuppressedModel

STREAMS = ['mid_tokens', 'register_tokens']
DETAILS = ['cross_base_hidden', 'cross_query', 'cross_joint', 'cross_delta'] + [name+'_'+suffix for name in STREAMS for suffix in ['projected', 'queries', 'keys', 'values', 'connections', 'attended', 'attention', 'pooled', 'bounded']]


class CrossTokenAttention(nn.Module):
    def __init__(self, rank, kind):
        super().__init__(); self.rank = rank; self.kind = kind
        assert rank in [16, 32] and kind in ['self', 'cross']
        self.query_norm = nn.LayerNorm(256); self.query = nn.Linear(256, rank)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(2048) for name in STREAMS})
        self.queries = nn.ModuleDict({name: nn.Linear(2048, rank) for name in STREAMS})
        self.keys = nn.ModuleDict({name: nn.Linear(2048, rank) for name in STREAMS})
        self.values = nn.ModuleDict({name: nn.Linear(2048, rank) for name in STREAMS})
        self.readout = nn.Linear(2*rank, 256); nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, streams, hidden):
        query = self.query(self.query_norm(hidden)); details = {'cross_query': query, 'cross_base_hidden': hidden}
        queries, keys, values = {}, {}, {}
        for name in STREAMS:
            normalized = self.norms[name](streams[name]); queries[name] = self.queries[name](normalized)
            keys[name] = self.keys[name](normalized); values[name] = F.gelu(self.values[name](normalized))
        pooled = []
        for i, name in enumerate(STREAMS):
            other = STREAMS[1-i] if self.kind == 'cross' else name
            connections = torch.einsum('btd,bsd->bts', queries[name], keys[other]).div(self.rank**.5).softmax(-1)
            attended = torch.einsum('bts,bsd->btd', connections, values[other])
            bounded = torch.tanh((values[name]+attended)/(2**.5))
            attention = torch.einsum('bd,btd->bt', query, queries[name]).div(self.rank**.5).softmax(1)
            summary = torch.einsum('bt,btd->bd', attention, bounded); pooled.append(summary)
            for suffix, value in [('projected', streams[name]), ('queries', queries[name]), ('keys', keys[name]), ('values', values[name]),
                ('connections', connections), ('attended', attended), ('bounded', bounded), ('attention', attention), ('pooled', summary)]:
                details[name+'_'+suffix] = value
        joint = torch.cat(pooled, -1)
        details.update(cross_joint=joint, cross_delta=.1*self.readout(joint)); return details


class LargeTokenCrossAttentionModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['hidden'] == 256 and all(dims[key] == 2048 for key in KEYS+STREAMS)
        for name in STREAMS:
            self.register_buffer(name+'_normal_center', torch.zeros(2048)); self.register_buffer(name+'_normal_basis', torch.zeros(2048, 32))
            if geometry is not None:
                getattr(self, name+'_normal_center').copy_(torch.as_tensor(geometry[name+'_center']))
                getattr(self, name+'_normal_basis').copy_(torch.as_tensor(geometry[name+'_basis']))
        with torch.random.fork_rng(devices=[]):
            self.token_cross = CrossTokenAttention(cfg['token_cross_rank'], cfg['token_cross_kind'])
        self.encoder[1].register_forward_hook(self._add)
        self.witness_path = None if witness_path is None else Path(witness_path); self.witness_written = False

    def _add(self, module, args, hidden):
        self._details = self.token_cross(self._streams, hidden)
        return hidden+self._details['cross_delta']

    def forward(self, x):
        context = None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device = x['global'].device
            context = {'model': self, 'raw': {name: x[name] for name in ['global']+STREAMS},
                'rng_cpu': torch.random.get_rng_state().clone(),
                'rng_cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                'initial_parameter_sha256': {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name, p in self.named_parameters()}}
        self._streams = {}
        for name in STREAMS:
            tokens = x[name]; center = getattr(self, name+'_normal_center'); basis = getattr(self, name+'_normal_basis')
            self._streams[name] = (tokens-((tokens-center)@basis)@basis.T).clamp(-10, 10)
        out = super().forward({name: x[name] for name in KEYS}); out.update(self._details)
        if context is not None: out['_token_cross_witness_context'] = context
        return out


def token_cross_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_token_cross_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'kind': np.array(cfg['token_cross_kind']),
            'rank': np.array(cfg['token_cross_rank']), 'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_'+name: value for name, value in context['raw'].items()}
        tensors.update(y=y, disease=disease, probability=out['p'], subtype=out['subtype'],
            parameter_gradient=torch.cat([g.flatten() for g in gradients]))
        tensors.update({key: out[key] for key in DETAILS}); tensors.update({'weight_'+key: value for key, value in weights.items()})
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss

"""Normal-projected register trajectories with matched depth pooling controls."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS, loss_fn
from normal_subspace import NormalSuppressedModel
from large_token_evidence import TokenQueryResidual, DETAILS as BASE_DETAILS
from register_depth_features import TOKEN_KEYS

DETAILS = BASE_DETAILS + [key+'_projected' for key in TOKEN_KEYS] + ['depth_tokens']


def depth_tokens(projected, kind):
    early, middle, final = [projected[key] for key in TOKEN_KEYS]
    if kind == 'final': return torch.cat([final, final, final], 1)
    if kind == 'middle': return torch.cat([middle, middle, middle], 1)
    if kind == 'raw': return torch.cat([early, middle, final], 1)
    assert kind == 'difference'
    return torch.cat([early, (middle-early)/(2**.5), (final-middle)/(2**.5)], 1).clamp(-10, 10)


class RegisterDepthModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['hidden'] == 256 and cfg['register_depth_kind'] in ['final', 'middle', 'raw', 'difference']
        assert all(dims[key] == 2048 for key in KEYS+TOKEN_KEYS)
        for key in TOKEN_KEYS:
            self.register_buffer(key+'_normal_center', torch.zeros(2048)); self.register_buffer(key+'_normal_basis', torch.zeros(2048, 32))
            if geometry is not None:
                getattr(self, key+'_normal_center').copy_(torch.as_tensor(geometry[key+'_center']))
                getattr(self, key+'_normal_basis').copy_(torch.as_tensor(geometry[key+'_basis']))
        with torch.random.fork_rng(devices=[]): self.token_evidence = TokenQueryResidual()
        self.encoder[1].register_forward_hook(self._add)
        self.witness_path = None if witness_path is None else Path(witness_path); self.witness_written = False

    def _add(self, module, args, hidden):
        self._details.update(self.token_evidence(self._tokens, hidden))
        return hidden+self._details['token_delta']

    def forward(self, x):
        context = None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device = x['global'].device
            context = {'model': self, 'raw': {key: x[key] for key in ['global']+TOKEN_KEYS},
                'rng_cpu': torch.random.get_rng_state().clone(),
                'rng_cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                'initial_parameter_sha256': {name: hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest() for name, parameter in self.named_parameters()}}
        projected = {}
        for key in TOKEN_KEYS:
            value = x[key]; center = getattr(self, key+'_normal_center'); basis = getattr(self, key+'_normal_basis')
            projected[key] = (value-((value-center)@basis)@basis.T).clamp(-10, 10)
        self._tokens = depth_tokens(projected, self.cfg['register_depth_kind'])
        self._details = {key+'_projected': value for key, value in projected.items()}
        out = super().forward({key: x[key] for key in KEYS})
        out.update(self._details, token_projected=self._tokens, depth_tokens=self._tokens)
        if context is not None: out['_depth_witness_context'] = context
        return out


def register_depth_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_depth_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'kind': np.array(cfg['register_depth_kind']),
            'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_'+key: value for key, value in context['raw'].items()}
        tensors.update(y=y, disease=disease, probability=out['p'], subtype=out['subtype'],
            parameter_gradient=torch.cat([gradient.flatten() for gradient in gradients]))
        tensors.update({key: out[key] for key in DETAILS}); tensors.update({'weight_'+key: value for key, value in weights.items()})
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss

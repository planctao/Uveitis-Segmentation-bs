"""Keep selected added projection weights at their random initialization."""
import numpy as np
import torch
from experiment import loss_fn
from large_token_evidence import LargeTokenEvidenceModel, DETAILS

KINDS = ['full_control', 'freeze_qk', 'freeze_value', 'freeze_qkv']


class RegisterProjectionFreezingModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        kind = cfg['register_projection_freeze']; assert kind in KINDS
        prefixes = []
        if kind in ['freeze_qk', 'freeze_qkv']: prefixes += ['token_evidence.query.', 'token_evidence.key.']
        if kind in ['freeze_value', 'freeze_qkv']: prefixes += ['token_evidence.value.']
        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes): parameter.requires_grad_(False)


def token_evidence_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_token_evidence_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(p for p in model.parameters() if p.requires_grad), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'stream': np.array(cfg['token_evidence_stream']),
            'trainable_parameter_names': np.asarray([name for name,p in model.named_parameters() if p.requires_grad],dtype=str), 'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'raw_side_all': context['raw_side_all'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'], 'parameter_gradient': torch.cat([g.flatten() for g in gradients]),
            **{key: out[key] for key in DETAILS}, **{'weight_'+key: value for key, value in weights.items()}}
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss

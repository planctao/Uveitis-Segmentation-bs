"""Patient-balanced routing information constraints on the unchanged M8 head."""
import math
import numpy as np
import torch
from experiment import loss_fn
from large_token_evidence import LargeTokenEvidenceModel, DETAILS

KINDS = ['control', 'smooth', 'balance', 'information']


class RegisterRoutingInformationModel(LargeTokenEvidenceModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        assert cfg['register_routing_information'] in KINDS and cfg['token_evidence_stream'] == 'registers'
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)


def routing_regularizer(attention, patient_ids, kind):
    assert kind in KINDS and attention.ndim == 2 and attention.shape[1] == 4
    if kind == 'control': return attention.sum()*0
    _, inverse, counts = torch.unique(patient_ids, return_inverse=True, return_counts=True)
    weight = counts[inverse].to(attention).reciprocal()/len(counts)
    conditional = (-(attention*attention.clamp_min(1e-8).log()).sum(1)*weight).sum()
    marginal = (attention*weight[:,None]).sum(0)
    marginal_entropy = -(marginal*marginal.clamp_min(1e-8).log()).sum()
    maximum = math.log(attention.shape[1])
    if kind == 'smooth': return (maximum-conditional)/maximum
    if kind == 'balance': return (maximum-marginal_entropy)/maximum
    return (conditional+maximum-marginal_entropy)/maximum


def token_evidence_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg)
    if cfg['register_routing_information'] != 'control':
        loss = loss+cfg['routing_information_weight']*routing_regularizer(out['token_attention'], weights['patient_ids'], cfg['register_routing_information'])
    context = out.get('_token_evidence_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)), 'stream': np.array(cfg['token_evidence_stream']),
            'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy()}
        tensors = {'raw_global_all': context['raw_global_all'], 'raw_side_all': context['raw_side_all'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'], 'parameter_gradient': torch.cat([g.flatten() for g in gradients]),
            **{key: out[key] for key in DETAILS}, **{'weight_'+key: value for key, value in weights.items()}}
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+name] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return loss

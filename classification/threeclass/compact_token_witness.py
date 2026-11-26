"""Actual first-task evidence without redundant frozen-feature array copies."""
import hashlib
import numpy as np
import torch
from experiment import loss_fn
from large_token_evidence import DETAILS

COMPACT_FIELDS = ['raw_global_all', 'raw_side_all', 'token_projected']


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def token_evidence_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg); context = out.get('_token_evidence_witness_context')
    if context is not None:
        model = context['model']; gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        proof = {'seed':np.array(cfg['seed']), 'batch_size':np.array(len(y)), 'stream':np.array(cfg['token_evidence_stream']),
            'loss':np.array(float(loss.detach())), 'rng_cpu':context['rng_cpu'].numpy(), 'compact_witness_version':np.array(1)}
        tensors = {'raw_global_all':context['raw_global_all'], 'raw_side_all':context['raw_side_all'], 'y':y, 'disease':disease,
            'probability':out['p'], 'subtype':out['subtype'], 'parameter_gradient':torch.cat([value.flatten() for value in gradients]),
            **{key:out[key] for key in DETAILS}, **{'weight_'+key:value for key,value in weights.items()}}
        for key,tensor in tensors.items():
            value = tensor.detach().cpu().numpy()
            if key in COMPACT_FIELDS:
                proof[key+'_sha256']=np.array(array_digest(value)); proof[key+'_shape']=np.array(value.shape,dtype='int64'); proof[key+'_dtype']=np.array(value.dtype.str)
            else:proof[key]=value
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for name,value in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+name]=np.array(value)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss


def verify_legacy_equivalence(compact_path, legacy_path):
    with np.load(compact_path) as bank:compact={key:bank[key] for key in bank.files}
    with np.load(legacy_path) as bank:legacy={key:bank[key] for key in bank.files}
    extra={'compact_witness_version'}|{key+'_'+suffix for key in COMPACT_FIELDS for suffix in ['sha256','shape','dtype']}
    assert int(compact['compact_witness_version'])==1 and set(compact)-extra==set(legacy)-set(COMPACT_FIELDS)
    for key,value in legacy.items():
        if key in COMPACT_FIELDS:
            assert str(compact[key+'_sha256'])==array_digest(value)
            assert np.array_equal(compact[key+'_shape'],value.shape) and str(compact[key+'_dtype'])==value.dtype.str
        else:assert np.array_equal(compact[key],value),key
    return {'all_retained_arrays_at_original_bits':True,'all_omitted_arrays_at_original_sha256':True}

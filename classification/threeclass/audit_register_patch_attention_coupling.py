"""Independent complete frozen MLP, trainable attention and patch reconstruction."""
import torch
from torch.nn import functional as F
from audit_register_patch_query_selection import independent_probability as patch_probability
from register_patch_attention_coupling import PATCH_NAMES, ATTENTION_PREFIX


def independent_probability(source, tail_state, parameters, raw, bank, config):
    # The inherited auditor reconstructs the reader with direct tensor formulas.
    # Supply its adapted tensors directly, retaining each differentiable leaf.
    state = dict(source['state_dict'])
    for name, value in parameters.items():
        if name.startswith(ATTENTION_PREFIX):
            state[name[len('head.'):]] = value
    adapted_source = {**source, 'state_dict': state}
    patches = {name: parameters[name] for name in PATCH_NAMES}
    return patch_probability(adapted_source, tail_state, patches, raw, bank, config)


def independent_reader(parameters, tokens, hidden):
    def get(name):
        return parameters[ATTENTION_PREFIX+name]

    def norm(value, name):
        centered = value-value.sum(-1, keepdim=True)/value.shape[-1]
        variance = centered.square().sum(-1, keepdim=True)/value.shape[-1]
        return centered*torch.rsqrt(variance+1e-5)*get(name+'.weight')+get(name+'.bias')

    token = norm(tokens, 'token_norm')
    query = F.linear(norm(hidden, 'query_norm'), get('query.weight'), get('query.bias'))
    keys = F.linear(token, get('key.weight'), get('key.bias'))
    values = F.gelu(F.linear(token, get('value.weight'), get('value.bias')))
    logits = (query[:, None]*keys).sum(-1)/(32**.5)
    positive = (logits-logits.max(-1, keepdim=True).values).exp()
    attention = positive/positive.sum(-1, keepdim=True)
    pooled = (attention[..., None]*values).sum(1)
    delta = F.linear(pooled, get('readout.weight'), get('readout.bias'))/10
    return {'token_query': query, 'token_keys': keys, 'token_values': values,
        'token_attention': attention, 'token_pooled': pooled, 'token_delta': delta}

"""Independent joint patch attention/content and complete frozen tail/head."""
import torch
from torch.nn import functional as F
from audit_register_patch_query import finish
from audit_large_token_evidence import ordinary
from experiment import KEYS


def independent_correct(source, tail_state, adapter_state, bank, raw_registers, config):
    context = bank['query_input'].float()
    if config['query_kind'] == 'shared':
        context = context.sum(2, keepdim=True).expand_as(context)/4
    code = F.linear(context, adapter_state['down.weight'])
    delta = F.linear(code, adapter_state['up.weight'])*(config['alpha']/config['rank'])
    delta = delta.reshape(len(context), 2, 4, 16, 64).transpose(2, 3).double()
    query, keys, values = [bank[key].double() for key in ['query_raw', 'patch_keys', 'patch_values']]
    def softmax(logits):
        positive = (logits-logits.max(-1, keepdim=True).values).exp()
        return positive/positive.sum(-1, keepdim=True)
    baseline = softmax(torch.einsum('bshqd,bshkd->bshqk', query, keys)/8)
    modified = softmax(torch.einsum('bshqd,bshkd->bshqk', query+delta, keys)/8)
    updates = []
    for head in range(16):
        source_head = 0 if config['value_kind'] == 'shared' else head
        code = F.linear(values[:, :, head], adapter_state['value_down'][source_head].double())
        updates.append(F.linear(code, adapter_state['value_up'][source_head].double())*(config['alpha']/config['rank']))
    delta_values = torch.stack(updates, 2)
    # Include the modified-attention times modified-value interaction explicitly.
    joint = torch.einsum('bshqk,bshkd->bshqd', modified, values+delta_values)
    baseline_pool = torch.einsum('bshqk,bshkd->bshqd', baseline, values)
    query_difference = torch.einsum('bshqk,bshkd->bshqd', modified, values)-baseline_pool
    value_difference = torch.einsum('bshqk,bshkd->bshqd', modified, delta_values)
    difference = (joint-baseline_pool).transpose(2, 3).reshape(len(query), 2, 4, 1024).float()
    branch = F.linear(difference, tail_state['proj.weight'])*tail_state['gamma1']
    before = bank['postattention'].float()
    correction = finish(tail_state, before+branch)-finish(tail_state, before)
    corrected = (raw_registers+torch.cat((correction[:, 0], correction[:, 1]), -1)).half().float()
    return corrected, {'delta_query': delta, 'selected_attention': modified, 'baseline_attention': baseline,
        'query_pool_difference': query_difference, 'value_pool_difference': value_difference,
        'pool_difference': difference, 'attention_difference': branch, 'tail_difference': correction}


def independent_probability(source, tail_state, adapter_state, raw, bank, config):
    registers, details = independent_correct(source, tail_state, adapter_state, bank, raw['side_tokens'], config)
    state = {name: value.to(registers.device) for name, value in source['state_dict'].items()}
    normalized = {}
    for key, value in {**raw, 'side_tokens': registers}.items():
        mean = torch.as_tensor(source['stats'][key]['mean'], device=value.device)
        std = torch.as_tensor(source['stats'][key]['std'], device=value.device)
        normalized[key] = ((value-mean)/std).clamp(-10, 10)
    projected = {}
    for key in KEYS:
        z = normalized[key]
        center, basis = state['normal_center_'+key], state['normal_basis_'+key]
        projected[key] = (z-((z-center)@basis)@basis.T).clamp(-10, 10)
    center, basis = state['side_normal_center'], state['side_normal_basis']
    tokens = normalized['side_tokens']
    tokens = (tokens-((tokens-center)@basis)@basis.T).clamp(-10, 10)

    def norm(value, prefix):
        centered = value-value.mean(-1, keepdim=True)
        return centered*torch.rsqrt(centered.square().mean(-1, keepdim=True)+1e-5)*state[prefix+'.weight']+state[prefix+'.bias']

    token = norm(tokens, 'token_evidence.token_norm')
    keys = F.linear(token, state['token_evidence.key.weight'], state['token_evidence.key.bias'])
    values = F.gelu(F.linear(token, state['token_evidence.value.weight'], state['token_evidence.value.bias']))

    def hook(module, args, hidden):
        query = F.linear(norm(hidden, 'token_evidence.query_norm'), state['token_evidence.query.weight'], state['token_evidence.query.bias'])
        logits = torch.einsum('bd,bnd->bn', query, keys)/(32**.5)
        positive = (logits-logits.max(1, keepdim=True).values).exp()
        attention = positive/positive.sum(1, keepdim=True)
        pooled = torch.einsum('bn,bnd->bd', attention, values)
        return hidden+F.linear(pooled, state['token_evidence.readout.weight'], state['token_evidence.readout.bias'])/10

    plain = ordinary(source).to(registers.device).eval().requires_grad_(False)
    handle = plain.encoder[1].register_forward_hook(hook)
    try:
        result = plain(projected)
    finally:
        handle.remove()
    result.update(details)
    return result

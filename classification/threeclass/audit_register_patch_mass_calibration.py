"""Independent per-head mass-weighted patch update and frozen head/tail reconstruction."""
import torch
from torch.nn import functional as F
from audit_register_patch_query import finish
from audit_large_token_evidence import ordinary
from experiment import KEYS


def independent_correct(source, tail_state, adapter_state, bank, raw_registers, config, fit_mass_mean):
    query = bank['query_raw'].double()
    keys = bank['patch_keys'].double()
    values = bank['patch_values'].double()
    logits = torch.einsum('bshqd,bshkd->bshqk', query, keys)/8
    positive = torch.exp(logits-logits.max(-1, keepdim=True).values)
    attention = positive/positive.sum(-1, keepdim=True)
    down = adapter_state['value_down'].double()
    gate = adapter_state['value_gate'].double()
    up = adapter_state['value_up'].double()
    updates = []
    for head in range(16):
        source_head = 0 if config['kind'] == 'shared' else head
        value = values[:, :, head]
        amplitude = torch.sqrt(torch.sum(value*value, dim=-1, keepdim=True)/64+1e-6)
        normalized = value/amplitude
        content = F.linear(normalized, down[source_head])
        control = F.linear(normalized, gate[source_head])
        code = (content*torch.sigmoid(content))*control
        updates.append(F.linear(code, up[source_head])*amplitude*(config['alpha']/config['rank']))
    delta_values = torch.stack(updates, 2)
    difference = torch.einsum('bshqk,bshkd->bshqd', attention, delta_values)
    conditional_difference = difference
    if config['mass_mode'] == 'source_mass':
        mass = bank['full_selected_mass'].to(dtype=torch.float64)
    else:
        mass = torch.as_tensor(fit_mass_mean, dtype=torch.float64, device=query.device)[None, :, None, None].expand(query.shape[:-1])
    difference = difference*mass.unsqueeze(-1)
    flat = difference.transpose(2, 3).reshape(len(query), 2, 4, 1024).float()
    branch = F.linear(flat, tail_state['proj.weight'])*tail_state['gamma1']
    before = bank['postattention'].float()
    correction = finish(tail_state, before+branch)-finish(tail_state, before)
    corrected = (raw_registers+torch.cat((correction[:, 0], correction[:, 1]), -1)).half().float()
    return corrected, {'selected_attention': attention,
        'baseline_pool': torch.einsum('bshqk,bshkd->bshqd', attention, values),
        'conditional_value_pool_difference': conditional_difference, 'applied_attention_mass': mass,
        'value_pool_difference': difference, 'pool_difference': flat,
        'attention_difference': branch, 'tail_difference': correction}


def independent_probability(source, tail_state, adapter_state, raw, bank, config, fit_mass_mean):
    registers, details = independent_correct(source, tail_state, adapter_state, bank, raw['side_tokens'], config, fit_mass_mean)
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

    plain = ordinary(source).to(registers.device).eval()
    handle = plain.encoder[1].register_forward_hook(hook)
    try:
        result = plain(projected)
    finally:
        handle.remove()
    result.update(details)
    return result

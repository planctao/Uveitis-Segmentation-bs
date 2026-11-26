"""Independent per-head mass-weighted patch update and frozen head/tail reconstruction."""
import torch
from torch.nn import functional as F
from audit_register_patch_query import finish
from audit_large_token_evidence import ordinary
from experiment import KEYS


def independent_correct(source,tail_state,adapter_state,bank,raw_registers,config):
    query=bank['query_raw'].to(dtype=torch.float64)
    prefix='query16_' if config['selection']=='query_top16' else 'shared16_'
    keys=bank[prefix+'keys'].double();values=bank[prefix+'values'].double()
    mass=bank[prefix+'mass'].to(dtype=torch.float64)
    head_attention=[];head_pool=[];head_delta=[]
    for h in range(16):
        head=0 if config['kind']=='shared' else h
        down=adapter_state['value_down'][head].double();gate=adapter_state['value_gate'][head].double();up=adapter_state['value_up'][head].double()
        qs=[];pools=[];deltas=[]
        for q in range(4):
            k=keys[:,:,h,q] if config['selection']=='query_top16' else keys[:,:,h]
            v=values[:,:,h,q] if config['selection']=='query_top16' else values[:,:,h]
            logits=torch.einsum('bsd,bskd->bsk',query[:,:,h,q],k)/8
            positive=torch.exp(logits-logits.max(-1,keepdim=True).values)
            weight=positive/positive.sum(-1,keepdim=True)
            rho=torch.sqrt(torch.sum(v*v,-1,keepdim=True)/64+1e-6)
            z=v/rho;content=F.linear(z,down);control=F.linear(z,gate)
            delta=F.linear((content*torch.sigmoid(content))*control,up)*rho*(config['alpha']/config['rank'])
            qs.append(weight);pools.append(torch.einsum('bsk,bskd->bsd',weight,v));deltas.append(torch.einsum('bsk,bskd->bsd',weight,delta))
        head_attention.append(torch.stack(qs,2));head_pool.append(torch.stack(pools,2));head_delta.append(torch.stack(deltas,2))
    attention=torch.stack(head_attention,2);pool=torch.stack(head_pool,2);conditional=torch.stack(head_delta,2)
    difference=conditional*mass.unsqueeze(-1)
    flat=difference.transpose(2,3).reshape(len(query),2,4,1024).float()
    branch=F.linear(flat,tail_state['proj.weight'])*tail_state['gamma1']
    before=bank['postattention'].float()
    correction=finish(tail_state,before+branch)-finish(tail_state,before)
    corrected=(raw_registers+torch.cat((correction[:,0],correction[:,1]),-1)).half().float()
    return corrected,{'selected_attention':attention,'baseline_pool':pool,'conditional_value_pool_difference':conditional,
        'applied_attention_mass':mass,'value_pool_difference':difference,'pool_difference':flat,
        'attention_difference':branch,'tail_difference':correction}


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

    plain = ordinary(source).to(registers.device).eval()
    handle = plain.encoder[1].register_forward_hook(hook)
    try:
        result = plain(projected)
    finally:
        handle.remove()
    result.update(details)
    return result

"""Independent source reconstruction, patch pooling, probability ratio and fine logits."""
import torch
from torch.nn import functional as F
from experiment import KEYS
from audit_large_token_evidence import ordinary


def source_probability(source,raw):
    device=raw['global'].device
    state={name:value.to(device) for name,value in source['state_dict'].items()}
    normalized={key:((value-torch.as_tensor(source['stats'][key]['mean'],device=device))/
        torch.as_tensor(source['stats'][key]['std'],device=device)).clamp(-10,10) for key,value in raw.items()}
    projected={}
    for key in KEYS:
        z=normalized[key];center,basis=state['normal_center_'+key],state['normal_basis_'+key]
        projected[key]=(z-((z-center)@basis)@basis.T).clamp(-10,10)
    center,basis=state['side_normal_center'],state['side_normal_basis']
    tokens=normalized['side_tokens'];tokens=(tokens-((tokens-center)@basis)@basis.T).clamp(-10,10)
    def norm(value,prefix):
        centered=value-value.sum(-1,keepdim=True)/value.shape[-1]
        variance=centered.square().sum(-1,keepdim=True)/value.shape[-1]
        return centered*torch.rsqrt(variance+1e-5)*state[prefix+'.weight']+state[prefix+'.bias']
    token=norm(tokens,'token_evidence.token_norm')
    keys=F.linear(token,state['token_evidence.key.weight'],state['token_evidence.key.bias'])
    values=F.gelu(F.linear(token,state['token_evidence.value.weight'],state['token_evidence.value.bias']))
    def hook(module,args,hidden):
        q=F.linear(norm(hidden,'token_evidence.query_norm'),state['token_evidence.query.weight'],state['token_evidence.query.bias'])
        logits=(q[:,None]*keys).sum(-1)/(32**.5)
        positive=(logits-logits.amax(1,keepdim=True)).exp();attention=positive/positive.sum(1,keepdim=True)
        pool=(attention[...,None]*values).sum(1)
        return hidden+F.linear(pool,state['token_evidence.readout.weight'],state['token_evidence.readout.bias'])/10
    plain=ordinary(source).to(device).eval();handle=plain.encoder[1].register_forward_hook(hook)
    try:result=plain(projected)
    finally:handle.remove()
    return result,state['subtype_to_class']


def independent_probability(source,tail_state,parameters,raw,bank,config):
    result,mapping=source_probability(source,raw)
    def unit(value):
        return value/torch.sqrt(value.pow(2).sum(-1,keepdim=True)/value.shape[-1]+1e-6)
    q=unit(unit(bank['query_raw'].float()).sum(-2)/4)
    k,v=unit(bank['patch_keys'].float()),unit(bank['patch_values'].float())
    logits=(q[...,None,:]*k).sum(-1)/8
    positive=(logits-logits.amax(-1,keepdim=True)).exp()
    source_weights=positive/positive.sum(-1,keepdim=True)
    weights=[];pools=[]
    for cls in range(2):
        if config['pooling']=='uniform':a=torch.ones_like(source_weights)/64
        elif config['pooling']=='source':a=source_weights
        else:
            query=q+parameters['patch_queries'][cls][None,None]
            scores=(query[...,None,:]*k).sum(-1)/8
            e=(scores-scores.amax(-1,keepdim=True)).exp();a=e/e.sum(-1,keepdim=True)
        effective=a-source_weights if config['pooling']=='contrast' else a
        pools.append((effective[...,None]*v).sum(-2));weights.append(a)
    evidence=torch.stack(pools,1)
    flat=evidence.flatten(2);readout=parameters['patch_readout'].flatten(1)
    scores=(flat*readout[None]).sum(-1)+parameters['patch_bias']
    shifts=torch.stack([2*torch.tanh(scores[:,0]),2*torch.tanh(scores[:,1]),torch.zeros_like(scores[:,0])],1)
    ratio=shifts.double().exp();original=result['p']
    denominator=1+(original.double()*(ratio-1)).sum(1,keepdim=True)
    corrected=(original.double()*ratio/denominator).to(original.dtype)
    fine_shift=torch.stack([shifts[:,int(row.argmax())] for row in mapping],1)
    result.update(patch_source_probability=original,patch_source_subtype=result['subtype'],
        patch_source_query=q,patch_source_attention=source_weights,patch_attention=torch.stack(weights,3),
        patch_evidence=evidence,patch_scores=scores,patch_class_shift=shifts,
        p=corrected,subtype=result['subtype']+fine_shift,flat=corrected.clamp(min=1e-7).log())
    return result

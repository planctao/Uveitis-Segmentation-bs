"""Functional frozen-tail oracle independent of the query-adapter forward."""
import torch
from torch.nn import functional as F
from audit_large_token_evidence import ordinary,independent_forward as independent_head
from experiment import KEYS


def finish(state,hidden):
    def norm(value,prefix):
        centered=value-value.sum(-1,keepdim=True)/1024
        return centered*torch.rsqrt(centered.square().sum(-1,keepdim=True)/1024+1e-5)*state[prefix+'.weight']+state[prefix+'.bias']
    code=F.gelu(F.linear(norm(hidden,'norm2'),state['mlp.fc1.weight'],state['mlp.fc1.bias']))
    branch=F.linear(code,state['mlp.fc2.weight'],state['mlp.fc2.bias'])
    return norm(hidden+state['gamma2']*branch,'norm')


def independent_correct(source,tail_state,query_state,bank,raw_registers,config):
    context=bank['query_input'].float()
    if config['kind']=='shared':context=context.sum(2,keepdim=True).expand_as(context)/4
    code=F.linear(context,query_state['down.weight']);change=F.linear(code,query_state['up.weight'])*(config['alpha']/config['rank'])
    change=change.reshape(len(context),2,4,16,64).transpose(2,3)
    change=change.double()
    q=bank['query_raw'].double();k=bank['patch_keys'].double();v=bank['patch_values'].double()
    def softmax(logits):
        positive=torch.exp(logits-logits.max(-1,keepdim=True).values)
        return positive/positive.sum(-1,keepdim=True)
    base=softmax(torch.einsum('bshqd,bshkd->bshqk',q,k)/8)
    adapted=softmax(torch.einsum('bshqd,bshkd->bshqk',q+change,k)/8)
    difference=torch.einsum('bshqk,bshkd->bshqd',adapted,v)-torch.einsum('bshqk,bshkd->bshqd',base,v)
    difference=difference.transpose(2,3).reshape(len(q),2,4,1024).float()
    branch=F.linear(difference,tail_state['proj.weight'])*tail_state['gamma1']
    before=bank['postattention'].float();correction=finish(tail_state,before+branch)-finish(tail_state,before)
    registers=(raw_registers+torch.cat((correction[:,0],correction[:,1]),-1)).half().float()
    return registers,{'delta_query':change,'selected_attention':adapted,'baseline_attention':base,
        'pool_difference':difference,'attention_difference':branch,'tail_difference':correction}


def independent_probability(source,tail_state,query_state,raw,bank,config):
    registers,details=independent_correct(source,tail_state,query_state,bank,raw['side_tokens'],config)
    stats=source['stats'];normalized={key:((value-torch.as_tensor(stats[key]['mean'],device=value.device))/torch.as_tensor(stats[key]['std'],device=value.device)).clamp(-10,10) for key,value in {**raw,'side_tokens':registers}.items()}
    # Projection and complete original head are restored functionally using its frozen tensors.
    plain=ordinary(source).to(registers.device).eval()
    # Existing independent_head projects through NumPy, which would detach corrected registers.
    # Restore the side projection in torch, then invoke the original plain encoder with a functional hook.
    state={key:value.to(registers.device) for key,value in source['state_dict'].items()}
    projected={}
    for key in KEYS:
        z=normalized[key];center=state['normal_center_'+key];basis=state['normal_basis_'+key]
        projected[key]=(z-((z-center)@basis)@basis.T).clamp(-10,10)
    tokens=normalized['side_tokens'];center=state['side_normal_center'];basis=state['side_normal_basis']
    tokens=(tokens-((tokens-center)@basis)@basis.T).clamp(-10,10)
    def norm(value,width,prefix):
        centered=value-value.mean(-1,keepdim=True)
        return centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)*state[prefix+'.weight']+state[prefix+'.bias']
    token=norm(tokens,2048,'token_evidence.token_norm')
    keys=F.linear(token,state['token_evidence.key.weight'],state['token_evidence.key.bias'])
    values=F.gelu(F.linear(token,state['token_evidence.value.weight'],state['token_evidence.value.bias']))
    def hook(module,args,hidden):
        query=F.linear(norm(hidden,256,'token_evidence.query_norm'),state['token_evidence.query.weight'],state['token_evidence.query.bias'])
        logits=(query[:,None]*keys).sum(-1)/(32**.5);p=(logits-logits.max(1,keepdim=True).values).exp();p=p/p.sum(1,keepdim=True)
        pooled=(p[:,:,None]*values).sum(1)
        return hidden+F.linear(pooled,state['token_evidence.readout.weight'],state['token_evidence.readout.bias'])/10
    handle=plain.encoder[1].register_forward_hook(hook)
    try:out=plain(projected)
    finally:handle.remove()
    out.update(details);return out

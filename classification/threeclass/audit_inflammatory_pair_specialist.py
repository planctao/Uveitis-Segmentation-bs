"""Independent frozen head plus conditional ratio correction, no production odds formula."""
import torch
from torch.nn import functional as F
from audit_large_token_evidence import ordinary
from experiment import KEYS


def independent_probability(source, tail_state, adapter_state, raw, bank, config):
    registers=raw['side_tokens'];details={}
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
        details['token_pooled']=pooled
        return hidden+F.linear(pooled, state['token_evidence.readout.weight'], state['token_evidence.readout.bias'])/10

    plain = ordinary(source).to(registers.device).eval()
    handle = plain.encoder[1].register_forward_hook(hook)
    try:
        result = plain(projected)
    finally:
        handle.remove()
    original=result['p']
    evidence=result['embedding'] if config['evidence']=='hidden' else torch.cat([result['embedding'],details['token_pooled']],-1)
    code=F.linear(evidence,adapter_state['specialist.0.weight'],adapter_state['specialist.0.bias'])
    if config['hidden']:
        code=F.linear(F.gelu(code),adapter_state['specialist.2.weight'],adapter_state['specialist.2.bias'])
    score=code[:,0];shift=2*torch.tanh(score.double())
    pair=original[:,:2].double();mass=pair.sum(1)
    positive=pair[:,0]*torch.exp(shift)
    conditional=positive/(positive+pair[:,1]).clamp_min(torch.finfo(torch.float64).tiny)
    baseline=pair[:,0]/mass.clamp_min(torch.finfo(torch.float64).tiny)
    transfer=mass*(conditional-baseline)
    result['p']=torch.stack([(pair[:,0]+transfer).to(original.dtype),(pair[:,1]-transfer).to(original.dtype),original[:,2]],1)
    result['flat']=result['p'].clamp(min=1e-7).log()
    result.update(details,pair_base_probability=original,pair_score=score,pair_shift=shift,pair_transferred_mass=transfer)
    return result

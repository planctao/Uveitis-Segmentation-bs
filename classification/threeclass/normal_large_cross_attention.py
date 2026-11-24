"""Normal-B hidden queries select complementary normal-L regional evidence."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS,loss_fn
from normal_large_residual import NormalLargeResidualModel
from normal_subspace import independent_transform


class LargeCrossAttention(nn.Module):
    def __init__(self,rank,kind):
        super().__init__();assert rank in [16,32] and kind in ['mean','attention']
        self.rank=rank;self.kind=kind
        self.norm=nn.LayerNorm(2048);self.query_norm=nn.LayerNorm(256)
        self.key=nn.Linear(2048,rank);self.value=nn.Linear(2048,rank)
        self.query=nn.Linear(256,rank);self.readout=nn.Linear(4*rank,256)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,large,hidden):
        normalized={k:self.norm(v) for k,v in large.items()}
        query=self.query(self.query_norm(hidden))
        keys=self.key(normalized['regions']);values={k:F.gelu(self.value(v)) for k,v in normalized.items()}
        scores=torch.einsum('bd,btd->bt',query,keys)/(self.rank**.5)
        if self.kind=='mean':scores=scores*0
        attention=scores.softmax(1);pooled=torch.sum(attention[:,:,None]*values['regions'],dim=1)
        summary=torch.cat([values['global'],pooled,values['mid'],values['dispersion']],dim=-1)
        return .1*self.readout(summary),summary,attention,query


class NormalLargeCrossAttentionModel(NormalLargeResidualModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry,witness_path)
        with torch.random.fork_rng(devices=[]):
            self.large_residual=LargeCrossAttention(cfg['large_cross_rank'],cfg['large_cross_kind'])

    def _add(self,module,args,hidden):
        self._base_hidden=hidden
        self._delta,self._residual_input,self._cross_attention,self._cross_query=self.large_residual(self._large,hidden)
        return hidden+self._delta

    def forward(self,x):
        out=super().forward(x)
        out.update(large_cross_attention=self._cross_attention,large_cross_query=self._cross_query)
        return out


def cross_attention_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg);context=out.get('_large_residual_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'rank':np.array(cfg['large_cross_rank']),'kind':np.array(cfg['large_cross_kind']),
            'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],'subtype':out['subtype']}
        tensors.update({k:out[k] for k in ['large_residual_input','large_residual_hidden','large_residual_delta','large_cross_attention','large_cross_query']})
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss


def independent_forward(plain,raw,state,cfg,proof=None):
    device=next(plain.parameters()).device;projected={};large={};details={}
    for key in KEYS:
        projected[key]=torch.from_numpy(independent_transform(raw[key][...,:1536],state['normal_center_'+key].detach().cpu().numpy(),state['normal_basis_'+key].detach().cpu().numpy(),1.)).to(device)
        large[key]=torch.from_numpy(independent_transform(raw[key][...,1536:],state['large_normal_center_'+key].detach().cpu().numpy(),state['large_normal_basis_'+key].detach().cpu().numpy(),1.)).to(device)
    get=lambda name:state['large_residual.'+name].to(device)
    def manual_norm(value,weight,bias):
        centered=value-value.sum(-1,keepdim=True)/value.shape[-1]
        return centered*torch.rsqrt(centered.square().sum(-1,keepdim=True)/value.shape[-1]+1e-5)*weight+bias
    normalized={k:manual_norm(v,get('norm.weight'),get('norm.bias')) for k,v in large.items()}
    values={k:F.gelu(F.linear(v,get('value.weight'),get('value.bias'))) for k,v in normalized.items()}
    keys=F.linear(normalized['regions'],get('key.weight'),get('key.bias'))
    def add(module,args,hidden):
        query=F.linear(manual_norm(hidden,get('query_norm.weight'),get('query_norm.bias')),get('query.weight'),get('query.bias'))
        scores=(query[:,None]*keys).sum(-1)/(cfg['large_cross_rank']**.5)
        if cfg['large_cross_kind']=='mean':scores=scores*0
        attention=(scores-scores.max(1,keepdim=True).values).exp();attention=attention/attention.sum(1,keepdim=True)
        pooled=torch.einsum('bt,btd->bd',attention,values['regions'])
        summary=torch.cat([values['global'],pooled,values['mid'],values['dispersion']],dim=-1)
        delta=.1*F.linear(summary,get('readout.weight'),get('readout.bias'))
        details.update(large_residual_input=summary,large_residual_hidden=hidden,large_residual_delta=delta,large_cross_attention=attention,large_cross_query=query)
        return hidden+delta
    handle=plain.encoder[1].register_forward_hook(add)
    try:
        if proof is None:out=plain(projected)
        else:
            from audit_normal_tangent import plain_replay
            out=plain_replay(plain,projected,proof)
    finally:handle.remove()
    out.update(details);return out

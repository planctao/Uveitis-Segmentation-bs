"""Separate normal-B classifier and zero-initialized normal-L residual tower."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import KEYS,Model,loss_fn
from normal_subspace import NormalSuppressedModel,fit_normal_geometry,independent_transform


def fit_separate_geometry(frame,normalized):
    base=fit_normal_geometry(frame,{k:v[...,:1536] for k,v in normalized.items()})
    large=fit_normal_geometry(frame,{k:v[...,1536:] for k,v in normalized.items()})
    assert np.array_equal(base['normal_patients'],large['normal_patients'])
    return {**base,**{'large_'+k:v for k,v in large.items()}}


class LargeResidual(nn.Module):
    def __init__(self,rank,scope):
        super().__init__();assert rank in [16,32] and scope in ['global','all']
        self.scope=scope;self.norm=nn.LayerNorm(8192)
        self.down=nn.Linear(8192,rank);self.readout=nn.Linear(rank,256)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,x):
        values=[x['global']]*4 if self.scope=='global' else [x['global'],x['regions'].mean(1),x['mid'],x['dispersion']]
        fused=torch.cat(values,dim=-1)
        return .1*self.readout(F.gelu(self.down(self.norm(fused)))),fused


class NormalLargeResidualModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        assert all(dims[k]==3584 for k in KEYS)
        base_dims={**dims,**{k:1536 for k in KEYS}}
        super().__init__(cfg,base_dims,n_diseases,geometry)
        assert cfg['hidden']==256 and cfg['normal_strength']==1. and cfg['normal_rank']==32
        for key in KEYS:
            self.register_buffer('large_normal_center_'+key,torch.zeros(2048))
            self.register_buffer('large_normal_basis_'+key,torch.zeros(2048,32))
            if geometry is not None:
                getattr(self,'large_normal_center_'+key).copy_(torch.as_tensor(geometry['large_'+key+'_center']))
                getattr(self,'large_normal_basis_'+key).copy_(torch.as_tensor(geometry['large_'+key+'_basis']))
        with torch.random.fork_rng(devices=[]):self.large_residual=LargeResidual(cfg['large_residual_rank'],cfg['large_residual_scope'])
        self.encoder[1].register_forward_hook(self._add)
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def _add(self,module,args,hidden):
        self._base_hidden=hidden;self._delta,self._residual_input=self.large_residual(self._large)
        return hidden+self._delta

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        self._large={}
        for key,value in x.items():
            large=value[...,1536:];center=getattr(self,'large_normal_center_'+key);basis=getattr(self,'large_normal_basis_'+key)
            self._large[key]=(large-((large-center)@basis)@basis.T).clamp(-10,10)
        out=super().forward({k:v[...,:1536] for k,v in x.items()})
        out.update(large_residual_input=self._residual_input,large_residual_hidden=self._base_hidden,large_residual_delta=self._delta)
        if context is not None:out['_large_residual_witness_context']=context
        return out


def large_residual_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg);context=out.get('_large_residual_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'rank':np.array(cfg['large_residual_rank']),'scope':np.array(cfg['large_residual_scope']),
            'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],'subtype':out['subtype']}
        tensors.update({k:out[k] for k in ['large_residual_input','large_residual_hidden','large_residual_delta']})
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss


def independent_forward(plain,normalized,state,cfg):
    base={};large={}
    for key,value in normalized.items():
        base[key]=independent_transform(value[...,:1536],state['normal_center_'+key].cpu().numpy(),state['normal_basis_'+key].cpu().numpy(),1.)
        large[key]=independent_transform(value[...,1536:],state['large_normal_center_'+key].cpu().numpy(),state['large_normal_basis_'+key].cpu().numpy(),1.)
    device=next(plain.parameters()).device;base={k:torch.as_tensor(v,device=device) for k,v in base.items()}
    large={k:torch.as_tensor(v,device=device) for k,v in large.items()}
    values=[large['global']]*4 if cfg['large_residual_scope']=='global' else [large['global'],large['regions'].sum(1)/9,large['mid'],large['dispersion']]
    fused=torch.cat(values,dim=-1);parameters={k:v for k,v in plain.named_parameters()}
    # Callers may supply differentiable adapter parameters for full gradient checks.
    get=lambda name:state['large_residual.'+name].to(device)
    mean=fused.mean(-1,keepdim=True);variance=(fused-mean).square().mean(-1,keepdim=True)
    normalized_side=(fused-mean)*torch.rsqrt(variance+1e-5)*get('norm.weight')+get('norm.bias')
    hidden=F.gelu(F.linear(normalized_side,get('down.weight'),get('down.bias')))
    delta=.1*F.linear(hidden,get('readout.weight'),get('readout.bias'))
    captured={}
    def add(module,args,output):captured['hidden']=output;return output+delta
    handle=plain.encoder[1].register_forward_hook(add)
    try:out=plain(base)
    finally:handle.remove()
    out.update(large_residual_input=fused,large_residual_hidden=captured['hidden'],large_residual_delta=delta)
    return out

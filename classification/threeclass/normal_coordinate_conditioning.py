"""Bounded affine hidden conditioning on fit-normal PCA coordinates.

Normal-axis deletion stays fixed. A small zero-initialized bypass can use the
removed coordinates as context; this is a dataset-specific use of existing FiLM.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiment import KEYS,loss_fn
from normal_subspace import NormalSuppressedModel


class CoordinateConditioning(nn.Module):
    def __init__(self,rank=16):
        super().__init__()
        self.down=nn.Linear(128,rank)
        self.readout=nn.Linear(rank,512)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,coordinates,hidden,mode):
        scale,shift=self.readout(torch.nn.functional.gelu(self.down(coordinates))).chunk(2,-1)
        gamma=.1*scale.tanh() if mode=='affine' else torch.zeros_like(scale)
        beta=.1*shift.tanh()
        return gamma*hidden+beta,gamma,beta


class NormalCoordinateModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['hidden']==256 and cfg['normal_strength']==1. and cfg['normal_rank']==32
        assert cfg['normal_coordinate_mode'] in ['shift','affine']
        assert isinstance(cfg['normal_coordinate_conditioned'],bool)
        with torch.random.fork_rng(devices=[]):self.coordinate_adapter=CoordinateConditioning(16)
        for key in KEYS:
            self.register_buffer('coordinate_deviation_'+key,torch.ones(32))
            if geometry is not None:
                getattr(self,'coordinate_deviation_'+key).copy_(torch.as_tensor(geometry[key+'_eigenvalues']).sqrt().clamp(min=.05))
        self.encoder[1].register_forward_hook(self._condition)
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def _condition(self,module,args,hidden):
        self._hidden=hidden
        context=self._coordinates if self.cfg['normal_coordinate_conditioned'] else torch.zeros_like(self._coordinates)
        self._delta,self._gamma,self._beta=self.coordinate_adapter(context,hidden,self.cfg['normal_coordinate_mode'])
        return hidden+self._delta

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        parts=[]
        for key in KEYS:
            value=x[key].mean(1) if x[key].ndim==3 else x[key]
            coordinate=(value-getattr(self,'normal_center_'+key))@getattr(self,'normal_basis_'+key)
            parts.append((coordinate/getattr(self,'coordinate_deviation_'+key)).clamp(-5,5))
        self._coordinates=torch.cat(parts,-1)
        out=super().forward(x)
        out.update(coordinate_context=self._coordinates,coordinate_hidden=self._hidden,coordinate_delta=self._delta,
            coordinate_gamma=self._gamma,coordinate_beta=self._beta)
        if context is not None:out['_coordinate_witness_context']=context
        return out


def coordinate_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg)
    context=out.get('_coordinate_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'mode':np.array(cfg['normal_coordinate_mode']),'conditioned':np.array(cfg['normal_coordinate_conditioned']),
            'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p']}
        tensors.update({k:out[k] for k in ['coordinate_context','coordinate_hidden','coordinate_delta','coordinate_gamma','coordinate_beta']})
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss

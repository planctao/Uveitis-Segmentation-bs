"""Normal-reference centered regional contrasts with D4 group pooling."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import loss_fn
from normal_subspace import NormalSuppressedModel


def d4_permutations():
    grid=np.arange(9).reshape(3,3)
    return np.stack([np.rot90(np.fliplr(grid) if reflected else grid,k).reshape(-1).copy() for reflected in [False,True] for k in range(4)])


class SpatialContrast(nn.Module):
    def __init__(self,rank,kind):
        super().__init__();assert rank in [16,32] and kind in ['mean','d4']
        self.rank=rank;self.kind=kind
        self.norm=nn.LayerNorm(128);self.down=nn.Linear(128,rank)
        self.mix=nn.Linear(9*rank,rank);self.readout=nn.Linear(rank,256)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        self.register_buffer('permutations',torch.from_numpy(d4_permutations()).long())

    def forward(self,tokens):
        assert tokens.shape[1:]==(9,128)
        centered=tokens-tokens.mean(1,keepdim=True)
        values=F.gelu(self.down(self.norm(centered)))
        if self.kind=='d4':views=values[:,self.permutations].flatten(2)
        else:views=values.mean(1,keepdim=True).expand(-1,9,-1).reshape(len(values),1,9*self.rank)
        summary=F.gelu(self.mix(views)).mean(1)
        return .1*self.readout(summary),centered,values,summary


class NormalSpatialContrastModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['hidden']==256 and cfg['normal_rank']==32 and cfg['normal_strength']==1.
        with torch.random.fork_rng(devices=[]):self.spatial_contrast=SpatialContrast(cfg['spatial_contrast_rank'],cfg['spatial_contrast_kind'])
        self.region_proj.register_forward_hook(self._capture);self.encoder[1].register_forward_hook(self._add)
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def _capture(self,module,args,output):self._tokens=F.gelu(output)

    def _add(self,module,args,hidden):
        self._hidden=hidden
        self._delta,self._centered,self._values,self._summary=self.spatial_contrast(self._tokens)
        return hidden+self._delta

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        out=super().forward(x)
        out.update(spatial_tokens=self._tokens,spatial_hidden=self._hidden,spatial_delta=self._delta,
            spatial_centered=self._centered,spatial_values=self._values,spatial_summary=self._summary)
        if context is not None:out['_spatial_contrast_witness_context']=context
        return out


def spatial_contrast_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg);context=out.get('_spatial_contrast_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'rank':np.array(cfg['spatial_contrast_rank']),'kind':np.array(cfg['spatial_contrast_kind']),
            'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],'subtype':out['subtype']}
        tensors.update({k:out[k] for k in ['spatial_tokens','spatial_hidden','spatial_delta','spatial_centered','spatial_values','spatial_summary']})
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss

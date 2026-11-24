"""Shared-dropout consistency under fit-normal patient spatial variation."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment import Model,loss_fn
from normal_subspace import NormalSuppressedModel
from normal_spatial_reference import fit_spatial_geometry,spatial_patient_data


def fit_consistency_geometry(frame,normalized):
    g=fit_spatial_geometry(frame,normalized);names,x,target=spatial_patient_data(frame,normalized,g)
    q=np.clip(((x-g['global_center'])@g['global_basis'][:,:8])/g['spatial_deviation'],-5,5)
    predicted=g['spatial_center'][None]+np.einsum('nk,ktd->ntd',q-g['spatial_qmean'],g['spatial_coefficients'])
    residual=target-predicted;residual-=residual.mean(0,keepdims=True);residual-=residual.mean(1,keepdims=True)
    g['consistency_bank']=residual.astype('float32');return g


class NormalSpatialConsistencyModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_rank']==32 and cfg['normal_strength']==1.
        assert cfg['normal_consistency_level'] in ['coarse','disease'] and cfg['normal_consistency_weight']>=0
        count=dims.get('normal_consistency_patients',len(geometry['normal_patients']) if geometry is not None else 0);assert count>1
        dims['normal_consistency_patients']=count
        self.register_buffer('consistency_bank',torch.zeros(count,9,dims['regions']))
        if geometry is not None:self.consistency_bank.copy_(torch.as_tensor(geometry['consistency_bank']))
        self.spatial_rng=np.random.default_rng(20283224+cfg.get('seed',0))
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def forward(self,x):
        if not self.training or not torch.is_grad_enabled():return super().forward(x)
        device=x['global'].device;before_cpu=torch.random.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
        context=None
        if self.witness_path is not None and not self.witness_written:
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':before_cpu,
                'rng_cuda':before_cuda,'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        base=self.transform(x);clean=Model.forward(self,base)
        after_cpu=torch.random.get_rng_state().clone();after_cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
        n=len(x['global']);count=len(self.consistency_bank);first=self.spatial_rng.integers(0,count,n)
        second=(first+self.spatial_rng.integers(1,count,n))%count
        a=torch.as_tensor(first,device=device);b=torch.as_tensor(second,device=device)
        delta=.5*(self.consistency_bank[a]-self.consistency_bank[b])/(2.**.5)
        augmented={**base,'regions':(base['regions']+delta).clamp(-10,10)}
        # Both views use identical dropout masks, and only the original draw stream advances.
        torch.random.set_rng_state(before_cpu)
        if before_cuda is not None:torch.cuda.set_rng_state(before_cuda,device)
        try:other=Model.forward(self,augmented)
        finally:
            torch.random.set_rng_state(after_cpu)
            if after_cuda is not None:torch.cuda.set_rng_state(after_cuda,device)
        clean['_consistency_view']=other
        if context is not None:
            context.update(first=first,second=second,augmented_regions=augmented['regions']);clean['_consistency_witness_context']=context
        return clean


def spatial_consistency_loss(out,y,disease,weights,cfg):
    base=loss_fn(out,y,disease,weights,cfg);view=out['_consistency_view']
    a,b=(out['p'],view['p']) if cfg['normal_consistency_level']=='coarse' else (F.softmax(out['subtype'],-1),F.softmax(view['subtype'],-1))
    mean=(a+b)/2;logmean=mean.clamp(min=1e-7).log()
    rows=.5*((a*(a.clamp(min=1e-7).log()-logmean)).sum(1)+(b*(b.clamp(min=1e-7).log()-logmean)).sum(1))
    penalty=(rows*weights['leaf']).mean();loss=base+cfg['normal_consistency_weight']*penalty
    out['_consistency_details']={'base':base,'penalty':penalty}
    context=out.get('_consistency_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),'level':np.array(cfg['normal_consistency_level']),
            'strength':np.array(cfg['normal_consistency_weight']),'loss':np.array(float(loss.detach())),'base_loss':np.array(float(base.detach())),
            'penalty':np.array(float(penalty.detach())),'first':context['first'],'second':context['second'],'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],'subtype':out['subtype'],
            'view_probability':view['p'],'view_subtype':view['subtype'],'augmented_regions':context['augmented_regions']}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss

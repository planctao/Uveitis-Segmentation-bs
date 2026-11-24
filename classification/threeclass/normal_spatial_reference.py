"""Fit-normal patient spatial reference with optional global-coordinate context."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS,loss_fn
from normal_subspace import NormalSuppressedModel,fit_normal_geometry,independent_transform


def spatial_patient_data(frame,normalized,geometry):
    normal=frame.label.eq(2).to_numpy();patients=frame.patient_group.to_numpy();names=geometry['normal_patients'].tolist()
    region=independent_transform(normalized['regions'][normal],geometry['regions_center'],geometry['regions_basis'],1.).astype('float64')
    spatial=region-region.mean(1,keepdims=True)
    global_values=normalized['global'][normal].astype('float64');normal_patients=patients[normal]
    x=[];target=[]
    for name in names:
        ids=np.flatnonzero(normal_patients==name);assert len(ids)
        x.append(global_values[ids].mean(0));target.append(spatial[ids].mean(0))
    return names,np.stack(x),np.stack(target)


def fit_spatial_geometry(frame,normalized):
    geometry=fit_normal_geometry(frame,normalized)
    names,x,target=spatial_patient_data(frame,normalized,geometry)
    deviation=np.maximum(np.sqrt(geometry['global_eigenvalues'][:8]),.05)
    q=np.clip(((x-geometry['global_center'])@geometry['global_basis'][:,:8])/deviation,-5,5)
    qmean=q.mean(0);center=target.mean(0);a=q-qmean
    # Ridge on mean squared patient error: lambda=1 per standardized coordinate.
    w=np.linalg.solve(a.T@a+len(a)*np.eye(8),a.T@(target-center).reshape(len(a),-1)).reshape(8,*target.shape[1:])
    geometry.update(spatial_qmean=qmean.astype('float32'),spatial_deviation=deviation.astype('float32'),
        spatial_center=center.astype('float32'),spatial_coefficients=w.astype('float32'))
    return geometry


def independent_spatial_transform(data,geometry,conditioned,strength):
    base={k:independent_transform(data[k],geometry[k+'_center'],geometry[k+'_basis'],1.) for k in KEYS}
    if strength==0:return base
    x=data['global'].astype('float64');basis=geometry['global_basis'][:,:8].astype('float64')
    q=np.clip(np.einsum('nd,dk->nk',x-geometry['global_center'],basis)/geometry['spatial_deviation'],-5,5)
    prediction=np.broadcast_to(geometry['spatial_center'],base['regions'].shape).astype('float64').copy()
    if conditioned:prediction+=np.einsum('nk,ktd->ntd',q-geometry['spatial_qmean'],geometry['spatial_coefficients'].astype('float64'))
    prediction-=prediction.mean(1,keepdims=True)
    base['regions']=np.clip(base['regions'].astype('float64')-strength*prediction,-10,10).astype('float32')
    return base


class NormalSpatialModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_rank']==32 and cfg['normal_strength']==1.
        assert isinstance(cfg['normal_spatial_conditioned'],bool) and cfg['normal_spatial_strength'] in [0.,.5,1.]
        self.register_buffer('spatial_qmean',torch.zeros(8));self.register_buffer('spatial_deviation',torch.ones(8))
        self.register_buffer('spatial_center',torch.zeros(9,dims['regions']))
        self.register_buffer('spatial_coefficients',torch.zeros(8,9,dims['regions']))
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False
        if geometry is not None:
            for key in ['spatial_qmean','spatial_deviation','spatial_center','spatial_coefficients']:getattr(self,key).copy_(torch.as_tensor(geometry[key]))

    def transform(self,x):
        base=super().transform(x);strength=self.cfg['normal_spatial_strength']
        if strength==0:return base
        q=(((x['global']-self.normal_center_global)@self.normal_basis_global[:,:8])/self.spatial_deviation).clamp(-5,5)
        prediction=self.spatial_center.unsqueeze(0).expand(len(q),-1,-1)
        if self.cfg['normal_spatial_conditioned']:prediction=prediction+torch.einsum('nk,ktd->ntd',q-self.spatial_qmean,self.spatial_coefficients)
        prediction=prediction-prediction.mean(1,keepdim=True)
        base['regions']=(base['regions']-strength*prediction).clamp(-10,10)
        return base

    def forward(self,x):
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'model':self,'raw_global_all':x['global'],'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        transformed=self.transform(x)
        from experiment import Model
        out=Model.forward(self,transformed)
        if context is not None:
            context['transformed_regions']=transformed['regions'];out['_spatial_witness_context']=context
        return out


def spatial_loss(out,y,disease,weights,cfg):
    loss=loss_fn(out,y,disease,weights,cfg);context=out.get('_spatial_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'conditioned':np.array(cfg['normal_spatial_conditioned']),'strength':np.array(cfg['normal_spatial_strength']),
            'loss':np.array(float(loss.detach())),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],'transformed_regions':context['transformed_regions']}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return loss

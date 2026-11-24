"""Class-conditional risk variance across fit-normal feature groups."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS,loss_fn
from normal_subspace import NormalSuppressedModel,fit_normal_geometry,patient_centers
from normal_atlas import cluster_normals


def fit_group_geometry(frame,normalized,count):
    assert count in [2,4]
    geometry=fit_normal_geometry(frame,normalized)
    names,centers=patient_centers(frame,normalized,'global')
    deviation=np.sqrt(geometry['global_eigenvalues'])
    points=((centers-geometry['global_center'])@geometry['global_basis'])/deviation
    clusters,assignment,iterations=cluster_normals(points,count)
    geometry.update(risk_deviation=deviation.astype('float32'),risk_clusters=clusters.astype('float32'),
        risk_patient_points=points.astype('float32'),risk_patient_assignment=assignment.astype('int64'),
        risk_cluster_iterations=np.asarray(iterations,dtype='int64'))
    assert np.array_equal(names,geometry['normal_patients'])
    return geometry


def independent_groups(global_values,geometry):
    points=(global_values.astype('float64')-geometry['global_center'].astype('float64'))@geometry['global_basis'].astype('float64')
    points/=geometry['risk_deviation'].astype('float64')
    distances=np.stack([np.sum((points-center.astype('float64'))**2,axis=1) for center in geometry['risk_clusters']],axis=1)
    return distances.argmin(1),points


def risk_variance(probability,y,weight,groups,count,smoothing):
    logp=probability.clamp(min=1e-7).log()
    rows=-(1-smoothing)*logp.gather(1,y[:,None]).squeeze(1)-smoothing*logp.mean(1)
    risks=[];masses=[];penalties=[]
    for parent in range(3):
        selected=weight*(y==parent);group_risks=[];group_masses=[]
        for group in range(count):
            w=selected*(groups==group);mass=w.sum()
            group_masses.append(mass);group_risks.append((rows*w).sum()/mass.clamp(min=1e-12))
        risk=torch.stack(group_risks);mass=torch.stack(group_masses);valid=mass>0
        if valid.any():penalties.append(risk[valid].var(unbiased=False))
        risks.append(risk);masses.append(mass)
    penalty=torch.stack(penalties).mean() if penalties else rows.sum()*0
    return penalty,torch.stack(risks),torch.stack(masses)


class NormalGroupRiskModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        count=cfg['normal_risk_groups'];assert count in [2,4] and cfg['normal_risk_weight'] in [0.,.1,.5]
        self.register_buffer('risk_deviation',torch.ones(32))
        self.register_buffer('risk_clusters',torch.zeros(count,32))
        if geometry is not None:
            self.risk_deviation.copy_(torch.as_tensor(geometry['risk_deviation']))
            self.risk_clusters.copy_(torch.as_tensor(geometry['risk_clusters']))
        self.witness_path=None if witness_path is None else Path(witness_path)
        self.witness_written=False

    def forward(self,x):
        points=((x['global']-self.normal_center_global)@self.normal_basis_global)/self.risk_deviation
        groups=(points[:,None]-self.risk_clusters).square().sum(-1).argmin(-1)
        context=None
        if self.training and torch.is_grad_enabled() and self.witness_path is not None and not self.witness_written:
            device=x['global'].device
            context={'input':x,'rng_cpu':torch.random.get_rng_state().clone(),
                'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None,
                'initial_parameter_sha256':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.named_parameters()}}
        out=super().forward(x);out.update(normal_risk_group=groups,normal_risk_points=points)
        if context is not None:out['_risk_context']={'model':self,**context}
        return out


def group_risk_loss(out,y,disease,weights,cfg):
    base=loss_fn(out,y,disease,weights,cfg)
    penalty,risks,masses=risk_variance(out['p'],y,weights['leaf'],out['normal_risk_group'],
        cfg['normal_risk_groups'],cfg.get('label_smoothing',0.))
    total=base+cfg['normal_risk_weight']*penalty
    context=out.get('_risk_context')
    if context is not None:
        model=context['model'];proof={'seed':np.asarray(cfg['seed']),'groups':np.asarray(cfg['normal_risk_groups']),
            'strength':np.asarray(cfg['normal_risk_weight']),'batch_size':np.asarray(len(y)),
            'raw_global_all':context['input']['global'].detach().cpu().numpy(),
            'rng_cpu':context['rng_cpu'].numpy(),'base_loss':np.asarray(float(base.detach())),
            'penalty':np.asarray(float(penalty.detach())),'total_loss':np.asarray(float(total.detach()))}
        tensors={'y':y,'disease':disease,'probability':out['p'],'risk_group':out['normal_risk_group'],
            'risk_points':out['normal_risk_points'],'class_group_risks':risks,'class_group_masses':masses}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for key in KEYS:tensors['input_'+key]=context['input'][key][:8]
        for key,value in tensors.items():proof[key]=value.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for key,value in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+key]=np.asarray(value)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return total

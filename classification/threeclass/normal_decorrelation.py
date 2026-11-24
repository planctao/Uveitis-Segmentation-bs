"""Patient-disease-centered conditional correlation penalty on normal axes.

This estimates linear conditional association only, not causal nuisance or
statistical independence. Inference remains the original normal-reference MLP.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS,loss_fn
from normal_subspace import NormalSuppressedModel


def conditional_correlation(hidden,coordinates,y,disease,patient_ids,scope):
    assert scope in ['coarse','disease']
    pairs,inverse=torch.unique(torch.stack([patient_ids,disease],1),dim=0,return_inverse=True)
    counts=torch.bincount(inverse,minlength=len(pairs)).to(hidden.dtype)
    h=torch.zeros(len(pairs),hidden.shape[1],device=hidden.device,dtype=hidden.dtype).index_add(0,inverse,hidden)/counts[:,None]
    c=torch.zeros(len(pairs),coordinates.shape[1],device=hidden.device,dtype=hidden.dtype).index_add(0,inverse,coordinates)/counts[:,None]
    # Every patient-disease pair has a single coarse label.
    coarse=torch.zeros(len(pairs),device=hidden.device,dtype=hidden.dtype).index_add(0,inverse,y.to(hidden.dtype))/counts
    strata=coarse.long() if scope=='coarse' else pairs[:,1]
    terms=[];sizes=[]
    for label in torch.unique(strata):
        selected=strata==label;n=int(selected.sum())
        if n<4:continue
        a=h[selected]-h[selected].mean(0);b=c[selected]-c[selected].mean(0)
        cross=a.T@b/n
        denominator=(a.square().mean(0)[:,None]*b.square().mean(0)[None,:]+1e-4).sqrt()
        terms.append((cross/denominator).square().mean());sizes.append(n)
    penalty=torch.stack(terms).mean() if terms else hidden.sum()*0
    return penalty,sizes


class NormalDecorrelationModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None,witness_path=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_strength']==1. and cfg['normal_rank']==32
        assert cfg['normal_decorrelation_scope'] in ['coarse','disease'] and cfg['normal_decorrelation_strength']>=0
        for key in KEYS:
            self.register_buffer('decorrelation_deviation_'+key,torch.ones(32))
            if geometry is not None:
                getattr(self,'decorrelation_deviation_'+key).copy_(torch.as_tensor(geometry[key+'_eigenvalues']).sqrt().clamp(min=.05))
        self.encoder[1].register_forward_hook(self._capture)
        self.witness_path=None if witness_path is None else Path(witness_path);self.witness_written=False

    def _capture(self,module,args,output):self._hidden=output

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
            parts.append((coordinate/getattr(self,'decorrelation_deviation_'+key)).clamp(-5,5))
        coordinates=torch.cat(parts,-1).detach()
        out=super().forward(x);out.update(decorrelation_context=coordinates,decorrelation_hidden=self._hidden)
        if context is not None:out['_decorrelation_witness_context']=context
        return out


def decorrelation_loss(out,y,disease,weights,cfg):
    base=loss_fn(out,y,disease,weights,cfg)
    penalty,sizes=conditional_correlation(out['decorrelation_hidden'],out['decorrelation_context'],y,disease,weights['patient_ids'],cfg['normal_decorrelation_scope'])
    total=base+cfg['normal_decorrelation_strength']*penalty
    out['_decorrelation_details']={'base':base,'penalty':penalty,'sizes':sizes}
    context=out.get('_decorrelation_witness_context')
    if context is not None:
        model=context['model'];proof={'seed':np.array(cfg['seed']),'batch_size':np.array(len(y)),
            'scope':np.array(cfg['normal_decorrelation_scope']),'strength':np.array(cfg['normal_decorrelation_strength']),
            'base_loss':np.array(float(base.detach())),'penalty':np.array(float(penalty.detach())),
            'loss':np.array(float(total.detach())),'stratum_sizes':np.array(sizes),'rng_cpu':context['rng_cpu'].numpy()}
        tensors={'raw_global_all':context['raw_global_all'],'y':y,'disease':disease,'probability':out['p'],
            'decorrelation_context':out['decorrelation_context'],'decorrelation_hidden':out['decorrelation_hidden']}
        tensors.update({'weight_'+k:v for k,v in weights.items()})
        for k,v in tensors.items():proof[k]=v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:proof['rng_cuda']=context['rng_cuda'].cpu().numpy()
        for k,v in context['initial_parameter_sha256'].items():proof['initial_parameter_sha256_'+k]=np.array(v)
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return total

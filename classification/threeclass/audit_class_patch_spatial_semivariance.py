"""Independent CPU lexicographic graph and direct edge-energy/probability formulas."""
import numpy as np
import torch
from audit_class_query_patch_evidence import independent_probability as mean_probability
from audit_class_patch_dispersion import independent_probability as variance_probability

def independent_neighbors(positions,grids=(20,28)):
    ids=positions.detach().cpu().numpy().astype('int64');answer=[]
    for view,grid in enumerate(grids):
        pos=ids[:,view];xy=np.stack([pos//grid,pos%grid],-1)
        delta=xy[...,None,:]-xy[...,None,:,:]
        distance=np.square(delta).sum(-1)
        distance[...,np.arange(pos.shape[-1]),np.arange(pos.shape[-1])]=np.iinfo('int64').max
        physical=np.broadcast_to(pos[...,None,:],distance.shape)
        answer.append(np.lexsort((physical,distance),axis=-1)[...,:4].copy())
    return torch.from_numpy(np.stack(answer,1)).to(positions.device)

def independent_probability(source,tail_state,parameters,raw,bank,config):
    if config['statistic']=='log_variance':
        out=variance_probability(source,tail_state,parameters,raw,bank,config)
    else:out=mean_probability(source,tail_state,parameters,raw,bank,{'pooling':config['pooling'],'learned_query':False})
    zero=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
    out.update(patch_semivariance=zero,patch_neighbors=torch.zeros((*bank['patch_values'].shape[:-1],4),device=out['p'].device,dtype=torch.int64))
    if config['statistic']!='log_variance':out['patch_variance']=zero
    if config['statistic']!='log_semivariance':return out
    v=bank['patch_values'].float();v=v/torch.sqrt(v.square().sum(-1,keepdim=True)/64+1e-6)
    neighbors=independent_neighbors(bank['patch_indices']);shape=v.shape
    x=v.double().reshape(-1,shape[-2],shape[-1]);ix=neighbors.reshape(-1,shape[-2],4)
    offsets=torch.arange(len(x),device=x.device)[:,None,None]*shape[-2]
    adjacent=x.reshape(-1,shape[-1])[(ix+offsets).reshape(-1)].reshape(len(x),shape[-2],4,shape[-1])
    node_energy=((x[:,:,None,:]-adjacent).square().sum(-2)/8).reshape_as(v)
    pooled=[]
    for cls in range(2):
        weights=out['patch_attention'][:,:,:,cls].double();weights=weights/weights.sum(-1,keepdim=True)
        pooled.append((weights[...,None]*node_energy).sum(-2))
    statistic=torch.stack(pooled,1);evidence=torch.log1p(statistic).float()
    scores=(evidence.flatten(2)*parameters['patch_readout'].flatten(1)[None]).sum(-1)+parameters['patch_bias']
    shifts=torch.stack([2*scores[:,0].tanh(),2*scores[:,1].tanh(),torch.zeros_like(scores[:,0])],1)
    original=out['patch_source_probability'];ratio=shifts.double().exp()
    probability=(original.double()*ratio/(1+(original.double()*(ratio-1)).sum(1,keepdim=True))).float()
    mapping=source['state_dict']['subtype_to_class'].argmax(1).to(shifts.device)
    out.update(patch_evidence=evidence,patch_semivariance=statistic,patch_neighbors=neighbors,patch_scores=scores,patch_class_shift=shifts,
        p=probability,subtype=out['patch_source_subtype']+shifts.index_select(1,mapping),flat=probability.clamp(min=1e-7).log())
    return out

"""Independent parity-matrix channel rotation and centered covariance diagonal formula."""
import numpy as np
import torch
from audit_class_query_patch_evidence import independent_probability as mean_probability
from audit_class_patch_dispersion import independent_probability as variance_probability


def parity_matrix(device='cpu'):
    matrix=np.asarray([[1 if (i&j).bit_count()%2==0 else -1 for j in range(64)] for i in range(64)],dtype='float64')/8
    return torch.from_numpy(matrix).to(device)


def independent_probability(source,tail_state,parameters,raw,bank,config):
    if config['statistic']=='log_variance':return variance_probability(source,tail_state,parameters,raw,bank,config)
    out=mean_probability(source,tail_state,parameters,raw,bank,{'pooling':config['pooling'],'learned_query':False})
    out['patch_variance']=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
    if config['statistic']=='mean':return out
    value=bank['patch_values'].float();normalized=value/torch.sqrt(value.square().sum(-1,keepdim=True)/64+1e-6)
    transformed=normalized.double()@parity_matrix(value.device)
    delta=transformed-transformed[...,:1,:];variances=[]
    for cls in range(2):
        weight=out['patch_attention'][:,:,:,cls].double();total=weight.sum(-1,keepdim=True)
        center=(weight[...,None]*delta).sum(-2)/total
        variances.append((weight[...,None]*(delta-center[...,None,:]).square()).sum(-2)/total)
    variance=torch.stack(variances,1);evidence=torch.log1p(variance).float()
    score=(evidence.flatten(2)*parameters['patch_readout'].flatten(1)[None]).sum(-1)+parameters['patch_bias']
    shift=torch.stack([2*score[:,0].tanh(),2*score[:,1].tanh(),torch.zeros_like(score[:,0])],1)
    original=out['patch_source_probability'];ratio=shift.double().exp()
    probability=(original.double()*ratio/(1+(original.double()*(ratio-1)).sum(1,keepdim=True))).float()
    mapping=source['state_dict']['subtype_to_class'].argmax(1).to(shift.device)
    out.update(patch_variance=variance,patch_evidence=evidence,patch_scores=score,patch_class_shift=shift,p=probability,
        subtype=out['patch_source_subtype']+shift.index_select(1,mapping),flat=probability.clamp(min=1e-7).log())
    return out

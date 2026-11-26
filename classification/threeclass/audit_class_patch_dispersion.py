"""Independent centered-deviation variance and probability-ratio reconstruction."""
import torch
from audit_class_query_patch_evidence import independent_probability as mean_probability


def independent_probability(source,tail_state,parameters,raw,bank,config):
    mean_config={'pooling':config['pooling'],'learned_query':False}
    out=mean_probability(source,tail_state,parameters,raw,bank,mean_config)
    if config['statistic']=='mean':
        out['patch_variance']=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
        return out
    raw_values=bank['patch_values'].float()
    values=raw_values/torch.sqrt(raw_values.pow(2).sum(-1,keepdim=True)/64+1e-6)
    pools=[];variances=[]
    for cls in range(2):
        weights=out['patch_attention'][:,:,:,cls].double()
        weights=weights/weights.sum(-1,keepdim=True)
        x=values.double()
        mean=(weights[...,None]*x).sum(-2,keepdim=True)
        variance=(weights[...,None]*(x-mean).square()).sum(-2)
        variances.append(variance);pools.append(variance.log1p().float())
    evidence=torch.stack(pools,1);variance=torch.stack(variances,1)
    scores=(evidence.flatten(2)*parameters['patch_readout'].flatten(1)[None]).sum(-1)+parameters['patch_bias']
    shifts=torch.stack([2*scores[:,0].tanh(),2*scores[:,1].tanh(),torch.zeros_like(scores[:,0])],1)
    original=out['patch_source_probability'];ratio=shifts.double().exp()
    probability=(original.double()*ratio/(1+(original.double()*(ratio-1)).sum(1,keepdim=True))).float()
    mapping=source['state_dict']['subtype_to_class'].argmax(1).to(shifts.device)
    fine_shift=shifts.index_select(1,mapping)
    out.update(patch_evidence=evidence,patch_variance=variance,patch_scores=scores,patch_class_shift=shifts,
        p=probability,subtype=out['patch_source_subtype']+fine_shift,flat=probability.clamp(min=1e-7).log())
    return out

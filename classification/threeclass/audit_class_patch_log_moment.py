"""Independent max-centered exponential/normalized-weight log moment and task outputs."""
import torch
from audit_class_query_patch_evidence import independent_probability as mean_probability
from audit_class_patch_dispersion import independent_probability as variance_probability


def independent_probability(source,tail_state,parameters,raw,bank,config):
    if config['statistic']=='log_variance':
        out=variance_probability(source,tail_state,parameters,raw,bank,config)
    else:
        out=mean_probability(source,tail_state,parameters,raw,bank,{'pooling':config['pooling'],'learned_query':False})
        out['patch_variance']=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
    shape=out['patch_evidence'].shape[:4]
    out['patch_directional_scores']=torch.zeros((*shape,64),device=out['p'].device,dtype=torch.float64)
    out['patch_log_moments']=torch.zeros(shape,device=out['p'].device,dtype=torch.float64)
    if config['statistic']!='log_moment':return out
    value=bank['patch_values'].float();value=(value/torch.sqrt(value.square().sum(-1,keepdim=True)/64+1e-6)).double()
    projection=torch.stack([torch.matmul(value,parameters['patch_readout'][c].double()[...,None]).squeeze(-1) for c in range(2)],1)
    attention=out['patch_attention'].permute(0,3,1,2,4).double()
    maximum=projection.max(-1,keepdim=True).values
    moment=maximum.squeeze(-1)+((attention*(projection-maximum).exp()).sum(-1)/attention.sum(-1)).log()
    score=moment.flatten(2).sum(-1).float()+parameters['patch_bias']
    shift=torch.stack([2*score[:,0].tanh(),2*score[:,1].tanh(),torch.zeros_like(score[:,0])],1)
    original=out['patch_source_probability'];ratio=shift.double().exp()
    probability=(original.double()*ratio/(1+(original.double()*(ratio-1)).sum(1,keepdim=True))).float()
    mapping=source['state_dict']['subtype_to_class'].argmax(1).to(shift.device)
    out.update(patch_directional_scores=projection,patch_log_moments=moment,patch_scores=score,patch_class_shift=shift,p=probability,
        subtype=out['patch_source_subtype']+shift.index_select(1,mapping),flat=probability.clamp(min=1e-7).log())
    return out

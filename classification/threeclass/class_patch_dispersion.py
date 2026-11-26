"""Class evidence from per-head weighted diagonal patch dispersion."""
import torch
from class_query_patch_evidence import ClassQueryPatchEvidenceModel, anchored_probability, rms, TRACE as PARENT_TRACE
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha

TRACE=PARENT_TRACE+['patch_variance']


def weighted_log_variance(attention, values):
    # Shifting by an arbitrary anchor is an algebraic stabilization only.
    # Constant values produce exactly zero even with rounded softmax weights.
    delta=(values.double()-values[...,:1,:].double())
    weights=attention.double()
    total=weights.sum(-1).permute(0,3,1,2)[...,None]
    mean=torch.einsum('bshck,bshkd->bcshd',weights,delta)/total
    second=torch.einsum('bshck,bshkd->bcshd',weights,delta.square())/total
    variance=(second-mean.square()).clamp(min=0)
    return variance.log1p().to(values.dtype),variance


class ClassPatchDispersionModel(ClassQueryPatchEvidenceModel):
    def __init__(self,source,tail,config,seed):
        assert config['pooling'] in ['uniform','source']
        assert config['statistic'] in ['mean','log_variance']
        super().__init__(source,tail,{'pooling':config['pooling'],'learned_query':False},seed)
        self.config=dict(config)
        self.config['learned_query']=False

    def forward(self,base,bank,raw_registers):
        out=super().forward(base,bank,raw_registers)
        if self.config['statistic']=='mean':
            out['patch_variance']=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
            return out
        values=rms(bank['patch_values'].float())
        evidence,variance=weighted_log_variance(out['patch_attention'],values)
        scores=(evidence*self.patch_readout[None]).sum((2,3,4))+self.patch_bias
        shifts=torch.cat([2*scores.tanh(),torch.zeros_like(scores[:,:1])],1)
        probability=anchored_probability(out['patch_source_probability'],shifts)
        out.update(patch_evidence=evidence,patch_variance=variance,patch_scores=scores,patch_class_shift=shifts,
            p=probability,subtype=out['patch_source_subtype']+shifts@self.head.subtype_to_class.T,
            flat=probability.clamp(min=1e-7).log())
        return out


@cuda_ready_restore
def restore(checkpoint,tail):
    path=checkpoint['source_checkpoint'];assert sha(path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=ClassPatchDispersionModel(source,tail,checkpoint['adapter_config'],20293500+checkpoint['seed'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict'])==set(parameters)=={'patch_readout','patch_bias'}
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(),source

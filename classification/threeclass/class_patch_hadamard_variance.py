"""Fixed orthogonal channel-basis patch variance with equal-capacity pathology readouts."""
import torch
from class_patch_dispersion import ClassPatchDispersionModel,weighted_log_variance,TRACE
from class_query_patch_evidence import rms,anchored_probability
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha


def hadamard64(values):
    assert values.shape[-1]==64
    answer=values.double()
    for stride in [1,2,4,8,16,32]:
        parts=answer.reshape(*answer.shape[:-1],64//(2*stride),2,stride)
        left,right=parts[...,0,:],parts[...,1,:]
        answer=torch.stack((left+right,left-right),-2).reshape_as(answer)
    return answer*.125


def weighted_log_hadamard_variance(attention,values):
    evidence,variance=weighted_log_variance(attention,hadamard64(values))
    return evidence.to(values.dtype),variance


class ClassPatchHadamardVarianceModel(ClassPatchDispersionModel):
    def __init__(self,source,tail,config,seed):
        assert config['pooling'] in ['uniform','source'] and config['statistic'] in ['mean','log_variance','log_hadamard_variance']
        parent={**config,'statistic':'mean' if config['statistic']=='mean' else 'log_variance'}
        super().__init__(source,tail,parent,seed)
        self.config=dict(config);self.config['learned_query']=False

    def forward(self,base,bank,raw_registers):
        out=super().forward(base,bank,raw_registers)
        if self.config['statistic']!='log_hadamard_variance':return out
        evidence,variance=weighted_log_hadamard_variance(out['patch_attention'],rms(bank['patch_values'].float()))
        scores=(evidence*self.patch_readout[None]).sum((2,3,4))+self.patch_bias
        shifts=torch.cat([2*scores.tanh(),torch.zeros_like(scores[:,:1])],1)
        probability=anchored_probability(out['patch_source_probability'],shifts)
        out.update(patch_evidence=evidence,patch_variance=variance,patch_scores=scores,patch_class_shift=shifts,p=probability,
            subtype=out['patch_source_subtype']+shifts@self.head.subtype_to_class.T,flat=probability.clamp(min=1e-7).log())
        return out


@cuda_ready_restore
def restore(checkpoint,tail):
    path=checkpoint['source_checkpoint'];assert sha(path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=ClassPatchHadamardVarianceModel(source,tail,checkpoint['adapter_config'],20293500+checkpoint['seed'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict'])==set(parameters)=={'patch_readout','patch_bias'}
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(),source

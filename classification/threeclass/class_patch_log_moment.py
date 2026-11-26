"""Learn patch evidence directions before fixed-temperature log-mean-exp pooling."""
import torch
from class_patch_dispersion import ClassPatchDispersionModel,TRACE as PARENT_TRACE
from class_query_patch_evidence import rms,anchored_probability,ClassQueryPatchEvidenceModel
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha
TRACE=list(PARENT_TRACE)+['patch_directional_scores','patch_log_moments']


def weighted_log_moment(attention,values,readout):
    projection=torch.einsum('bshkd,cshd->bcshk',values.double(),readout.double())
    log_weight=attention.permute(0,3,1,2,4).double().log()
    moments=torch.logsumexp(log_weight+projection,-1)-torch.logsumexp(log_weight,-1)
    return moments,projection


class ClassPatchLogMomentModel(ClassPatchDispersionModel):
    def __init__(self,source,tail,config,seed):
        assert config['pooling'] in ['uniform','source'] and config['statistic'] in ['mean','log_variance','log_moment']
        parent={**config,'statistic':'log_variance' if config['statistic']=='log_variance' else 'mean'}
        super().__init__(source,tail,parent,seed)
        self.config=dict(config);self.config['learned_query']=False

    def forward(self,base,bank,raw_registers):
        if self.config['statistic']=='log_moment':
            out=ClassQueryPatchEvidenceModel.forward(self,base,bank,raw_registers)
            out['patch_variance']=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
        else:out=super().forward(base,bank,raw_registers)
        shape=out['patch_evidence'].shape[:4]
        out['patch_directional_scores']=torch.zeros((*shape,64),device=out['p'].device,dtype=torch.float64)
        out['patch_log_moments']=torch.zeros(shape,device=out['p'].device,dtype=torch.float64)
        if self.config['statistic']!='log_moment':return out
        moments,projection=weighted_log_moment(out['patch_attention'],rms(bank['patch_values'].float()),self.patch_readout)
        scores=moments.sum((2,3)).float()+self.patch_bias
        shifts=torch.cat([2*scores.tanh(),torch.zeros_like(scores[:,:1])],1)
        probability=anchored_probability(out['patch_source_probability'],shifts)
        out.update(patch_directional_scores=projection,patch_log_moments=moments,patch_scores=scores,patch_class_shift=shifts,p=probability,
            subtype=out['patch_source_subtype']+shifts@self.head.subtype_to_class.T,flat=probability.clamp(min=1e-7).log())
        return out


@cuda_ready_restore
def restore(checkpoint,tail):
    path=checkpoint['source_checkpoint'];assert sha(path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=ClassPatchLogMomentModel(source,tail,checkpoint['adapter_config'],20293500+checkpoint['seed'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict'])==set(parameters)=={'patch_readout','patch_bias'}
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(),source

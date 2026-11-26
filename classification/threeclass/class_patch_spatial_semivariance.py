"""Equal-capacity class evidence from fixed sparse spatial-neighbour differences."""
import torch
from class_query_patch_evidence import ClassQueryPatchEvidenceModel,anchored_probability,rms,TRACE as PARENT_TRACE
from class_patch_dispersion import weighted_log_variance
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha
TRACE=PARENT_TRACE+['patch_variance','patch_semivariance','patch_neighbors']
GRIDS=(20,28)

def nearest_four(positions,grids=GRIDS):
    assert positions.ndim==4 and positions.shape[1]==len(grids) and positions.shape[-1]>=5
    assert not positions.is_floating_point(),'Patch positions must be integer metadata.'
    ids=positions.long();neighbors=[]
    assert (ids.sort(-1).values.diff(dim=-1)!=0).all(),'Positions must be distinct within each head.'
    for view,grid in enumerate(grids):
        p=ids[:,view];assert ((p>=0)&(p<grid*grid)).all()
        row,col=p//grid,p%grid
        distance=(row[...,None]-row[...,None,:]).square()+(col[...,None]-col[...,None,:]).square()
        score=distance*(grid*grid)+p[...,None,:]
        eye=torch.eye(p.shape[-1],device=p.device,dtype=torch.bool)
        score=score.masked_fill(eye,torch.iinfo(torch.int64).max)
        neighbors.append(score.topk(4,dim=-1,largest=False,sorted=True).indices)
    return torch.stack(neighbors,dim=1)

def weighted_log_semivariance(attention,values,positions,grids=GRIDS):
    neighbors=nearest_four(positions,grids)
    x=values.double();energy=torch.zeros_like(x)
    for edge in range(4):
        index=neighbors[...,edge,None].expand_as(values)
        other=x.gather(-2,index)
        energy=energy+(x-other).square()*.125
    weights=attention.double();weights=weights/weights.sum(-1,keepdim=True)
    statistic=torch.einsum('bshck,bshkd->bcshd',weights,energy)
    return statistic.log1p().to(values.dtype),statistic,neighbors

class ClassPatchSpatialSemivarianceModel(ClassQueryPatchEvidenceModel):
    def __init__(self,source,tail,config,seed):
        assert config['pooling'] in ['uniform','source'] and config['statistic'] in ['mean','log_variance','log_semivariance']
        super().__init__(source,tail,{'pooling':config['pooling'],'learned_query':False},seed)
        self.config=dict(config);self.config['learned_query']=False
    def forward(self,base,bank,raw_registers):
        out=super().forward(base,bank,raw_registers)
        zero=torch.zeros_like(out['patch_evidence'],dtype=torch.float64)
        out.update(patch_variance=zero,patch_semivariance=zero,patch_neighbors=torch.zeros((*bank['patch_values'].shape[:-1],4),device=raw_registers.device,dtype=torch.int64))
        if self.config['statistic']=='mean':return out
        values=rms(bank['patch_values'].float())
        if self.config['statistic']=='log_variance':
            evidence,statistic=weighted_log_variance(out['patch_attention'],values);out['patch_variance']=statistic
        else:
            assert bank['patch_indices'].shape[-1]==64 and bank['patch_indices'].shape[:3]==bank['patch_values'].shape[:3]
            evidence,statistic,neighbors=weighted_log_semivariance(out['patch_attention'],values,bank['patch_indices'])
            out.update(patch_semivariance=statistic,patch_neighbors=neighbors)
        scores=(evidence*self.patch_readout[None]).sum((2,3,4))+self.patch_bias
        shifts=torch.cat([2*scores.tanh(),torch.zeros_like(scores[:,:1])],1)
        probability=anchored_probability(out['patch_source_probability'],shifts)
        out.update(patch_evidence=evidence,patch_scores=scores,patch_class_shift=shifts,p=probability,
            subtype=out['patch_source_subtype']+shifts@self.head.subtype_to_class.T,flat=probability.clamp(min=1e-7).log())
        return out

@cuda_ready_restore
def restore(checkpoint,tail):
    path=checkpoint['source_checkpoint'];assert sha(path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=ClassPatchSpatialSemivarianceModel(source,tail,checkpoint['adapter_config'],20293500+checkpoint['seed'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict'])==set(parameters)=={'patch_readout','patch_bias'}
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(),source

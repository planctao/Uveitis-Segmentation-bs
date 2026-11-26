"""Direct pathology log-odds evidence from a frozen DINO spatial K/V bank."""
import torch
from torch import nn
from register_patch_query import RegisterPatchQueryModel
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha

TRACE=['patch_source_probability','patch_source_subtype','patch_source_query','patch_source_attention',
    'patch_attention','patch_evidence','patch_scores','patch_class_shift']


def rms(value):
    return value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)


def anchored_probability(probability, shift):
    change=shift.double().expm1()
    mean=(probability.double()*change).sum(1,keepdim=True)
    return (probability.double()+probability.double()*(change-mean)/(1+mean)).to(probability.dtype)


class ClassQueryPatchEvidenceModel(RegisterPatchQueryModel):
    def __init__(self,source,tail,config,seed):
        assert config['pooling'] in ['uniform','source','learned','contrast']
        assert config['learned_query']==(config['pooling'] in ['learned','contrast'])
        with torch.random.fork_rng(devices=[]):
            super().__init__(source,tail,{'kind':'shared','rank':4,'alpha':8},seed)
        del self.down,self.up
        self.config=dict(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.patch_readout=nn.Parameter(torch.zeros(2,2,16,64))
            self.patch_bias=nn.Parameter(torch.zeros(2))
            if config['learned_query']:
                self.patch_queries=nn.Parameter(torch.randn(2,16,64)*.02)
        self.eval()

    def parameter_groups(self):
        return [{'params':[p for p in self.parameters() if p.requires_grad],'lr':.001}]

    def forward(self,base,bank,raw_registers):
        side=((raw_registers-self.source_mean_side_tokens)/self.source_std_side_tokens).clamp(-10,10)
        out=self.head({**base,'side_tokens':side})
        query=rms(rms(bank['query_raw'].float()).mean(-2))
        keys,values=rms(bank['patch_keys'].float()),rms(bank['patch_values'].float())
        mode=self.config['pooling']
        class_queries=query[:,:,:,None].expand(-1,-1,-1,2,-1)
        if self.config['learned_query']:
            class_queries=class_queries+self.patch_queries.permute(1,0,2)[None,None]
        all_queries=torch.cat([query[:,:,:,None],class_queries],3)
        all_attention=torch.einsum('bshcd,bshkd->bshck',all_queries,keys).div(8).softmax(-1)
        source_attention=all_attention[:,:,:,0]
        if mode=='uniform':
            attention=torch.full_like(source_attention,.015625)[:,:,:,None].expand(-1,-1,-1,2,-1)
        elif mode=='source':
            attention=source_attention[:,:,:,None].expand(-1,-1,-1,2,-1)
        else:
            attention=all_attention[:,:,:,1:]
        pooling=attention-source_attention[:,:,:,None] if mode=='contrast' else attention
        evidence=torch.einsum('bshck,bshkd->bcshd',pooling,values)
        scores=(evidence*self.patch_readout[None]).sum((2,3,4))+self.patch_bias
        shifts=torch.cat([2*scores.tanh(),torch.zeros_like(scores[:,:1])],1)
        probability=anchored_probability(out['p'],shifts)
        subtype=out['subtype']+shifts@self.head.subtype_to_class.T
        out.update(patch_source_probability=out['p'],patch_source_subtype=out['subtype'],
            patch_source_query=query,patch_source_attention=source_attention,patch_attention=attention,
            patch_evidence=evidence,patch_scores=scores,patch_class_shift=shifts,
            p=probability,subtype=subtype,flat=probability.clamp(min=1e-7).log())
        return out


@cuda_ready_restore
def restore(checkpoint,tail):
    path=checkpoint['source_checkpoint'];assert sha(path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=ClassQueryPatchEvidenceModel(source,tail,checkpoint['adapter_config'],20293500+checkpoint['seed'])
    names={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict'])==names
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(),source

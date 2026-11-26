"""Query-aligned or shared top16 residual pooling with full source attention mass."""
import torch
from torch.nn import functional as F
from register_patch_gated_value import RegisterPatchGatedValueModel


class RegisterPatchQuerySelectionModel(RegisterPatchGatedValueModel):
    def __init__(self,source,tail,config,seed):
        assert config['selection'] in ['query_top16','shared_top16']
        super().__init__(source,tail,config,seed)

    def query_patch_update(self,values):
        values=values.double()
        amplitude=(values.square().mean(-1,keepdim=True)+1e-6).sqrt()
        normalized=values/amplitude
        down=self.value_down.double().expand(16,-1,-1)
        gate=self.value_gate.double().expand(16,-1,-1)
        up=self.value_up.double().expand(16,-1,-1)
        content=torch.einsum('bshqkd,hrd->bshqkr',normalized,down)
        control=torch.einsum('bshqkd,hrd->bshqkr',normalized,gate)
        code=F.silu(content)*control
        return torch.einsum('bshqkr,hdr->bshqkd',code,up)*amplitude*(self.config['alpha']/self.config['rank'])

    def correct_registers(self,bank,raw_registers):
        query=bank['query_raw'].double()
        prefix='query16_' if self.config['selection']=='query_top16' else 'shared16_'
        keys=bank[prefix+'keys'].double();values=bank[prefix+'values'].double()
        if self.config['selection']=='shared_top16':
            keys=keys[:,:, :,None].expand(-1,-1,-1,4,-1,-1)
            values=values[:,:,:,None].expand(-1,-1,-1,4,-1,-1)
        assert keys.shape==values.shape==(len(query),2,16,4,16,64)
        logits=(query[...,None,:]*keys).sum(-1)/8
        exp=(logits-logits.amax(-1,keepdim=True)).exp()
        attention=exp/exp.sum(-1,keepdim=True)
        pooled=(attention[...,None]*values).sum(-2)
        conditional=(attention[...,None]*self.query_patch_update(values)).sum(-2)
        mass=bank[prefix+'mass'].double();assert mass.shape==query.shape[:-1]
        difference=conditional*mass[...,None]
        flat=difference.permute(0,1,3,2,4).reshape(len(query),2,4,1024).float()
        branch=F.linear(flat,self.tail.proj.weight,None)*self.tail.gamma1
        before=bank['postattention'].float()
        correction=self.tail.finish(before+branch)-self.tail.finish(before)
        registers=(raw_registers+torch.cat([correction[:,0],correction[:,1]],-1)).half().float()
        return registers,{'selected_attention':attention,'baseline_pool':pooled,'conditional_value_pool_difference':conditional,
            'applied_attention_mass':mass,'value_pool_difference':difference,'pool_difference':flat,
            'attention_difference':branch,'tail_difference':correction}

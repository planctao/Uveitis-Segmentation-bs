"""Frozen DINO tail and M8 head with trainable, anchored register patch queries."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from large_token_evidence import LargeTokenEvidenceModel
from experiment import KEYS


class FrozenDinoTail(nn.Module):
    def __init__(self,model):
        super().__init__();last=model.blocks[-1]
        assert isinstance(last.attn.q_norm,nn.Identity) and isinstance(last.attn.k_norm,nn.Identity) and isinstance(last.attn.norm,nn.Identity)
        self.proj=copy.deepcopy(last.attn.proj);self.norm2=copy.deepcopy(last.norm2);self.mlp=copy.deepcopy(last.mlp);self.norm=copy.deepcopy(model.norm)
        self.register_buffer('gamma1',last.gamma_1.detach().clone());self.register_buffer('gamma2',last.gamma_2.detach().clone())
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

    def finish(self,hidden):return self.norm(hidden+self.gamma2*self.mlp(self.norm2(hidden)))


class RegisterPatchQueryModel(nn.Module):
    def __init__(self,source,tail,config,seed):
        super().__init__();self.config=dict(config);self.tail=copy.deepcopy(tail)
        self.head=LargeTokenEvidenceModel(source['cfg'],source['dims'],len(source['diseases']))
        self.head.load_state_dict(source['state_dict'],strict=True)
        for p in self.head.parameters():p.requires_grad_(False)
        for p in self.tail.parameters():p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down=nn.Linear(1024,config['rank'],bias=False);self.up=nn.Linear(config['rank'],1024,bias=False)
            nn.init.zeros_(self.up.weight)
        for name in ['side_tokens']:
            self.register_buffer('source_mean_'+name,torch.as_tensor(source['stats'][name]['mean']).clone())
            self.register_buffer('source_std_'+name,torch.as_tensor(source['stats'][name]['std']).clone())
        self.eval()

    def train(self,mode=True):
        super().train(mode);self.head.eval();self.tail.eval();return self

    def correct_registers(self,bank,raw_registers):
        batch=len(raw_registers);input_value=bank['query_input'].float()
        if self.config['kind']=='shared':input_value=input_value.mean(2,keepdim=True).expand_as(input_value)
        update=self.up(self.down(input_value))*(self.config['alpha']/self.config['rank'])
        delta_query=update.reshape(batch,2,4,16,64).permute(0,1,3,2,4)
        query=bank['query_raw'].double();keys=bank['patch_keys'].double();values=bank['patch_values'].double()
        delta_query=delta_query.double()
        base_attention=((query@keys.transpose(-2,-1))/8).softmax(-1)
        attention=(((query+delta_query)@keys.transpose(-2,-1))/8).softmax(-1)
        pool_difference=(attention@values)-(base_attention@values)
        pool_difference=pool_difference.permute(0,1,3,2,4).reshape(batch,2,4,1024).float()
        attention_difference=F.linear(pool_difference,self.tail.proj.weight,None)*self.tail.gamma1
        before=bank['postattention'].float();baseline=self.tail.finish(before)
        after=self.tail.finish(before+attention_difference);correction=after-baseline
        corrected=raw_registers+torch.cat([correction[:,0],correction[:,1]],-1)
        corrected=corrected.to(torch.float16).float()
        return corrected,{'delta_query':delta_query,'selected_attention':attention,'baseline_attention':base_attention,
            'pool_difference':pool_difference,'attention_difference':attention_difference,'tail_difference':correction}

    def forward(self,base,bank,raw_registers):
        registers,details=self.correct_registers(bank,raw_registers)
        side=((registers-self.source_mean_side_tokens)/self.source_std_side_tokens).clamp(-10,10)
        out=self.head({**base,'side_tokens':side});out.update(details);return out

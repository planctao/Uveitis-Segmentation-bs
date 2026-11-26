"""Final-block attention LoRA or equal-capacity token residual with exact source-feature anchoring."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call
from experiment import KEYS
from extract_dinov3_large import make_model
from large_token_evidence import LargeTokenEvidenceModel
from large_lastblock_delta_lora_cache_v2 import summarize

def make_templates(parent_protocol):
    model=make_model(parent_protocol)
    block=copy.deepcopy(model.blocks[-1]).float().eval()
    norm=copy.deepcopy(model.norm).float().eval()
    for module in [block,norm]:
        for parameter in module.parameters():parameter.requires_grad_(False)
    assert isinstance(block.attn.q_norm,nn.Identity) and isinstance(block.attn.k_norm,nn.Identity)
    assert isinstance(block.attn.norm,nn.Identity)
    assert block.attn.qkv_bias_separate is False and block.attn.qkv.weight.shape==(3072,1024)
    assert block.attn.num_heads==16 and block.attn.num_prefix_tokens==5
    del model;torch.cuda.empty_cache()
    return block,norm

class LastBlockDeltaLoRA(nn.Module):
    def __init__(self,source,templates,config,seed):
        super().__init__()
        assert config in [{'kind':'attention','rank':4,'alpha':8},{'kind':'token','rank':8,'alpha':16}]
        self.config=dict(config);self.block=copy.deepcopy(templates[0]).float().eval();self.norm=copy.deepcopy(templates[1]).float().eval()
        self.head=LargeTokenEvidenceModel(source['cfg'],source['dims'],len(source['diseases']))
        self.head.load_state_dict(source['state_dict'],strict=True)
        for module in [self.block,self.norm,self.head]:
            for parameter in module.parameters():parameter.requires_grad_(False)
        for key in list(KEYS)+['side_tokens']:
            self.register_buffer('source_mean_'+key,torch.as_tensor(source['stats'][key]['mean']).clone())
            self.register_buffer('source_std_'+key,torch.as_tensor(source['stats'][key]['std']).clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.adapter_a=nn.Parameter(torch.empty(config['rank'],1024))
            self.adapter_b=nn.Parameter(torch.zeros(3072 if config['kind']=='attention' else 1024,config['rank']))
            nn.init.kaiming_uniform_(self.adapter_a,a=math.sqrt(5))
        assert self.adapter_a.numel()+self.adapter_b.numel()==16384
        self.eval()

    def train(self,mode=True):
        super().train(mode)
        self.block.eval();self.norm.eval();self.head.eval()
        return self

    def adapted_tokens(self,inputs,rope):
        if self.config['kind']=='attention':
            weight=self.block.attn.qkv.weight+2*(self.adapter_b@self.adapter_a)
            hidden=functional_call(self.block,{'attn.qkv.weight':weight},(inputs,),{'rope':rope},strict=False)
        else:
            with torch.no_grad():hidden=self.block(inputs,rope=rope)
            hidden=hidden+2*torch.nn.functional.linear(torch.nn.functional.linear(hidden,self.adapter_a),self.adapter_b)
        return self.norm(hidden)

    def delta_summary(self,inputs,rope,size):
        with torch.autocast('cuda',enabled=False):
            inputs=inputs.cuda().float()
            with torch.no_grad():original=summarize(self.norm(self.block(inputs,rope=rope)),size)
            adapted=summarize(self.adapted_tokens(inputs,rope),size)
            return tuple(adapted[key]-original[key] for key in ['global','regions','dispersion','side_tokens'])

    def anchored_features(self,cache,features,ids,microbatch=8,use_checkpoint=True,independent=False):
        assert microbatch==8
        ids=np.asarray(ids,dtype='int64')
        locations={int(index):position for position,index in enumerate(cache['indices'])}
        positions=np.asarray([locations[int(index)] for index in ids])
        summaries=[]
        for size in [320,448]:
            rows=[]
            rope=cache['rope'][size].cuda().float()
            for begin in range(0,len(ids),microbatch):
                cpu=torch.from_numpy(np.array(cache['tokens'][size][positions[begin:begin+microbatch]],copy=True))
                if independent:
                    from audit_large_lastblock_delta_lora import independent_delta_summary
                    def compute(value,local_rope=rope,local_size=size):
                        return independent_delta_summary(self,value,local_rope,local_size)
                else:
                    def compute(value,local_rope=rope,local_size=size):
                        return self.delta_summary(value,local_rope,local_size)
                if use_checkpoint and torch.is_grad_enabled():
                    values=checkpoint(compute,cpu,use_reentrant=False,preserve_rng_state=False)
                else:values=compute(cpu)
                rows.append(values)
            summaries.append({key:torch.cat([row[k] for row in rows],0) for k,key in enumerate(['global','regions','dispersion','side_tokens'])})
        raw={}
        for key in list(KEYS)+['side_tokens']:
            cached=torch.from_numpy(np.array(features[key][ids],copy=True)).cuda().float()
            raw[key]=cached if key=='mid' else cached+torch.cat([view[key] for view in summaries],-1)
        return raw

    def forward(self,cache,features,ids,microbatch=8,use_checkpoint=True,independent=False,return_raw=False):
        raw=self.anchored_features(cache,features,ids,microbatch,use_checkpoint,independent)
        normalized={key:((value-getattr(self,'source_mean_'+key))/getattr(self,'source_std_'+key)).clamp(-10,10) for key,value in raw.items()}
        out=self.head(normalized)
        if return_raw:out['_anchored_raw']=raw
        return out

def adapter_config(arm):return {key:arm[key] for key in ['kind','rank','alpha']}

def restore(checkpoint,source,templates,device='cuda'):
    model=LastBlockDeltaLoRA(source,templates,checkpoint['adapter_config'],20294400+checkpoint['seed']).to(device)
    parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
    assert set(parameters)==set(checkpoint['trainable_state_dict'])=={'adapter_a','adapter_b'}
    with torch.no_grad():
        for name,parameter in parameters.items():parameter.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval()

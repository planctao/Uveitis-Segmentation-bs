"""Zero-initialized nonlinear output adapters on final frozen DINO transformer blocks."""
import torch
from torch import nn
import torch.nn.functional as F
from anchored_lora import AnchoredMultiScaleLoRA
from lora_seed import seed_all


class TokenAdapter(nn.Module):
    def __init__(self,dimension=768,rank=16):
        super().__init__();assert rank in [16,64]
        self.norm=nn.LayerNorm(dimension)
        self.down=nn.Linear(dimension,rank)
        self.up=nn.Linear(rank,dimension)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self,tokens):
        return .1*self.up(F.gelu(self.down(self.norm(tokens))))


class MultiScaleNonlinearAdapter(AnchoredMultiScaleLoRA):
    def __init__(self,checkpoint,config):
        super().__init__(checkpoint,{'rank':0,'blocks':0,'alpha':0})
        self.nonlinear_config=config
        assert config['blocks'] in [0,4]
        # Independent of fold disease count and unused source-head construction draws.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20361124+checkpoint['cfg'].get('seed',0))
            self.backbone.token_adapters=nn.ModuleList([TokenAdapter(768,config['rank']) for _ in range(config['blocks'])])
        self._adapter_handles=[]
        if config['blocks']:
            for block,adapter in zip(self.backbone.blocks[-config['blocks']:],self.backbone.token_adapters):
                def add(module,args,output,adapter=adapter):return output+adapter(output)
                self._adapter_handles.append(block.register_forward_hook(add))
        if 'nonlinear_state_dict' in checkpoint:self.load_delta(checkpoint['nonlinear_state_dict'])

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval();self.backbone.token_adapters.train(mode);self.head.train(mode)
        return self


def construct_nonlinear(checkpoint,config,seed):
    seed_all(seed);model=MultiScaleNonlinearAdapter(checkpoint,config);seed_all(seed)
    return model

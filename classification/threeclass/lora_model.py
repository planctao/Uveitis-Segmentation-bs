"""Low-rank updates to final DINO attention blocks with the frozen feature recipe."""
import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from experiment import Model
from extract_vit import WEIGHTS


class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.scale=alpha/rank
        for p in self.base.parameters():p.requires_grad=False
        self.lora_a=nn.Linear(base.in_features,rank,bias=False)
        self.lora_b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight,a=math.sqrt(5));nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self):
        # EVA's separate Q/V bias path accesses qkv.weight through F.linear.
        # Expose the differentiable low-rank weight update to that path too.
        with torch.autocast(device_type=self.base.weight.device.type,enabled=False):
            return self.base.weight+self.scale*(self.lora_b.weight.float()@self.lora_a.weight.float())

    @property
    def bias(self):return self.base.bias

    def forward(self,x):return F.linear(x,self.weight,self.bias)


class MultiScaleLoRA(nn.Module):
    def __init__(self,checkpoint,adapter):
        super().__init__();self.adapter=adapter
        self.backbone=timm.create_model('vit_base_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=320,dynamic_img_size=True)
        state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(state,self.backbone),strict=True)
        for p in self.backbone.parameters():p.requires_grad=False
        if adapter['rank']:
            for block in self.backbone.blocks[-adapter['blocks']:]:
                block.attn.qkv=LoRALinear(block.attn.qkv,adapter['rank'],adapter['alpha'])
        self.head=Model(checkpoint['cfg'],checkpoint['dims'],len(checkpoint['diseases']))
        self.head.load_state_dict(checkpoint['state_dict'],strict=True)
        for name,stat in checkpoint['stats'].items():
            self.register_buffer('mean_'+name,torch.as_tensor(stat['mean']))
            self.register_buffer('std_'+name,torch.as_tensor(stat['std']))
        self.captured={};self.backbone.blocks[7].register_forward_hook(self.capture)
        if 'lora_state_dict' in checkpoint:self.load_delta(checkpoint['lora_state_dict'])

    def capture(self,module,inputs,output):self.captured['mid']=output

    def raw_features(self,x320,x448):
        views=[]
        for size,x in [(320,x320),(448,x448)]:
            tokens=self.backbone.forward_features(x);patch=tokens[:,self.backbone.num_prefix_tokens:]
            regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(x),768,size//16,size//16),(3,3)).flatten(2).transpose(1,2)
            values={'global':tokens[:,0],'regions':regions,'mid':self.captured.pop('mid')[:,self.backbone.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
            # Match the float16 cache representation while retaining cast gradients.
            views.append({k:v.to(torch.float16).float() for k,v in values.items()})
        return {k:torch.cat([v[k] for v in views],dim=-1) for k in views[0]}

    def forward(self,x320,x448):
        features=self.raw_features(x320,x448)
        normalized={k:((v-getattr(self,'mean_'+k))/getattr(self,'std_'+k)).clamp(-10,10) for k,v in features.items()}
        return self.head(normalized)

    def delta(self):return {k:p.detach().cpu().clone() for k,p in self.backbone.named_parameters() if p.requires_grad}

    def load_delta(self,state):
        expected={k for k,p in self.backbone.named_parameters() if p.requires_grad}
        assert set(state)==expected,'Restore all and only adapter parameters'
        restored=self.backbone.load_state_dict(state,strict=False)
        assert not restored.unexpected_keys and not expected.intersection(restored.missing_keys)

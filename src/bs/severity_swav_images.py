"""Image-only SwAV views and a trainable pretrained ConvNeXt adapter."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backbone/dinov3'))
from dinov3.hub.backbones import dinov3_convnext_tiny


class FAMulticrop(torch.utils.data.Dataset):
    def __init__(self,paths):
        self.paths=list(paths)

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        image=Image.open(self.paths[index]).convert('RGB');width,height=image.size;views=[]
        for j in range(4):
            scale=float(torch.empty(()).uniform_(.70,1.) if j<2 else torch.empty(()).uniform_(.30,.60))
            side=min(width,height,max(16,int(min(width,height)*scale**.5)))
            left=int(torch.randint(width-side+1,()).item());top=int(torch.randint(height-side+1,()).item())
            size=256 if j<2 else 128
            crop=image.crop((left,top,left+side,top+side)).resize((size,size),Image.Resampling.BILINEAR)
            a=torch.from_numpy(np.asarray(crop,dtype=np.float32)/255).permute(2,0,1)
            if float(torch.rand(()))<.5:a=a.flip(2)
            if float(torch.rand(()))<.5:a=a.flip(1)
            gamma=float(torch.empty(()).uniform_(.8,1.2));gain=float(torch.empty(()).uniform_(.9,1.1))
            a=(a.pow(gamma)*gain).clamp(0,1)
            if float(torch.rand(()))<.2:
                coords=torch.arange(-2,3,dtype=a.dtype);sigma=float(torch.empty(()).uniform_(.3,1.))
                kernel=torch.exp(-coords.square()/(2*sigma**2));kernel/=kernel.sum()
                kernel=(kernel[:,None]*kernel[None,:]).view(1,1,5,5).expand(3,1,5,5)
                a=F.conv2d(F.pad(a.unsqueeze(0),(2,2,2,2),mode='reflect'),kernel,groups=3).squeeze(0)
            mean=a.new_tensor([.485,.456,.406]).view(3,1,1);std=a.new_tensor([.229,.224,.225]).view(3,1,1)
            views.append((a-mean)/std)
        return views


class ImageSwAV(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.encoder=dinov3_convnext_tiny(pretrained=False)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.projection=nn.Sequential(nn.Linear(768,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,128))
        self.prototypes=nn.Linear(128,64,bias=False)

    def forward(self,images):
        features=self.encoder.forward_features(images)['x_norm_clstoken']
        projected=F.normalize(self.projection(features),dim=1)
        return features,projected,self.prototypes(projected)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight,dim=1))

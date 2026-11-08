"""Point-supervised localization for the supplied fovea annotations.

The marker radius is an annotation tool setting, not an anatomical boundary.
Image orientation is preserved; inference returns original-image pixel coordinates.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def image_tensor(path, size=384):
    with Image.open(path) as im:
        im = im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())

def normalize(x):
    return (x - x.new_tensor(MEAN)[None,:,None,None]) / x.new_tensor(STD)[None,:,None,None]

def make_backbone(weights=None):
    code = str(ROOT / 'backbone/dinov3')
    if code not in sys.path: sys.path.insert(0, code)
    from dinov3.models.convnext import ConvNeXt
    model = ConvNeXt(depths=[3,3,9,3], dims=[96,192,384,768])
    if weights:
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model

def feature_maps(backbone, x):
    fs=[]
    for down, stage in zip(backbone.downsample_layers, backbone.stages):
        x=stage(down(x)); fs.append(x)
    return fs

class ConvBlock(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.GELU(),
                         nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.GELU())

class PointHead(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.proj=nn.ModuleList([nn.Sequential(nn.Conv2d(c,channels,1),nn.GroupNorm(8,channels),nn.GELU()) for c in [96,192,384,768]])
        self.blocks=nn.ModuleList([ConvBlock(channels,channels) for _ in range(3)])
        self.out=nn.Conv2d(channels,1,1)
    def forward(self, fs):
        x=self.proj[3](fs[3])
        for i in [2,1,0]:
            x=F.interpolate(x,size=fs[i].shape[-2:],mode='bilinear',align_corners=False)+self.proj[i](fs[i])
            x=self.blocks[i](x)
        return self.out(x)[:,0]

class TinyPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.ModuleList([ConvBlock(3,16),ConvBlock(16,32),ConvBlock(32,64),ConvBlock(64,128)])
        self.dec=nn.ModuleList([ConvBlock(192,64),ConvBlock(96,32)])
        self.out=nn.Conv2d(32,1,1)
    def forward(self, x):
        fs=[]
        for layer in self.enc:
            x=F.avg_pool2d(x,2);x=layer(x);fs.append(x)
        for layer,skip in zip(self.dec,[fs[2],fs[1]]):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=layer(torch.cat([x,skip],1))
        return self.out(x)[:,0]

def gaussian_targets(xy, shape, sigma_px=12., image_size=768.):
    h,w=shape
    yy=(torch.arange(h,device=xy.device,dtype=xy.dtype)+.5)/h
    xx=(torch.arange(w,device=xy.device,dtype=xy.dtype)+.5)/w
    d=(xx[None,None,:]-xy[:,0,None,None]).square()+(yy[None,:,None]-xy[:,1,None,None]).square()
    t=torch.exp(-d/(2*(sigma_px/image_size)**2))
    return t/t.sum((1,2),keepdim=True).clamp_min(1e-10)

def point_loss(logits, xy, sigma_px=12.):
    target=gaussian_targets(xy,logits.shape[-2:],sigma_px)
    return -(target.flatten(1)*F.log_softmax(logits.flatten(1),1)).sum(1).mean()

def decode_heatmap(logits):
    """Argmax plus quadratic subpixel refinement, normalized image coordinates."""
    b,h,w=logits.shape; flat=logits.flatten(1); index=flat.argmax(1)
    y=index//w;x=index%w
    offsets=torch.arange(-1,2,device=logits.device)
    iy=(y[:,None,None]+offsets[None,:,None]).expand(b,3,3).clamp(0,h-1)
    ix=(x[:,None,None]+offsets[None,None,:]).expand(b,3,3).clamp(0,w-1)
    bid=torch.arange(b,device=logits.device)
    center=logits[bid,y,x]
    def offset(left,right):
        den=(left-2*center+right).clamp_max(-1e-6)
        return (.5*(left-right)/den).clamp(-.5,.5)
    dx=offset(logits[bid,y,(x-1).clamp_min(0)],logits[bid,y,(x+1).clamp_max(w-1)])
    dy=offset(logits[bid,(y-1).clamp_min(0),x],logits[bid,(y+1).clamp_max(h-1),x])
    dx=torch.where((x>0)&(x<w-1),dx,0.);dy=torch.where((y>0)&(y<h-1),dy,0.)
    px=(x+.5+dx)/w;py=(y+.5+dy)/h
    # Peak mass is a raw concentration score, not a calibrated probability of correctness.
    mass=(flat.softmax(1).gather(1,(iy*w+ix).flatten(1))).sum(1)
    return torch.stack([px,py],1),mass

def localization_metrics(pred,target):
    d=np.linalg.norm((np.asarray(pred)-np.asarray(target))*768.,axis=1)
    return {'n':len(d),'mean_error_px':float(d.mean()),'median_error_px':float(np.median(d)),
            'p90_error_px':float(np.quantile(d,.9)),**{f'within_{k}px':float((d<=k).mean()) for k in [10,20,30,50,80]}}

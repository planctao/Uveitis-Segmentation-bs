"""Strictly load DINOv3 and cache CLS and regional tokens, with row identity."""
import os
os.environ.setdefault('NUMEXPR_NUM_THREADS','6')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T

ROOT=Path(__file__).resolve().parent
OLD=Path('/root/autodl-tmp/uveitis_cls_20261021')
sys.path.insert(0,str(OLD/'vendor'))
from convnext import ConvNeXt
WEIGHTS=Path('/root/autodl-tmp/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')

class Images(Dataset):
    def __init__(self,frame,size):
        self.paths=frame.cache_path.tolist()
        self.transform=T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im: return self.transform(im.convert('RGB'))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=320);p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(20261124)
    df=pd.read_csv(ROOT/'metadata/manifest.csv')
    model=ConvNeXt(depths=[3,3,9,3],dims=[96,192,384,768])
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    model.eval().cuda()
    loader=DataLoader(Images(df,args.size),batch_size=args.batch,num_workers=args.workers,pin_memory=True)
    outputs={k:[] for k in ['global','regions','dispersion','mid']}
    start=time.time()
    for step,x in enumerate(loader):
        with torch.autocast('cuda',dtype=torch.float16):
            z=x.cuda(non_blocking=True)
            mid=None
            for i in range(4):
                z=model.stages[i](model.downsample_layers[i](z))
                if i==2: mid=z.mean((-2,-1))
            glob=model.norm(z.mean((-2,-1)))
            regions=model.norm(F.adaptive_avg_pool2d(z,(3,3)).flatten(2).transpose(1,2))
            dispersion=z.float().std((-2,-1)).to(z.dtype)
        for k,v in [('global',glob),('regions',regions),('dispersion',dispersion),('mid',mid)]:
            outputs[k].append(v.cpu().numpy().astype('float16'))
        if step%10==0 or step==len(loader)-1: print(json.dumps({'batch':step+1,'total':len(loader),'seconds':round(time.time()-start,1)}),flush=True)
    out=ROOT/'features';out.mkdir(exist_ok=True)
    arrays={k:np.concatenate(v) for k,v in outputs.items()}
    arrays['path']=df.path.to_numpy(dtype=str)
    arrays['manifest_sha256']=np.array(hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest())
    np.savez_compressed(out/f'convnext_{args.size}.npz',**arrays)
    meta={'backbone':'DINOv3 ConvNeXt-Tiny','weights':str(WEIGHTS),'weights_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),'manifest_sha256':str(arrays['manifest_sha256']),'shapes':{k:list(v.shape) for k,v in arrays.items()},'size':args.size,'seconds':time.time()-start,'torch':torch.__version__}
    (out/f'convnext_{args.size}.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__': main()

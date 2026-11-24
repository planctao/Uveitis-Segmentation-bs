"""Preserve finer DINO patch evidence and a label-independent foreground mask."""
import argparse
import hashlib
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import DataLoader
from extract import Images
from extract_vit import WEIGHTS
from experiment import ROOT,dump

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=7);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261124)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');base=np.load(ROOT/'features/vitb_320.npz')
    digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    assert str(base['manifest_sha256'])==digest and np.array_equal(base['path'],frame.path)
    model=timm.create_model('vit_base_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=320)
    model.load_state_dict(checkpoint_filter_fn(torch.load(WEIGHTS,map_location='cpu',weights_only=True),model),strict=True)
    model.cuda().eval();loader=DataLoader(Images(frame,320),batch_size=24,num_workers=6,pin_memory=True)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    region_outputs=[];mask_outputs=[];fallback=0;max_delta=0.;offset=0;start=time.time()
    for step,x in enumerate(loader):
        x=x.cuda(non_blocking=True)
        image=(x*std+mean).clamp(0,1)
        foreground=(image.amax(1,keepdim=True)>20/255)&(image[:,1:2]>10/255)
        occupancy=F.adaptive_avg_pool2d(foreground.float(),(a.grid,a.grid)).flatten(1)
        valid=occupancy>.2
        empty=~valid.any(1)
        if empty.any():
            fallback+=int(empty.sum().item())
            valid[empty,occupancy[empty].argmax(1)]=True
        with torch.autocast('cuda',dtype=torch.float16):
            tokens=model.forward_features(x);patch=tokens[:,model.num_prefix_tokens:]
            region=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(x),-1,20,20),(a.grid,a.grid)).flatten(2).transpose(1,2)
        reference=base['global'][offset:offset+len(x)].astype('float32')
        max_delta=max(max_delta,float(np.max(np.abs(tokens[:,0].float().cpu().numpy()-reference))))
        offset+=len(x);region_outputs.append(region.cpu().numpy().astype('float16'));mask_outputs.append(valid.cpu().numpy().astype('uint8'))
        if step%20==0:print(json.dumps({'batch':step+1,'total':len(loader),'seconds':round(time.time()-start,1)}),flush=True)
    assert max_delta<.03,'Backbone loading or preprocessing does not reproduce reference features'
    arrays={k:base[k] for k in ['global','mid','dispersion','path','manifest_sha256']}
    arrays['regions']=np.concatenate(region_outputs);arrays['region_mask']=np.concatenate(mask_outputs)
    assert arrays['region_mask'].any(1).all()
    target=ROOT/'features'/f'vitb_patch{a.grid}.npz';tmp=target.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(target)
    meta={'backbone':'DINOv3 ViT-B/16','weights':WEIGHTS,'weights_sha256':hashlib.sha256(open(WEIGHTS,'rb').read()).hexdigest(),'manifest_sha256':digest,'size':320,'grid':a.grid,'strict_loading':True,'global_mid_dispersion':'Copied exactly from vitb_320 for matched controls','reference_global_max_abs_difference':max_delta,'foreground_rule':'max(R,G,B)>20/255 and G>10/255; pooled foreground fraction>0.2; if empty keep highest-occupancy region','fallback_images':fallback,'valid_regions_min':int(arrays['region_mask'].sum(1).min()),'valid_regions_mean':float(arrays['region_mask'].sum(1).mean()),'valid_regions_max':int(arrays['region_mask'].sum(1).max()),'shapes':{k:list(v.shape) for k,v in arrays.items()},'seconds':time.time()-start}
    dump(target.with_suffix('.json'),meta);print(json.dumps(meta),flush=True)

if __name__=='__main__':main()

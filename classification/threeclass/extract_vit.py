"""DINOv3 ViT-B/16 features using timm's official checkpoint conversion."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import DataLoader
from extract import ROOT,Images

WEIGHTS='/root/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int,default=320);parser.add_argument('--batch',type=int,default=24);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261124)
    model=timm.create_model('vit_base_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=args.size)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint_filter_fn(state,model),strict=True)
    model.eval().cuda();captured={}
    def hook(module,inputs,output): captured['mid']=output
    handle=model.blocks[7].register_forward_hook(hook)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    loader=DataLoader(Images(frame,args.size),batch_size=args.batch,num_workers=6,pin_memory=True)
    outputs={k:[] for k in ['global','regions','mid','dispersion']};start=time.time()
    for step,x in enumerate(loader):
        with torch.autocast('cuda',dtype=torch.float16):
            tokens=model.forward_features(x.cuda(non_blocking=True));patch=tokens[:,model.num_prefix_tokens:]
            grid=args.size//16
            regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(x),-1,grid,grid),(3,3)).flatten(2).transpose(1,2)
            values={'global':tokens[:,0],'regions':regions,'mid':captured['mid'][:,model.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
        for k,v in values.items(): outputs[k].append(v.cpu().numpy().astype('float16'))
        if step%10==0 or step==len(loader)-1: print(json.dumps({'batch':step+1,'total':len(loader),'seconds':round(time.time()-start,1)}),flush=True)
    handle.remove();arrays={k:np.concatenate(v) for k,v in outputs.items()};arrays['path']=frame.path.to_numpy(dtype=str)
    arrays['manifest_sha256']=np.array(hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest())
    out=ROOT/'features';out.mkdir(exist_ok=True);np.savez_compressed(out/f'vitb_{args.size}.npz',**arrays)
    meta={'backbone':'DINOv3 ViT-B/16','weights':WEIGHTS,'weights_sha256':hashlib.sha256(open(WEIGHTS,'rb').read()).hexdigest(),'strict_loading':True,'converter':'timm.models.eva.checkpoint_filter_fn','timm':timm.__version__,'torch':torch.__version__,'size':args.size,'seconds':time.time()-start,'manifest_sha256':str(arrays['manifest_sha256']),'shapes':{k:list(v.shape) for k,v in arrays.items()}}
    (out/f'vitb_{args.size}.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)

if __name__=='__main__': main()

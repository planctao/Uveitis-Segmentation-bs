"""Adapt final DINOv3 ViT blocks with the validation-selected hierarchical head.

Keep pretrained frozen parameters by reference and save only adapted parameters.
Epoch zero is eligible, so fine tuning cannot displace its frozen initialization
unless the fixed validation criterion improves. No test set is loaded for fitting.
"""
import os
os.environ.setdefault('NUMEXPR_NUM_THREADS','4')
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from PIL import Image
from experiment import ROOT,Model,loss_fn,sample_weights,metrics,dump,summarize_group
from extract_vit import WEIGHTS

class Images(Dataset):
    def __init__(self,frame,diseases,augment=False):
        self.frame=frame.reset_index(drop=True);self.weights=sample_weights(self.frame,self.frame.label)
        self.disease=self.frame.disease.map({d:i for i,d in enumerate(diseases)}).fillna(-1).astype(int).to_numpy()
        ops=[T.Resize((320,320),antialias=True)]
        if augment: ops += [T.RandomHorizontalFlip(),T.ColorJitter(brightness=.12,contrast=.12,saturation=.06)]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.cache_path) as im:x=self.transform(im.convert('RGB'))
        return x,int(row.label),int(self.disease[i]),float(self.weights[i])

class Fine(nn.Module):
    def __init__(self,checkpoint,blocks):
        super().__init__()
        self.backbone=timm.create_model('vit_base_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=320)
        state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(state,self.backbone),strict=True)
        for p in self.backbone.parameters():p.requires_grad=False
        for block in self.backbone.blocks[-blocks:]:
            for p in block.parameters():p.requires_grad=True
        for p in self.backbone.norm.parameters():p.requires_grad=True
        self.head=Model(checkpoint['cfg'],checkpoint['dims'],len(checkpoint['diseases']))
        self.head.load_state_dict(checkpoint['state_dict'],strict=True)
        self.stats=checkpoint['stats'];self.captured={}
        self.backbone.blocks[7].register_forward_hook(self.capture)
        for k,s in self.stats.items():
            self.register_buffer('mean_'+k,torch.from_numpy(s['mean']))
            self.register_buffer('std_'+k,torch.from_numpy(s['std']))
    def capture(self,module,inputs,output):self.captured['mid']=output
    def forward(self,x):
        tokens=self.backbone.forward_features(x);patch=tokens[:,self.backbone.num_prefix_tokens:]
        regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(x),-1,20,20),(3,3)).flatten(2).transpose(1,2)
        features={'global':tokens[:,0],'regions':regions,'mid':self.captured.pop('mid')[:,self.backbone.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
        features={k:((v.float()-getattr(self,'mean_'+k))/getattr(self,'std_'+k)).clamp(-10,10) for k,v in features.items()}
        return self.head(features)
    def adapted(self):
        return {k:p.detach().cpu().clone() for k,p in self.backbone.named_parameters() if p.requires_grad}

@torch.inference_mode()
def predict(model,loader):
    model.eval();probs=[]
    for x,*_ in loader:
        with torch.autocast('cuda',dtype=torch.float16):o=model(x.cuda(non_blocking=True))
        probs.append(o['p'].float().cpu().numpy())
    return np.concatenate(probs)

def train(seed_checkpoint,frame,blocks,lr,seed,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return
    random.seed(20261124+seed);np.random.seed(20261124+seed);torch.manual_seed(20261124+seed);torch.cuda.manual_seed_all(20261124+seed)
    ckpt=torch.load(seed_checkpoint,map_location='cpu',weights_only=False);cfg=ckpt['cfg'];model=Fine(ckpt,blocks).cuda()
    tr=frame[frame.split.eq('train')];va=frame[frame.split.eq('val')]
    loader=DataLoader(Images(tr,ckpt['diseases'],True),batch_size=20,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(20261124+seed))
    val=DataLoader(Images(va,ckpt['diseases']),batch_size=32,num_workers=4,pin_memory=True,persistent_workers=True)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':lr},{'params':model.head.parameters(),'lr':.0001}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');history=[];wait=0;best=None;start=time.time()
    for epoch in range(9):
        loss_sum=0
        if epoch:
            model.train()
            for x,y,disease,w in loader:
                x,y,disease,w=x.cuda(non_blocking=True),y.cuda(),disease.cuda(),w.float().cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):o=model(x)
                o={k:v.float() for k,v in o.items()}
                loss=loss_fn(o,y,disease,{'leaf':w},cfg)
                if not torch.isfinite(loss):raise ValueError('Nonfinite fine-tuning loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),3)
                scaler.step(optimizer);scaler.update();loss_sum+=loss.item()*len(y)
        p=predict(model,val);m=metrics(va.label,p);key=(m['macro_f1'],-m['log_loss'])
        history.append({'epoch':epoch,'train_loss':loss_sum/len(tr),'val':m})
        print(json.dumps({'event':'fine_epoch','path':str(out),'epoch':epoch,'val':m,'seconds':time.time()-start}),flush=True)
        if best is None or key>best:
            best=key;wait=0
            package={**ckpt,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'backbone_adapted':model.adapted(),'adapt_blocks':blocks,'backbone_lr':lr,'epoch':epoch,'val':m,'initial_head_checkpoint':str(seed_checkpoint)}
            torch.save(package,out/'model.pt');np.savez_compressed(out/'val_predictions.npz',p=p,g=np.zeros((len(p),0),dtype='float32'))
        else:wait+=1
        if wait>=3:break
    dump(out/'history.json',history)
    dump(out/'summary.json',{'cfg':cfg,'adapt_blocks':blocks,'backbone_lr':lr,'epoch':package['epoch'],'val':package['val'],'seconds':time.time()-start,'manifest_sha256':ckpt['manifest_sha256']})
    del model,loader,val;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);a=p.parse_args();torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    selection=json.loads((ROOT/'runs/vitb_320/explore/selection.json').read_text());initial=selection['best'];assert initial['mode'] in ['flat','dual']
    out=ROOT/'runs/vitb_320/finetune';groups=[]
    for blocks,lr in [(1,3e-6),(2,1e-5)]:
        group=out/f'blocks{blocks}_lr{lr}'
        for seed in range(a.seeds):train(initial['checkpoints'][seed],frame,blocks,lr,seed,group/f'seed_{seed}')
        summary=summarize_group(group,frame);summary['adapt_blocks']=blocks;summary['backbone_lr']=lr;groups.append(summary)
        dump(out/'leaderboard.json',sorted(groups,key=lambda x:x['val']['macro_f1'],reverse=True))
    best=max(groups,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':selection['manifest_sha256'],'selection_rule':'Maximum validation image macro-F1; original validation split; 3 seeds; epoch zero eligible; external tests untouched during adaptation.','best':best,'frozen_initialization':initial,'improved_over_frozen':best['val']['macro_f1']>initial['val']['macro_f1'],'selected':{'fine':best}})

if __name__=='__main__':main()

"""Reproducible cache, grouped cross-validation, and full-data fit for fovea points."""
from __future__ import annotations
import argparse, copy, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bs.white_spot import (image_tensor,normalize,make_backbone,feature_maps,PointHead,TinyPointNet,point_loss,decode_heatmap,localization_metrics)
DEFAULT=ROOT/'outputs/white_spot_20261109'

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def read_manifest(out):return json.loads((out/'manifest.json').read_text())

def cache(args):
    rows=read_manifest(args.output);dest=args.output/'cache';dest.mkdir(exist_ok=True)
    backbone=make_backbone(ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth').cuda().eval()
    n=len(rows);pooled=[]
    for start in range(0,n,args.batch_size):
        part=rows[start:start+args.batch_size]
        x=torch.stack([image_tensor(r['image'],args.size) for r in part]).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):fs=feature_maps(backbone,normalize(x))
        desc=torch.cat([f.float().mean((2,3)) for f in fs],1).cpu().numpy()
        pooled.append(desc)
        for i,r in enumerate(part):
            if r['posterior']:
                p=dest/(r['id'].replace('/','__')+'.pt')
                torch.save({'maps':[f[i].cpu().half() for f in fs]},p)
        if start%100<args.batch_size:print(f'cache {min(start+args.batch_size,n)}/{n}',flush=True)
    np.savez_compressed(dest/'global.npz',features=np.concatenate(pooled),ids=np.array([r['id'] for r in rows]))
    (dest/'metadata.json').write_text(json.dumps({'size':args.size,'backbone':'DINOv3 ConvNeXt-Tiny, supplied pretrained weights','manifest_sha256':__import__('hashlib').sha256((args.output/'manifest.json').read_bytes()).hexdigest()}))

def baselines(args):
    from joblib import dump
    rows=read_manifest(args.output);g=np.load(args.output/'cache/global.npz');X=g['features']
    assert list(g['ids'])==[r['id'] for r in rows]
    results=[];outputs=[];clsout=[]
    for fold in range(3):
        train=np.array([i for i,r in enumerate(rows) if r['cv_fold']!=fold]);test=np.array([i for i,r in enumerate(rows) if r['cv_fold']==fold and not r['augmented']])
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=2000))
        clf.fit(X[train],[rows[i]['posterior'] for i in train]);probs=clf.predict_proba(X[test])[:,1];y=np.array([rows[i]['posterior'] for i in test])
        cm={'fold':fold,'auc':float(roc_auc_score(y,probs)),'balanced_accuracy':float(balanced_accuracy_score(y,probs>=.5)),'n':len(test)}
        for i,p in zip(test,probs):clsout.append({'id':rows[i]['id'],'posterior':rows[i]['posterior'],'probability':float(p)})
        itrain=np.array([i for i in train if rows[i]['posterior']]);itest=np.array([i for i in test if rows[i]['posterior']]);target=np.array([[rows[i]['x']/768,rows[i]['y']/768] for i in itest]);yt=np.array([[rows[i]['x']/768,rows[i]['y']/768] for i in itrain])
        # Family balancing prevents the 11 original images with 10 augmentations dominating the fit.
        counts=__import__('collections').Counter(rows[i]['family'] for i in itrain);sw=np.array([1/counts[rows[i]['family']] for i in itrain])
        for name in ['center','train_mean','ridge']:
            if name=='center':pred=np.tile([.5,.5],(len(itest),1))
            elif name=='train_mean':pred=np.tile(np.average(yt,axis=0,weights=sw),(len(itest),1))
            else:
                model=make_pipeline(StandardScaler(),Ridge(alpha=1000.))
                model.fit(X[itrain],yt,ridge__sample_weight=sw);pred=model.predict(X[itest]).clip(0,1)
            results.append({'method':name,'fold':fold,**localization_metrics(pred,target)})
            for i,p,t in zip(itest,pred,target):outputs.append({'method':name,'fold':fold,'id':rows[i]['id'],'pred_x':float(p[0]),'pred_y':float(p[1]),'target_x':float(t[0]),'target_y':float(t[1])})
        results.append({'method':'posterior_classifier',**cm})
    (args.output/'baseline_results.json').write_text(json.dumps(results,indent=2));(args.output/'baseline_predictions.json').write_text(json.dumps(outputs,indent=2));(args.output/'classifier_oof.json').write_text(json.dumps(clsout,indent=2))
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=2000));clf.fit(X,[r['posterior'] for r in rows]);dump(clf,args.output/'posterior_classifier.joblib')
    print(json.dumps(results,indent=2),flush=True)

class PointData:
    def __init__(self, rows,args):
        self.rows=rows;self.args=args;self.data={}
        for r in rows:
            if args.method=='tiny':self.data[r['id']]=image_tensor(r['image'],args.size)
            else:self.data[r['id']]=torch.load(args.output/'cache'/(r['id'].replace('/','__')+'.pt'),weights_only=True)['maps']
    def batch(self,indices,aug=False):
        rr=[self.rows[i] for i in indices]
        if self.args.method=='tiny':xs=torch.stack([self.data[r['id']] for r in rr]).cuda()
        else:xs=[torch.stack([self.data[r['id']][j] for r in rr]).cuda().float() for j in range(4)]
        xy=torch.tensor([[r['x']/768,r['y']/768] for r in rr],device='cuda',dtype=torch.float32)
        # Consistent transforms per sample; no test-time labels enter augmentation.
        if aug:
            items=[xs] if isinstance(xs,torch.Tensor) else xs
            for i in range(len(rr)):
                for dim,axis in [(-1,0),(-2,1)]:
                    if random.random()<.5:
                        for x in items:x[i]=x[i].flip(dim)
                        xy[i,axis]=1-xy[i,axis]
                if self.args.method=='tiny':
                    xs[i]=(xs[i]*random.uniform(.8,1.2)+random.uniform(-.08,.08)).clamp(0,1)
                    if self.args.translate:
                        dx=random.randint(-self.args.translate,self.args.translate);dy=random.randint(-self.args.translate,self.args.translate)
                        shift=xy.new_tensor([dx/xs.shape[-1],dy/xs.shape[-2]])
                        if ((xy[i]+shift>.02)&(xy[i]+shift<.98)).all():
                            moved=torch.roll(xs[i],shifts=(dy,dx),dims=(-2,-1))
                            if dy>0:moved[:,:dy,:]=0
                            elif dy<0:moved[:,dy:,:]=0
                            if dx>0:moved[:,:,:dx]=0
                            elif dx<0:moved[:,:,dx:]=0
                            xs[i]=moved;xy[i]+=shift
        return xs,xy

@torch.inference_mode()
def predict(model,data,idx,bs,tta=False):
    model.eval();pred=[];truth=[]
    for start in range(0,len(idx),bs):
        x,xy=data.batch(idx[start:start+bs]);items=[x] if isinstance(x,torch.Tensor) else x
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(x)
            if tta:
                maps=[logits]
                for dims in [(-1,),(-2,),(-2,-1)]:
                    xx=[a.flip(dims) for a in items];p=model(xx[0] if isinstance(x,torch.Tensor) else xx).flip(dims);maps.append(p)
                # Spatial distributions are normalized before averaging, no per-image offsets.
                probs=torch.stack([a.float().flatten(1).softmax(1).reshape_as(a) for a in maps]).mean(0)
                logits=probs.clamp_min(1e-12).log()
        p,_=decode_heatmap(logits.float());pred.extend(p.cpu().numpy());truth.extend(xy.cpu().numpy())
    return np.asarray(pred),np.asarray(truth)

def run_name(args):
    return f"{args.method}_s{args.seed}"+(f"_shift{args.translate}" if args.translate else "")

def train(args):
    rows=[r for r in read_manifest(args.output) if r['posterior']];seed_all(args.seed)
    data=PointData(rows,args);summary=[]
    folds=[-1] if args.final else ([args.fold] if args.fold is not None else range(3))
    for fold in folds:
        seed_all(args.seed+max(fold,0));where=args.output/run_name(args)/('full' if fold==-1 else f'fold{fold}');where.mkdir(parents=True,exist_ok=True)
        trainidx=[i for i,r in enumerate(rows) if r['cv_fold']!=fold];testidx=[i for i,r in enumerate(rows) if r['cv_fold']==fold and not r['augmented']]
        if fold!=-1:
            # Inner grouped validation only chooses the stopping epoch; outer fold remains untouched.
            rr=[rows[i] for i in trainidx]
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.22,random_state=args.seed+fold).split(rr,groups=[r['group'] for r in rr]))
            valgroups={rr[i]['group'] for i in b};validx=[i for i in trainidx if rows[i]['group'] in valgroups and not rows[i]['augmented']];trainidx=[i for i in trainidx if rows[i]['group'] not in valgroups]
        else:validx=[]
        assert not ({rows[i]['group'] for i in trainidx}&{rows[i]['group'] for i in testidx+validx})
        split={'train_ids':[rows[i]['id'] for i in trainidx],'val_ids':[rows[i]['id'] for i in validx],'test_ids':[rows[i]['id'] for i in testidx]};(where/'split.json').write_text(json.dumps(split,indent=2))
        model=(TinyPointNet() if args.method=='tiny' else PointHead()).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');best=1e9;bestepoch=0
        counts=__import__('collections').Counter(rows[i]['family'] for i in trainidx);w=np.array([1/counts[rows[i]['family']] for i in trainidx]);w/=w.sum()
        history=[];start=time.time()
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            for pg in opt.param_groups:pg['lr']=args.lr*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
            # Fixed original-family-balanced number of updates; augmented copies do not dominate.
            sampled=np.random.choice(trainidx,size=max(64,len(set(rows[i]['family'] for i in trainidx))*2),replace=True,p=w)
            for k in range(0,len(sampled),args.batch_size):
                x,xy=data.batch(sampled[k:k+args.batch_size],True);opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):logits=model(x);loss=point_loss(logits.float(),xy,args.sigma)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch {epoch}')
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();losses.append(float(loss))
            vm={}
            if validx:
                p,t=predict(model,data,validx,args.batch_size);vm=localization_metrics(p,t);score=vm['mean_error_px']
            else:score=-epoch
            hist={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':round(time.time()-start,2),**{f'val_{k}':v for k,v in vm.items()}};history.append(hist)
            if score<best:
                best=score;bestepoch=epoch
                torch.save({'model':model.state_dict(),'method':args.method,'size':args.size,'epoch':epoch,'seed':args.seed,'sigma':args.sigma,'target':'macula_center','fold':fold,'translate':args.translate},where/'best.pt')
            if epoch%10==0 or epoch==1:print(json.dumps({'method':args.method,'fold':fold,'bestepoch':bestepoch,**hist}),flush=True)
            if epoch%10==0:(where/'history.json').write_text(json.dumps(history,indent=2))
        (where/'history.json').write_text(json.dumps(history,indent=2))
        ckpt=torch.load(where/'best.pt',weights_only=True);model.load_state_dict(ckpt['model'])
        if testidx:
            pred,target=predict(model,data,testidx,args.batch_size,tta=True);metrics=localization_metrics(pred,target)
            pp=[{'id':rows[i]['id'],'image':rows[i]['image'],'group':rows[i]['group'],'pred_x':float(p[0]),'pred_y':float(p[1]),'target_x':float(t[0]),'target_y':float(t[1]),'error_px':float(np.linalg.norm(p-t)*768)} for i,p,t in zip(testidx,pred,target)]
            result={'method':args.method,'seed':args.seed,'fold':fold,'selected_epoch':int(ckpt['epoch']),'selection':'inner grouped validation mean localization error; outer original images evaluated once',**metrics}
            (where/'test_predictions.json').write_text(json.dumps(pp,indent=2));(where/'metrics.json').write_text(json.dumps(result,indent=2));summary.append(result);print('OUTER',json.dumps(result),flush=True)
    if args.final:
        fit={'run':run_name(args),'fit':'all annotated image families','epochs':args.epochs,'target':'macula_center','n_original_images':sum(not r['augmented'] for r in rows),'n_images_with_augmentations':len(rows)}
        (args.output/run_name(args)/'full/fit_summary.json').write_text(json.dumps(fit,indent=2))
    else:
        (args.output/run_name(args)/'summary.json').write_text(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['cache','baselines','train']);ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--size',type=int,default=384);ap.add_argument('--batch-size',type=int,default=12);ap.add_argument('--method',choices=['frozen','tiny'],default='frozen');ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--sigma',type=float,default=12.);ap.add_argument('--seed',type=int,default=3407);ap.add_argument('--fold',type=int,choices=[0,1,2]);ap.add_argument('--translate',type=int,default=0);ap.add_argument('--final',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    {'cache':cache,'baselines':baselines,'train':train}[args.action](args)
if __name__=='__main__':main()

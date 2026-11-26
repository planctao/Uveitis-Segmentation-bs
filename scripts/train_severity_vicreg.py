"""Fixed-budget full-encoder VICReg training; fitting images only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_swav_images import FAMulticrop
from bs.severity_vicreg import ImageVICReg,vicreg_loss
from bs.severity_synthetic import SYNTHETIC_METHODS,SyntheticDirectionOrderer,synthetic_views


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fold',required=True,choices=['f1','f2','f3','f4','f5'])
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.25)
    root=ROOT/'outputs/severity_research_20261013'
    run=root/'domain_ssl/vicreg'/f'seed{args.seed}'/args.fold
    if args.probe:run=root/'domain_ssl/probes'/f'vicreg_seed{args.seed}_{args.fold}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():raise RuntimeError('Completed task already exists')
    frame=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str})
    train=(frame.fold!=args.fold)&~frame.exclude_exact_group
    test=(frame.fold==args.fold)&~frame.exclude_exact_group
    assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
    seed=args.seed+1000*int(args.fold[1:])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    weights=ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    paths=[Path(__file__),ROOT/'src/bs/severity_vicreg.py',ROOT/'src/bs/severity_swav_images.py',
        ROOT/'src/bs/severity_synthetic.py',ROOT/'docs/severity_vicreg_protocol_20261013.md']
    sources={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(run/'plan.json',dict(seed=args.seed,actual_seed=seed,fold=args.fold,probe=args.probe,
        epochs=20,batch_size=16,loss_coefficients=[25,25,1],projection_dimensions=[768,512,512,128],
        train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist(),
        masks_read=False,manual_grading_labels_read=False,sources_sha256=sources,pretrained_sha256=digest(weights)))
    model=ImageVICReg(weights).cuda().train()
    initial_stem=model.encoder.downsample_layers[0][0].weight.detach().clone()
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-5,base_lr=1e-5),
        dict(params=model.expander.parameters(),lr=1e-3,base_lr=1e-3)],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    trace=[];start_epoch=0
    if (run/'latest.pt').exists() and not args.probe:
        resume=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert resume['sources_sha256']==sources
        model.load_state_dict(resume['model'],strict=True);optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
        start_epoch=resume['epoch'];trace=resume['trace']
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        print('resuming completed epoch',start_epoch,flush=True)
    dataset=FAMulticrop(frame.loc[train,'image_path'])
    batches=len(dataset)//16;total_iters=20*batches
    start=time.monotonic();total_success=0;total_skipped=0
    for epoch in range(start_epoch,1 if args.probe else 20):
        loader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=2,pin_memory=True,
            generator=torch.Generator().manual_seed(seed+epoch))
        model.train();seen=0;sums=dict(loss=0.,invariance=0.,variance=0.,covariance=0.,mean_std=0.)
        successful=0;skipped=0;epoch_start=time.monotonic()
        for step,views in enumerate(loader):
            iteration=epoch*batches+step
            factor=(iteration+1)/batches if iteration<batches else .01+.99*(1+math.cos(math.pi*(iteration-batches)/(total_iters-batches)))/2
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                _,x=model(views[0].cuda(non_blocking=True));_,y=model(views[1].cuda(non_blocking=True))
            loss,parts=vicreg_loss(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite VICReg loss')
            old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grad_norm=float(torch.stack([p.grad.detach().float().norm() for p in model.encoder.parameters() if p.grad is not None]).norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            was_skipped=scaler.get_scale()<old_scale
            successful+=int(not was_skipped);skipped+=int(was_skipped)
            total_success+=int(not was_skipped);total_skipped+=int(was_skipped)
            sums['loss']+=float(loss.detach())*16
            for name,value in parts.items():sums[name]+=float(value.detach())*16
            seen+=16
            if step%25==0:
                atomic_json(run/'progress.json',dict(status='training',pid=os.getpid(),epoch=epoch+1,step=step+1,batches=batches,
                    loss=sums['loss']/seen,elapsed_seconds=time.monotonic()-start,successful_updates=total_success,amp_skips=total_skipped))
                print(epoch+1,step+1,'loss',sums['loss']/seen,'mean_std',sums['mean_std']/seen,flush=True)
            if args.probe and step==2:
                torch.cuda.synchronize()
                stem_delta=float((model.encoder.downsample_layers[0][0].weight-initial_stem).abs().max())
                probe=dict(steps=3,n_fit=len(dataset),successful_updates=total_success,amp_skips=total_skipped,
                    finite_loss=True,encoder_gradient_norm_last_step=grad_norm,encoder_stem_max_abs_update=stem_delta,
                    peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    elapsed_seconds=time.monotonic()-start,sources_sha256=sources)
                atomic_json(run/'probe.json',probe)
                if total_success==0 or not math.isfinite(grad_norm) or grad_norm==0 or stem_delta==0:
                    raise RuntimeError('Probe did not verify finite encoder updates')
                print('VICReg resource/gradient probe passed',flush=True);return
        row=dict(epoch=epoch+1,n_seen=seen,successful_updates=successful,amp_skips=skipped,
            seconds=time.monotonic()-epoch_start,encoder_gradient_norm_last_step=grad_norm,
            encoder_stem_max_abs_update=float((model.encoder.downsample_layers[0][0].weight-initial_stem).abs().max()))
        row.update({f'mean_{name}':value/seen for name,value in sums.items()})
        trace.append(row);pd.DataFrame(trace).to_csv(run/'training_trace.csv',index=False)
        state=dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            trace=trace,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sources_sha256=sources)
        torch.save(state,run/'latest.tmp');(run/'latest.tmp').replace(run/'latest.pt')
        print('completed epoch',epoch+1,row['mean_loss'],row['seconds'],flush=True)
    finish(model,frame,train,test,run,args,sources,trace,start)


def finish(model,frame,train,test,run,args,sources,trace,start):
    model.eval();features=[[] for _ in range(6)];projections=[[] for _ in range(6)]
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    atomic_json(run/'progress.json',dict(status='extracting_final_features',pid=os.getpid(),completed_epochs=len(trace)))
    for offset in range(0,len(frame),8):
        images=[]
        for row in frame.iloc[offset:offset+8].itertuples():
            rgb=np.asarray(Image.open(row.image_path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=float)/255
            synth,_=synthetic_views(rgb,row.uid)
            images.append([np.rint(rgb*255).astype(np.uint8),np.rint(rgb**.8*255).astype(np.uint8),np.rint(rgb**1.2*255).astype(np.uint8),*synth])
        for condition in range(6):
            tensor=torch.from_numpy(np.stack([a[condition] for a in images])).permute(0,3,1,2).float().cuda()/255
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                encoded,projected=model((tensor-mean)/std)
            features[condition].append(encoded.float().cpu().numpy());projections[condition].append(projected.float().cpu().numpy())
        if offset%200==0:print('cached final encoder',min(offset+8,len(frame)),len(frame),flush=True)
    encoded=np.stack([np.concatenate(a) for a in features])
    projected=np.stack([np.concatenate(a) for a in projections])
    np.savez_compressed(run/'final_representations.npz',uids=frame.uid.to_numpy(dtype=str),features=encoded,projections=projected)
    predictions=[];responses=[];diagnostics={}
    for method in SYNTHETIC_METHODS:
        calibrator=SyntheticDirectionOrderer(method,args.seed).fit(encoded[0,train],encoded[3,train],encoded[4,train],encoded[5,train],encoded[1,train],encoded[2,train])
        name=f'vicreg_encoder_{method}';joblib.dump(calibrator,run/f'{name}.joblib',compress=3);diagnostics[name]=calibrator.diagnostics
        for gi,gamma in enumerate([1.,.8,1.2]):
            scores,levels=calibrator.predict(encoded[gi,test])
            predictions.extend(dict(uid=uid,scope='main',fold=args.fold,method=name,gamma=gamma,score=float(score),level=int(level))
                for uid,score,level in zip(frame.loc[test,'uid'],scores,levels))
        raw=calibrator.raw_score(encoded[0,test]);low=calibrator.raw_score(encoded[3,test]);high=calibrator.raw_score(encoded[4,test])
        responses.append(dict(method=name,fold=args.fold,n=int(test.sum()),monotonic_fraction=float(((low>raw)&(high>low)).mean())))
    for name,z in [('encoder',encoded[0,train]),('projection',projected[0,train])]:
        singular=np.linalg.svd(z-z.mean(0),compute_uv=False);variance=singular.astype(float)**2;variance/=variance.sum()
        nonzero=variance[variance>0]
        diagnostics[name]=dict(std_mean=float(z.std(0).mean()),effective_covariance_rank=float(np.exp(-(nonzero*np.log(nonzero)).sum())))
    prediction_path=run/'predictions_frozen_before_evaluation.csv'
    pd.DataFrame(predictions).to_csv(prediction_path,index=False)
    pd.DataFrame(responses).to_csv(run/'synthetic_response.csv',index=False)
    atomic_json(run/'diagnostics.json',diagnostics)
    atomic_json(run/'complete.json',dict(seed=args.seed,fold=args.fold,epochs=len(trace),source_sha256=sources,
        optimizer_updates=sum(row['successful_updates'] for row in trace),amp_skips=sum(row['amp_skips'] for row in trace),
        predictions_sha256=digest(prediction_path),masks_read=False,manual_grading_labels_read=False,
        elapsed_seconds=time.monotonic()-start,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20))
    print('VICReg task complete',args.seed,args.fold,flush=True)


if __name__=='__main__':main()

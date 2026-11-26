"""Full encoder domain SwAV; training only reads fitting-fold image inputs."""
from __future__ import annotations
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
from bs.severity_ssl import sinkhorn_assignments,swapped_prediction_loss
from bs.severity_swav_images import FAMulticrop,ImageSwAV
from bs.severity_synthetic import SYNTHETIC_METHODS,SyntheticDirectionOrderer,synthetic_views


def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',required=True)
    p.add_argument('--probe',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    root=ROOT/'outputs/severity_research_20261013'
    run=root/'domain_ssl/swav'/f'seed{args.seed}'/args.fold
    if args.probe:run=root/'domain_ssl/probes'/f'swav_seed{args.seed}_{args.fold}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():raise RuntimeError('Completed SwAV task exists')
    frame=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str})
    train=(frame.fold!=args.fold)&~frame.exclude_exact_group;test=(frame.fold==args.fold)&~frame.exclude_exact_group
    assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
    seed=args.seed+1000*int(args.fold[1:]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    weights=ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    paths=[Path(__file__),ROOT/'src/bs/severity_swav_images.py',ROOT/'src/bs/severity_ssl.py',ROOT/'src/bs/severity_synthetic.py',
           ROOT/'docs/severity_domain_ssl_protocol_20261013.md',ROOT/'docs/severity_swav_addendum_20261013.md']
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plan=dict(seed=args.seed,actual_seed=seed,fold=args.fold,epochs=20,batch_size=8,prototypes=64,probe=args.probe,
              train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist(),
              masks_read=False,manual_grading_labels_read=False,sources_sha256=sources,
              pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    atomic_json(run/'plan.json',plan)
    model=ImageSwAV(weights).cuda().train()
    initial_stem=model.encoder.downsample_layers[0][0].weight.detach().clone()
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-5,base_lr=1e-5),
                                dict(params=[*model.projection.parameters(),*model.prototypes.parameters()],lr=1e-3,base_lr=1e-3)],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    queues=torch.zeros(2,512,128,device='cuda');counts=[0,0];trace=[];start_epoch=0
    if (run/'latest.pt').exists() and not args.probe:
        resume=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert resume['sources_sha256']==sources
        model.load_state_dict(resume['model'],strict=True);optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
        queues=resume['queues'].cuda();counts=resume['counts'];start_epoch=resume['epoch'];trace=resume['trace']
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        print('resuming completed epoch',start_epoch,flush=True)
    dataset=FAMulticrop(frame.loc[train,'image_path']);batches=len(dataset)//8;total_iters=20*batches
    elapsed_start=time.monotonic();successful_updates=0;skipped_updates=0
    for epoch in range(start_epoch,1 if args.probe else 20):
        loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,drop_last=True,num_workers=2,pin_memory=True,
                                           generator=torch.Generator().manual_seed(seed+epoch))
        model.train();epoch_loss=0.;seen=0;epoch_success=0;epoch_skipped=0;epoch_start=time.monotonic()
        for step,views in enumerate(loader):
            iteration=epoch*batches+step
            factor=(iteration+1)/batches if iteration<batches else .01+.99*(1+math.cos(math.pi*(iteration-batches)/(total_iters-batches)))/2
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            model.normalize_prototypes();optimizer.zero_grad(set_to_none=True)
            projections=[];outputs=[]
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                for view in views:
                    _,projection,output=model(view.cuda(non_blocking=True));projections.append(projection);outputs.append(output)
            targets=[]
            for crop_id in range(2):
                with torch.no_grad():
                    current=outputs[crop_id].detach().float()
                    if epoch>=1 and counts[crop_id]==512:
                        current=torch.cat([queues[crop_id]@model.prototypes.weight.detach().T,current])
                    targets.append(sinkhorn_assignments(current)[-8:])
                    queues[crop_id,8:]=queues[crop_id,:-8].clone();queues[crop_id,:8]=projections[crop_id].detach().float()
                    counts[crop_id]=min(512,counts[crop_id]+8)
            loss=swapped_prediction_loss([x.float() for x in outputs],targets)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SwAV loss')
            old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
            encoder_grad_norm=float(torch.stack([p.grad.detach().float().norm() for p in model.encoder.parameters() if p.grad is not None]).norm())
            if epoch==0:model.prototypes.weight.grad=None
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<old_scale
            epoch_skipped+=int(skipped);epoch_success+=int(not skipped)
            successful_updates+=int(not skipped);skipped_updates+=int(skipped)
            epoch_loss+=float(loss.detach())*8;seen+=8
            if step%25==0:
                atomic_json(run/'progress.json',dict(status='training',pid=os.getpid(),epoch=epoch+1,step=step+1,batches=batches,
                    loss=epoch_loss/seen,elapsed_seconds=time.monotonic()-elapsed_start,successful_updates=successful_updates,amp_skips=skipped_updates))
                print(f'epoch={epoch+1} step={step+1}/{batches} loss={epoch_loss/seen:.6f} elapsed={time.monotonic()-elapsed_start:.1f}s',flush=True)
            if args.probe and step==2:
                torch.cuda.synchronize()
                atomic_json(run/'probe.json',dict(n_fit=len(dataset),steps=3,successful_updates=successful_updates,amp_skips=skipped_updates,
                    elapsed_seconds=time.monotonic()-elapsed_start,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
                    peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,finite_loss=True,
                    encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),sources_sha256=sources))
                probe=json.loads((run/'probe.json').read_text())
                probe.update(encoder_gradient_norm_last_step=encoder_grad_norm,
                             encoder_stem_max_abs_update=float((model.encoder.downsample_layers[0][0].weight-initial_stem).abs().max()))
                atomic_json(run/'probe.json',probe)
                if successful_updates==0 or probe['encoder_stem_max_abs_update']==0 or not math.isfinite(encoder_grad_norm) or encoder_grad_norm==0:
                    raise RuntimeError('Probe did not verify finite encoder updates')
                print('CUDA SwAV probe passed',flush=True);return
        trace.append(dict(epoch=epoch+1,mean_loss=epoch_loss/seen,n_seen=seen,successful_updates=epoch_success,amp_skips=epoch_skipped,
                          seconds=time.monotonic()-epoch_start,encoder_gradient_norm_last_step=encoder_grad_norm,
                          encoder_stem_max_abs_update=float((model.encoder.downsample_layers[0][0].weight-initial_stem).abs().max())))
        pd.DataFrame(trace).to_csv(run/'training_trace.csv',index=False)
        state=dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),queues=queues.cpu(),
                   counts=counts,trace=trace,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sources_sha256=sources)
        torch.save(state,run/'latest.tmp');(run/'latest.tmp').replace(run/'latest.pt')
        print('finished epoch',epoch+1,'mean_loss',epoch_loss/seen,'seconds',trace[-1]['seconds'],flush=True)
    finish_and_calibrate(model,frame,train,test,run,args,sources,trace,elapsed_start)


def finish_and_calibrate(model,frame,train,test,run,args,sources,trace,elapsed_start):
    """Final feature cache and label-free scores; never loads evaluation masks."""
    model=model.eval();features=[[] for _ in range(6)];projections=[[] for _ in range(6)]
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    atomic_json(run/'progress.json',dict(status='extracting_final_features',pid=os.getpid(),completed_epochs=len(trace)))
    for offset in range(0,len(frame),8):
        chunk=frame.iloc[offset:offset+8];all_views=[]
        for r in chunk.itertuples():
            rgb=np.asarray(Image.open(r.image_path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float64)/255
            original=np.rint(rgb*255).astype(np.uint8)
            gamma08=np.rint(rgb**.8*255).astype(np.uint8);gamma12=np.rint(rgb**1.2*255).astype(np.uint8)
            synth,_=synthetic_views(rgb,r.uid);all_views.append([original,gamma08,gamma12,*synth])
        for ci in range(6):
            x=torch.from_numpy(np.stack([v[ci] for v in all_views])).permute(0,3,1,2).float().cuda()/255
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                f,z,_=model((x-mean)/std)
            features[ci].append(f.float().cpu().numpy());projections[ci].append(z.float().cpu().numpy())
        if offset%200==0:print('cached final encoder',min(offset+8,len(frame)),len(frame),flush=True)
    encoded=np.stack([np.concatenate(x) for x in features]);projected=np.stack([np.concatenate(x) for x in projections])
    prototype_weights=model.prototypes.weight.detach().float().cpu().numpy()
    np.savez_compressed(run/'final_representations.npz',uids=frame.uid.to_numpy(dtype=str),features=encoded,projections=projected,prototype_weights=prototype_weights)
    predictions=[];responses=[];diagnostics={}
    for method in SYNTHETIC_METHODS:
        calibrator=SyntheticDirectionOrderer(method,args.seed).fit(encoded[0,train],encoded[3,train],encoded[4,train],encoded[5,train],encoded[1,train],encoded[2,train])
        name=f'swav_encoder_{method}';joblib.dump(calibrator,run/f'{name}.joblib',compress=3);diagnostics[name]=calibrator.diagnostics
        for gi,gamma in enumerate([1.,.8,1.2]):
            scores,levels=calibrator.predict(encoded[gi,test])
            for uid,score,level in zip(frame.loc[test,'uid'],scores,levels):
                predictions.append(dict(uid=uid,scope='main',fold=args.fold,method=name,gamma=gamma,score=float(score),level=int(level)))
        raw=calibrator.raw_score(encoded[0,test]);low=calibrator.raw_score(encoded[3,test]);high=calibrator.raw_score(encoded[4,test])
        responses.append(dict(method=name,fold=args.fold,n=int(test.sum()),monotonic_fraction=float(((low>raw)&(high>low)).mean())))
    # Order the 64 representation prototypes by a synthetic, image-only anchor.
    anchor=SyntheticDirectionOrderer('synthetic_direction',args.seed).fit(encoded[0,train],encoded[3,train],encoded[4,train],encoded[5,train],encoded[1,train],encoded[2,train])
    probabilities=[]
    for z in projected:
        logits=z@prototype_weights.T/.1;logits-=logits.max(1,keepdims=True)
        q=np.exp(logits);probabilities.append(q/q.sum(1,keepdims=True))
    fit_q=probabilities[0][train]
    anchor_means=(fit_q.T@anchor.raw_score(encoded[0,train]))/np.maximum(fit_q.sum(0),1e-12)
    order=np.argsort(np.argsort(anchor_means));raw_fit=fit_q@order
    sorted_scores=np.sort(raw_fit);cuts=np.quantile(raw_fit,[.25,.5,.75])
    for gi,gamma in enumerate([1.,.8,1.2]):
        raw=probabilities[gi][test]@order
        ranks=(np.searchsorted(sorted_scores,raw,'left')+np.searchsorted(sorted_scores,raw,'right'))/(2*len(sorted_scores))
        levels=np.searchsorted(cuts,raw,'right')
        for uid,rank,level in zip(frame.loc[test,'uid'],ranks,levels):
            predictions.append(dict(uid=uid,scope='main',fold=args.fold,method='swav_ordered_prototypes',gamma=gamma,score=float(rank),level=int(level)))
    singular=np.linalg.svd(projected[0,train]-projected[0,train].mean(0),compute_uv=False)
    variance=singular**2;variance/=max(variance.sum(),1e-15)
    positive=variance[variance>0]
    diagnostics['representation']=dict(encoder_std_mean=float(encoded[0,train].std(0).mean()),
        projection_std_mean=float(projected[0,train].std(0).mean()),
        projection_effective_covariance_rank=float(np.exp(-(positive*np.log(positive)).sum())),
        prototype_hard_counts=np.bincount(fit_q.argmax(1),minlength=64).tolist(),
        prototype_soft_mass=fit_q.sum(0).tolist(),prototype_order=order.tolist(),
        sources_have_manual_grades=False)
    np.savez(run/'prototype_order.npz',order=order,sorted_scores=sorted_scores,cutpoints=cuts,anchor_means=anchor_means)
    pd.DataFrame(predictions).to_csv(run/'predictions_frozen_before_evaluation.csv',index=False)
    pd.DataFrame(responses).to_csv(run/'synthetic_response.csv',index=False)
    atomic_json(run/'diagnostics.json',diagnostics)
    atomic_json(run/'complete.json',dict(seed=args.seed,fold=args.fold,epochs=len(trace),source_sha256=sources,
        masks_read=False,manual_grading_labels_read=False,elapsed_seconds=time.monotonic()-elapsed_start,
        optimizer_updates=sum(t['successful_updates'] for t in trace),amp_skips=sum(t['amp_skips'] for t in trace),
        peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
        predictions_sha256=hashlib.sha256((run/'predictions_frozen_before_evaluation.csv').read_bytes()).hexdigest()))


if __name__=='__main__':main()

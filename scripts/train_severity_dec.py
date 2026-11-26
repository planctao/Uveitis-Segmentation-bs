"""Matched-budget reconstruction and DEC on frozen DINO training features."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,normalize
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ssl import DECEncoder,dec_target,student_assignments
from bs.severity_synthetic import SyntheticDirectionOrderer


def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def tensor_outputs(model,array):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(array),256):out.append(model.encoder(torch.from_numpy(array[start:start+256]).float()).numpy())
    return np.concatenate(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',required=True)
    p.add_argument('--probe',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    root=ROOT/'outputs/severity_research_20261013'
    run=root/'domain_ssl/dec'/f'seed{args.seed}'/args.fold;run.mkdir(parents=True,exist_ok=True)
    if args.probe:
        run=root/'domain_ssl/probes'/f'dec_seed{args.seed}_{args.fold}';run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists() and not args.probe:raise RuntimeError('Completed task exists; refusing to overwrite')
    frame=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str})
    original=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    synthetic=np.load(root/'embeddings/frozen_dino_synthetic.npz',allow_pickle=False)
    assert np.array_equal(frame.uid.to_numpy(),original['uids']) and np.array_equal(original['uids'],synthetic['uids'])
    train=(frame.fold!=args.fold)&~frame.exclude_exact_group;test=(frame.fold==args.fold)&~frame.exclude_exact_group
    assert not set(frame.loc[train,'pixel_sha256'])&set(frame.loc[test,'pixel_sha256'])
    seed=args.seed+1000*int(args.fold[1:]);torch.manual_seed(seed)
    scaler=StandardScaler().fit(normalize(original['embeddings'][0,train]))
    inputs=np.stack([scaler.transform(normalize(z)).astype(np.float32) for z in np.concatenate([original['embeddings'],synthetic['embeddings']])])
    x=torch.from_numpy(inputs[0,train]);n=len(x);epochs=2 if args.probe else 200
    source_paths=[Path(__file__),ROOT/'src/bs/severity_ssl.py',ROOT/'src/bs/severity_synthetic.py',ROOT/'docs/severity_domain_ssl_protocol_20261013.md']
    sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in source_paths}
    atomic_json(run/'plan.json',dict(seed=args.seed,actual_seed=seed,fold=args.fold,probe=args.probe,
        ae_pretrain_epochs=epochs,branch_epochs=epochs,batch_size=128,learning_rate=.001,
        train_uids=frame.loc[train,'uid'].tolist(),test_uids=frame.loc[test,'uid'].tolist(),
        masks_read=False,manual_grading_labels_read=False,sources_sha256=sources))
    model=DECEncoder();trace=[];start=time.monotonic()

    def train_reconstruction(network,phase,seed_offset):
        optimizer=torch.optim.Adam(network.parameters(),lr=.001)
        for epoch in range(epochs):
            network.train();indices=torch.randperm(n,generator=torch.Generator().manual_seed(seed+seed_offset+epoch));total=0.
            for idx in indices.split(128):
                optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(network(x[idx]),x[idx])
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {phase}')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
            trace.append(dict(phase=phase,epoch=epoch+1,loss=total/n,optimizer_updates=(n+127)//128))
            if (epoch+1)%25==0 or epoch==0:
                pd.DataFrame(trace).to_csv(run/'training_trace.csv',index=False)
                atomic_json(run/'progress.json',dict(phase=phase,epoch=epoch+1,epochs=epochs,elapsed_seconds=time.monotonic()-start))
                print(f'{phase} epoch {epoch+1}/{epochs} loss={total/n:.6f} elapsed={time.monotonic()-start:.1f}s',flush=True)
        return network

    model=train_reconstruction(model,'pretrain',0)
    initial=copy.deepcopy(model.state_dict())
    torch.save(dict(model=initial,sources=sources),run/'shared_branch_initial.pt')
    control=copy.deepcopy(model)
    assert all(torch.equal(initial[k],control.state_dict()[k]) for k in initial)
    control=train_reconstruction(control,'ae_control',10000)
    dec=copy.deepcopy(model)
    assert all(torch.equal(initial[k],dec.state_dict()[k]) for k in initial)
    latent=tensor_outputs(dec,inputs[0,train]);kmeans=KMeans(4,n_init=20,random_state=seed).fit(latent)
    centers=torch.nn.Parameter(torch.from_numpy(kmeans.cluster_centers_).float())
    optimizer=torch.optim.Adam([*dec.encoder.parameters(),centers],lr=.001)
    previous=kmeans.labels_
    for epoch in range(epochs):
        dec.train()
        with torch.no_grad():
            q=student_assignments(dec.encoder(x),centers);target=dec_target(q)
            labels=q.argmax(1).numpy();changed=float(np.mean(labels!=previous));previous=labels
        total=0.;indices=torch.randperm(n,generator=torch.Generator().manual_seed(seed+10000+epoch))
        for idx in indices.split(128):
            optimizer.zero_grad(set_to_none=True);q=student_assignments(dec.encoder(x[idx]),centers)
            loss=F.kl_div(q.clamp_min(1e-12).log(),target[idx],reduction='batchmean')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DEC KL')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        trace.append(dict(phase='dec',epoch=epoch+1,loss=total/n,assignment_change_fraction=changed,
                          optimizer_updates=(n+127)//128))
        if (epoch+1)%25==0 or epoch==0:
            pd.DataFrame(trace).to_csv(run/'training_trace.csv',index=False)
            atomic_json(run/'progress.json',dict(phase='dec',epoch=epoch+1,epochs=epochs,elapsed_seconds=time.monotonic()-start))
            print(f'dec epoch {epoch+1}/{epochs} KL={total/n:.6f} changes={changed:.4f} elapsed={time.monotonic()-start:.1f}s',flush=True)
    if args.probe:
        atomic_json(run/'probe.json',dict(elapsed_seconds=time.monotonic()-start,finite_losses=True,n_train=n,
                                        model_parameters=sum(v.numel() for v in model.parameters())))
        return
    predictions=[];responses=[];diagnostics={}
    for branch,network in [('ae',control),('dec',dec)]:
        encoded=np.stack([tensor_outputs(network,a) for a in inputs])
        branch_centers=centers.detach().numpy() if branch=='dec' else KMeans(4,n_init=20,random_state=seed).fit(encoded[0,train]).cluster_centers_
        torch.save(dict(model=network.state_dict(),centers=branch_centers,scaler=scaler,sources=sources),run/f'{branch}_final.pt')
        np.savez_compressed(run/f'{branch}_representations.npz',uids=frame.uid.to_numpy(dtype=str),conditions=np.array(['original','gamma08','gamma12','localized008','localized016','meanmatched016']),features=encoded)
        for method in ['synthetic_direction','gamma_orthogonal_direction']:
            score_model=SyntheticDirectionOrderer(method,args.seed).fit(encoded[0,train],encoded[3,train],encoded[4,train],encoded[5,train],encoded[1,train],encoded[2,train])
            name=f'{branch}_{method}'
            joblib.dump(score_model,run/f'{name}.joblib',compress=3);diagnostics[name]=score_model.diagnostics
            for gi,gamma in enumerate([1.,.8,1.2]):
                scores,levels=score_model.predict(encoded[gi,test])
                for uid,score,level in zip(frame.loc[test,'uid'],scores,levels):
                    predictions.append(dict(uid=uid,scope='main',fold=args.fold,method=name,gamma=gamma,score=float(score),level=int(level)))
            raw=score_model.raw_score(encoded[0,test]);low=score_model.raw_score(encoded[3,test]);high=score_model.raw_score(encoded[4,test])
            responses.append(dict(method=name,fold=args.fold,n=int(test.sum()),monotonic_fraction=float(((low>raw)&(high>low)).mean())))
        direction_model=SyntheticDirectionOrderer('synthetic_direction',args.seed).fit(encoded[0,train],encoded[3,train],encoded[4,train],encoded[5,train],encoded[1,train],encoded[2,train])
        order=np.argsort(np.argsort(direction_model.raw_score(branch_centers)))
        qtrain=student_assignments(torch.from_numpy(encoded[0,train]),torch.from_numpy(branch_centers)).numpy()
        sorted_scores=np.sort(qtrain@order)
        diagnostics[f'{branch}_clusters']=dict(counts=np.bincount(qtrain.argmax(1),minlength=4).tolist(),
            latent_std_mean=float(encoded[0,train].std(0).mean()),cluster_order=order.tolist(),
            assignment_entropy=float(-(qtrain*np.log(np.maximum(qtrain,1e-12))).sum(1).mean()))
        for gi,gamma in enumerate([1.,.8,1.2]):
            q=student_assignments(torch.from_numpy(encoded[gi,test]),torch.from_numpy(branch_centers)).numpy();raw=q@order
            ranks=(np.searchsorted(sorted_scores,raw,'left')+np.searchsorted(sorted_scores,raw,'right'))/(2*len(sorted_scores))
            for uid,rank,level in zip(frame.loc[test,'uid'],ranks,order[q.argmax(1)]):
                predictions.append(dict(uid=uid,scope='main',fold=args.fold,method=f'{branch}_ordered_clusters',gamma=gamma,score=float(rank),level=int(level)))
        np.savez(run/f'{branch}_cluster_order.npz',centers=branch_centers,order=order,sorted_scores=sorted_scores)
    pd.DataFrame(predictions).to_csv(run/'predictions_frozen_before_evaluation.csv',index=False)
    pd.DataFrame(responses).to_csv(run/'synthetic_response.csv',index=False)
    pd.DataFrame(trace).to_csv(run/'training_trace.csv',index=False)
    atomic_json(run/'diagnostics.json',diagnostics)
    atomic_json(run/'complete.json',dict(seed=args.seed,fold=args.fold,elapsed_seconds=time.monotonic()-start,
        source_sha256=sources,masks_read=False,manual_grading_labels_read=False,shared_initial_tensors_match=True,
        optimizer_updates=sum(t['optimizer_updates'] for t in trace),phase_epochs={k:int(v) for k,v in pd.DataFrame(trace).groupby('phase').size().items()},
        predictions_sha256=hashlib.sha256((run/'predictions_frozen_before_evaluation.csv').read_bytes()).hexdigest()))
    print('completed',args.seed,args.fold,time.monotonic()-start,flush=True)


if __name__=='__main__':main()

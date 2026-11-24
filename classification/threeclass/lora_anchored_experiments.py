"""Fit-only teacher anchored DINO adaptation with an actual16image contrastive batch."""
import fcntl
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from experiment import ROOT,load_data,loss_fn,sample_weights,metrics,dump,summarize_group
from extract_vit import WEIGHTS
from anchored_lora import AnchoredMultiScaleLoRA as MultiScaleLoRA,construct_anchored,fit_only_teacher,anchor_terms
from lora_seed import adapter_hash
from amp_optimizer import optimizer_update

OUT=ROOT/'runs/lora_anchored'


class Pairs(Dataset):
    def __init__(self,frame,diseases):
        self.frame=frame.reset_index(drop=True);self.weights=sample_weights(self.frame,self.frame.label)
        self.disease=self.frame.disease.map({d:i for i,d in enumerate(diseases)}).fillna(-1).astype(int).to_numpy()
        self.patient=pd.factorize(self.frame.patient_group)[0]
        self.transforms=[T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]) for size in [320,448]]
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.cache_path) as source:
            im=source.convert('RGB');a,b=[t(im) for t in self.transforms]
        return a,b,int(row.label),int(self.disease[i]),float(self.weights[i]),int(self.patient[i]),i


@torch.inference_mode()
def predict(model,loader):
    model.eval();values=[]
    for a,b,*_ in loader:
        with torch.autocast('cuda',dtype=torch.float16):p=model(a.cuda(non_blocking=True),b.cuda(non_blocking=True))['p']
        values.append(p.float().cpu().numpy())
    return np.concatenate(values)


def audit_source(checkpoint,frame,features,fit,stop):
    tf=frame.loc[fit];vf=frame.loc[stop]
    assert not set(tf.patient_group).intersection(vf.patient_group)
    assert set(checkpoint['diseases'])==set(tf.disease)
    disease_map=tf.groupby('disease').label.first()
    assert checkpoint['dims']['disease_to_class']==[int(disease_map[d]) for d in checkpoint['diseases']]
    for name,values in features.items():
        fitted=values[fit];axes=tuple(range(fitted.ndim-1))
        assert np.allclose(checkpoint['stats'][name]['mean'],fitted.mean(axes),atol=1e-6,rtol=1e-6),name
        assert np.allclose(checkpoint['stats'][name]['std'],np.maximum(fitted.std(axes),.05),atol=1e-6,rtol=1e-6),name


def train(source,frame,features,digest,fit,stop,cfg,seed,destination):
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'summary.json').exists():return json.loads((destination/'summary.json').read_text())
    checkpoint=torch.load(source,map_location='cpu',weights_only=False);assert checkpoint['manifest_sha256']==digest
    audit_source(checkpoint,frame,features,fit,stop)
    teacher_p,teacher_features=fit_only_teacher(checkpoint,features,fit)
    np.savez_compressed(destination/'teacher_predictions.npz',fit_indices=np.flatnonzero(fit),p=teacher_p)
    teacher_probability=torch.from_numpy(teacher_p).cuda();teacher_global=torch.from_numpy(teacher_features).cuda()
    assert not teacher_probability.requires_grad and not teacher_global.requires_grad
    rng=20261124+seed;model=construct_anchored(checkpoint,cfg,rng).cuda()
    initial_adapter=model.delta();initial_adapter_sha=adapter_hash(initial_adapter)
    assert all(not value.count_nonzero() for name,value in initial_adapter.items() if 'lora_b' in name)
    if cfg['rank']:
        verified=json.loads((ROOT/'verification_lora_seed.json').read_text())['expected_initializations']
        expected=next(row for row in verified if row['rank']==cfg['rank'] and row['blocks']==cfg['blocks'] and row['seed']==rng)
        assert initial_adapter_sha==expected['initial_adapter_sha256'],'Adapter initialization changed from independently verified seed'
    torch.save(initial_adapter,destination/'initial_adapter.pt')
    dump(destination/'initialization.json',{'seed':rng,'initial_adapter_sha256':initial_adapter_sha,'policy':'Seed before construction and reset training streams afterward; independent of resume and arm order.'})
    tf=frame.loc[fit];vf=frame.loc[stop];generator=torch.Generator().manual_seed(rng)
    training=DataLoader(Pairs(tf,checkpoint['diseases']),batch_size=16,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=generator)
    validation=DataLoader(Pairs(vf,checkpoint['diseases']),batch_size=12,num_workers=4,pin_memory=True,persistent_workers=True)
    groups=[{'params':model.head.parameters(),'lr':.0001}]
    adapter_parameters=[p for p in model.backbone.parameters() if p.requires_grad]
    if adapter_parameters:groups.append({'params':adapter_parameters,'lr':cfg['adapter_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=6*len(training))
    history=[];best=None;waiting=0;started=time.time();initial=None;successful_updates=0;skipped_updates=0
    for epoch in range(7):
        total_loss=0.;supervised_sum=0.;kl_sum=0.;feature_sum=0.;optimizer.zero_grad(set_to_none=True)
        if epoch:
            model.train()
            for batch,(a,b,y,disease,w,patient,local_index) in enumerate(training):
                a,b=a.cuda(non_blocking=True),b.cuda(non_blocking=True);y,disease,w,patient=y.cuda(),disease.cuda(),w.float().cuda(),patient.cuda()
                with torch.autocast('cuda',dtype=torch.float16):out=model(a,b)
                out={k:v.float() for k,v in out.items()};supervised=loss_fn(out,y,disease,{'leaf':w,'patient_ids':patient},checkpoint['cfg'])
                local_index=local_index.cuda();kl,feature=anchor_terms(out,teacher_probability[local_index],teacher_global[local_index],w)
                loss=supervised+cfg['anchor_kl']*kl+cfg['anchor_feature']*feature
                supervised_sum+=supervised.item()*len(y);kl_sum+=kl.item()*len(y);feature_sum+=feature.item()*len(y)
                assert torch.isfinite(loss),'Nonfinite adaptation loss'
                scaler.scale(loss).backward();total_loss+=loss.item()*len(y)
                updated,norm=optimizer_update(optimizer,scaler,scheduler,model.parameters())
                successful_updates+=int(updated);skipped_updates+=int(not updated)
                if batch%20==0:
                    progress={'status':'training','pid':os.getpid(),'group':str(destination.parent),'seed':seed,'epoch':epoch,'batch':batch+1,'total_batches':len(training),'seconds':time.time()-started}
                    dump(OUT/'state.json',progress);print(json.dumps(progress),flush=True)
        p=predict(model,validation);m=metrics(vf.label,p);key=(m['macro_f1'],-m['log_loss'])
        if epoch==0:
            initial=m;np.savez_compressed(destination/'initial_predictions.npz',p=p)
            cached=np.load(Path(source).parent/'val_predictions.npz')['p'];assert cached.shape==p.shape
            cache_difference=float(np.abs(cached-p).mean());assert cache_difference<.001,'Image initialization differs from frozen feature source'
        history.append({'epoch':epoch,'train_loss':total_loss/len(tf),'train_supervised':supervised_sum/len(tf),'train_teacher_kl':kl_sum/len(tf),'train_feature_mse':feature_sum/len(tf),'val':m,'seconds':time.time()-started,'successful_optimizer_updates':successful_updates,'amp_skipped_updates':skipped_updates,'scheduler_epoch':scheduler.last_epoch})
        print(json.dumps({'event':'lora_epoch','path':str(destination),'epoch':epoch,'val':m,'seconds':time.time()-started}),flush=True)
        dump(destination/'history.json',history)
        if best is None or key>best:
            best=key;waiting=0
            package={**checkpoint,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'lora_state_dict':model.delta(),'lora_config':cfg,'backbone_adapted':True,'adaptation_method':'low_rank_qkv_weight_update','epoch':epoch,'val':m,'initial_live_val':initial,'initial_cached_probability_difference':cache_difference,'source_checkpoint':str(source),'source_checkpoint_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'backbone_sha256':hashlib.sha256(Path(WEIGHTS).read_bytes()).hexdigest(),'fit_patient_groups':sorted(set(tf.patient_group)),'stop_patient_groups':sorted(set(vf.patient_group)),'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist()}
            package.update(algorithm_version=4,actual_loss_batch=16,gradient_accumulation=1,initialization_seed=rng,initial_adapter_sha256=initial_adapter_sha,successful_optimizer_updates=successful_updates,amp_skipped_updates=skipped_updates,scheduler_epoch=scheduler.last_epoch)
            torch.save(package,destination/'model.pt');np.savez_compressed(destination/'val_predictions.npz',p=p,g=np.zeros((len(p),0),dtype='float32'))
        else:waiting+=1
        if waiting>=3:break
    summary={'cfg':checkpoint['cfg'],'lora_config':cfg,'epoch':package['epoch'],'val':package['val'],'initial_live_val':initial,'initial_cached_probability_difference':cache_difference,'manifest_sha256':digest,'seconds':time.time()-started,'trainable_adapter_parameters':sum(p.numel() for p in adapter_parameters),'fit_patients':tf.patient_group.nunique(),'stop_patients':vf.patient_group.nunique(),'all_feature_source_audit_passed':True}
    summary.update(algorithm_version=4,actual_loss_batch=16,gradient_accumulation=1,initialization_seed=rng,initial_adapter_sha256=initial_adapter_sha,successful_optimizer_updates=successful_updates,amp_skipped_updates=skipped_updates,scheduler_epoch=scheduler.last_epoch)
    assert scheduler.last_epoch==successful_updates
    dump(destination/'summary.json',summary);del model,training,validation,teacher_probability,teacher_global;torch.cuda.empty_cache();return summary


def crossvalidate(candidate,source_cv,frame,features,digest):
    indices=np.flatnonzero(frame.split.eq('train').to_numpy());tf=frame.iloc[indices]
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261125);folds=[]
    out=OUT/'crossvalidation'/candidate['candidate_id']
    for fold,(fit_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
        target=out/f'fold_{fold}';report_path=target/'result.json'
        if report_path.exists():folds.append(json.loads(report_path.read_text()));continue
        outer=indices[outer_local];fit_frame=frame.iloc[indices[fit_local]];groups=fit_frame.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
        train_groups,stop_groups=train_test_split(groups.index,test_size=.2,random_state=20261124+fold,stratify=groups)
        fit=frame.index.isin(fit_frame[fit_frame.split_group.isin(train_groups)].index);stop=frame.index.isin(fit_frame[fit_frame.split_group.isin(stop_groups)].index)
        assert not set(frame.loc[fit|stop,'patient_group']).intersection(frame.iloc[outer].patient_group)
        probabilities=[]
        for seed in range(3):
            source=source_cv/f'fold_{fold}'/f'seed_{seed}'/'model.pt';seed_dir=target/f'seed_{seed}'
            train(source,frame,features,digest,fit,stop,candidate['lora_config'],seed,seed_dir)
            checkpoint=torch.load(seed_dir/'model.pt',map_location='cpu',weights_only=False)
            assert set(checkpoint['fit_patient_groups'])==set(frame.loc[fit,'patient_group']) and set(checkpoint['stop_patient_groups'])==set(frame.loc[stop,'patient_group'])
            model=MultiScaleLoRA(checkpoint,checkpoint['lora_config']).eval().cuda()
            loader=DataLoader(Pairs(frame.iloc[outer],checkpoint['diseases']),batch_size=12,num_workers=4,pin_memory=True)
            probabilities.append(predict(model,loader));del model,loader;torch.cuda.empty_cache()
        p=np.mean(probabilities,axis=0);report={'fold':fold,'metrics':metrics(frame.iloc[outer].label,p),'fit_images':int(fit.sum()),'stop_images':int(stop.sum()),'outer_images':len(outer),'source_and_patient_audit_passed':True}
        np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p,pred=p.argmax(1));dump(report_path,report);folds.append(report)
    summary={'candidate':candidate['candidate_id'],'lora_config':candidate['lora_config'],'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'external_test_evaluated':False}
    dump(out/'summary.json',summary);return summary


def run():
    for name in ['verification_lora.json','verification_amp_optimizer.json','verification_lora_seed.json','verification_anchored_lora.json']:
        assert json.loads((ROOT/name).read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    dump(OUT/'implementation.json',{'algorithm_version':4,'scheduler_policy':'Advance only after successful AMP update.','initialization_policy':'Seed before construction and reset after; compare saved actual initial adapters to independent hashes.','code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['lora_anchored_experiments.py','anchored_lora.py','lora_model.py','amp_optimizer.py','lora_seed.py']}})
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    fit=frame.split.eq('train').to_numpy();stop=frame.split.eq('val').to_numpy();records=[]
    for index,cfg in enumerate(protocol['configuration_plan']):
        group=OUT/f'{index:02d}_{cfg["name"]}'
        for seed,source in enumerate(reference['selected']['checkpoints']):
            train(source,frame,features,digest,fit,stop,cfg,seed,group/f'seed_{seed}')
        record=summarize_group(group,frame)
        initial=np.mean([np.load(group/f'seed_{seed}'/'initial_predictions.npz')['p'] for seed in range(3)],axis=0)
        record.update(candidate_id=group.name,lora_config=cfg,initial_live_val=metrics(frame.loc[stop,'label'],initial))
        records.append(record);dump(OUT/'leaderboard.json',records)
    eligible=[record for record in records if record['lora_config']['rank'] and record['val']['macro_f1']>record['initial_live_val']['macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']]
    if eligible:
        winner=max(eligible,key=lambda row:row['val']['macro_f1'])
        dump(OUT/'selection_before_cv.json',{'winner':winner,'reference_candidate':reference['selected']['candidate_id'],'external_test_evaluated':False})
        source_cv=Path(reference['selected']['source_cv_summary']).parent
        for record in [records[0],winner]:crossvalidate(record,source_cv,frame,features,digest)
    dump(OUT/'state.json',{'status':'completed_development_experiments','pid':os.getpid(),'qualifying_adapters':len(eligible),'external_test_evaluated':False})

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'algorithm_version':4,'error':traceback.format_exc(),'external_test_evaluated':False});raise


if __name__=='__main__':main()

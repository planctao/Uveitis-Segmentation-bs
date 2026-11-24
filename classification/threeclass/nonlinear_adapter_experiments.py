"""Real-image final-block nonlinear adaptation with fixed teacher and matched MLP control."""
import fcntl
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment import ROOT,load_data,loss_fn,metrics,dump,summarize_group
from extract_vit import WEIGHTS
from nonlinear_backbone_adapter import MultiScaleNonlinearAdapter,construct_nonlinear
from norm_adaptation_experiments import Pairs,predict,audit_source
from anchored_lora import fit_only_teacher,anchor_terms
from lora_seed import adapter_hash
from amp_optimizer import optimizer_update
from kernel_residual_experiments import phase_plans,sha
from kernel_residual_experiments import collect_cpu
from verify_nonlinear_backbone_adapter import independent_adapter

OUT=ROOT/'runs/nonlinear_backbone_adapter'


def train(source,frame,features,digest,fit,stop,cfg,seed,destination):
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'summary.json').exists():return json.loads((destination/'summary.json').read_text())
    checkpoint=torch.load(source,map_location='cpu',weights_only=False);assert checkpoint['manifest_sha256']==digest
    assert checkpoint['cfg']['seed']==seed
    audit_source(checkpoint,frame,features,fit,stop)
    teacher_p,teacher_features=fit_only_teacher(checkpoint,features,fit)
    np.savez_compressed(destination/'teacher_predictions.npz',fit_indices=np.flatnonzero(fit),p=teacher_p)
    teacher_probability=torch.from_numpy(teacher_p).cuda();teacher_global=torch.from_numpy(teacher_features).cuda()
    assert not teacher_probability.requires_grad and not teacher_global.requires_grad
    rng=20261124+seed;model=construct_nonlinear(checkpoint,cfg,rng).cuda()
    initial_adapter=model.delta();initial_adapter_sha=adapter_hash(initial_adapter)
    verified=json.loads((ROOT/'verification_nonlinear_backbone_adapter.json').read_text())
    expected=verified['initialization_hashes'][f'blocks{cfg["blocks"]}_rank{cfg["rank"]}_seed{seed}']
    assert initial_adapter_sha==expected
    torch.save(initial_adapter,destination/'initial_adapter.pt')
    dump(destination/'initialization.json',{'seed':rng,'initial_adapter_sha256':initial_adapter_sha,'policy':'Seed before construction, verify all seeded zero-output token adapter tensors, then reset training stream; independent of resume/arm order.'})
    tf=frame.loc[fit];vf=frame.loc[stop];generator=torch.Generator().manual_seed(rng)
    training=DataLoader(Pairs(tf,checkpoint['diseases']),batch_size=16,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=generator)
    validation=DataLoader(Pairs(vf,checkpoint['diseases']),batch_size=12,num_workers=4,pin_memory=True,persistent_workers=True)
    groups=[{'params':model.head.parameters(),'lr':.00001,'weight_decay':.01}]
    adapter_parameters=[p for p in model.backbone.parameters() if p.requires_grad]
    if adapter_parameters:groups.append({'params':adapter_parameters,'lr':cfg['adapter_lr'],'weight_decay':.01})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
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
        print(json.dumps({'event':'nonlinear_epoch','path':str(destination),'epoch':epoch,'val':m,'seconds':time.time()-started}),flush=True)
        dump(destination/'history.json',history)
        if best is None or key>best:
            best=key;waiting=0
            package={**checkpoint,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'nonlinear_state_dict':model.delta(),'nonlinear_config':cfg,'backbone_adapted':bool(adapter_parameters),'adaptation_method':'nonlinear_output_bottlenecks','epoch':epoch,'val':m,'initial_live_val':initial,'initial_cached_probability_difference':cache_difference,'source_checkpoint':str(source),'source_checkpoint_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'backbone_sha256':hashlib.sha256(Path(WEIGHTS).read_bytes()).hexdigest(),'fit_patient_groups':sorted(set(tf.patient_group)),'stop_patient_groups':sorted(set(vf.patient_group)),'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist()}
            package.update(algorithm_version=1,actual_loss_batch=16,gradient_accumulation=1,initialization_seed=rng,initial_adapter_sha256=initial_adapter_sha,successful_optimizer_updates=successful_updates,amp_skipped_updates=skipped_updates,scheduler_epoch=scheduler.last_epoch)
            torch.save(package,destination/'model.pt');np.savez_compressed(destination/'val_predictions.npz',p=p,g=np.zeros((len(p),0),dtype='float32'))
        else:waiting+=1
        if waiting>=3:break
    summary={'cfg':checkpoint['cfg'],'nonlinear_config':cfg,'epoch':package['epoch'],'val':package['val'],'initial_live_val':initial,'initial_cached_probability_difference':cache_difference,'manifest_sha256':digest,'seconds':time.time()-started,'trainable_adapter_parameters':sum(p.numel() for p in adapter_parameters),'fit_patients':tf.patient_group.nunique(),'stop_patients':vf.patient_group.nunique(),'all_feature_source_audit_passed':True}
    summary.update(algorithm_version=1,actual_loss_batch=16,gradient_accumulation=1,initialization_seed=rng,initial_adapter_sha256=initial_adapter_sha,successful_optimizer_updates=successful_updates,amp_skipped_updates=skipped_updates,scheduler_epoch=scheduler.last_epoch)
    assert scheduler.last_epoch==successful_updates
    dump(destination/'summary.json',summary);del model,training,validation,teacher_probability,teacher_global;torch.cuda.empty_cache();return summary


def crossvalidate(record,reference,frame,features,digest):
    folds=[]
    for phase,fit_ids,stop_ids,outer,sources in phase_plans(frame,reference):
        if phase=='validation':continue
        fit=np.zeros(len(frame),dtype=bool);stop=fit.copy();fit[fit_ids]=True;stop[stop_ids]=True
        target=OUT/'crossvalidation'/record['candidate_id']/phase;ps=[]
        for seed,source in enumerate(sources):
            folder=target/f'seed_{seed}'
            train(source,frame,features,digest,fit,stop,record['nonlinear_config'],seed,folder)
            c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            model=MultiScaleNonlinearAdapter(c,c['nonlinear_config']).eval().cuda()
            loader=DataLoader(Pairs(frame.iloc[outer],c['diseases']),batch_size=12,num_workers=4,pin_memory=True)
            ps.append(predict(model,loader));del model,loader;torch.cuda.empty_cache()
        p=np.mean(ps,0);np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p,pred=p.argmax(1))
        row={'phase':phase,'metrics':metrics(frame.iloc[outer].label,p)};dump(target/'metrics.json',row);folds.append(row)
    result={'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
        'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))}
    dump(OUT/'crossvalidation'/record['candidate_id']/'summary.json',result);record['cv']=result


@torch.inference_mode()
def audit_fit(folder,cfg,seed,source,frame,features,digest,fit,stop):
    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    original=torch.load(source,map_location='cpu',weights_only=False);audit_source(original,frame,features,fit,stop)
    assert c['manifest_sha256']==digest and c['cfg']==original['cfg'] and c['nonlinear_config']==cfg
    assert c['source_checkpoint_sha256']==sha(source) and c['source_checkpoint']==str(source)
    assert c['backbone_sha256']==sha(WEIGHTS)
    assert np.array_equal(c['fit_indices'],np.flatnonzero(fit)) and np.array_equal(c['stop_indices'],np.flatnonzero(stop))
    assert c['fit_patient_groups']==sorted(set(frame.loc[fit,'patient_group']))
    assert c['stop_patient_groups']==sorted(set(frame.loc[stop,'patient_group']))
    for key in ['patient_group','split_group']:assert not set(frame.loc[fit,key])&set(frame.loc[stop,key])
    for key,value in original['stats'].items():
        for name in ['mean','std']:assert np.array_equal(value[name],c['stats'][key][name])
    assert c['dims']==original['dims'] and c['diseases']==original['diseases']
    initial=torch.load(folder/'initial_adapter.pt',map_location='cpu',weights_only=True)
    expected=json.loads((ROOT/'verification_nonlinear_backbone_adapter.json').read_text())['initialization_hashes'][f'blocks{cfg["blocks"]}_rank{cfg["rank"]}_seed{seed}']
    assert adapter_hash(initial)==c['initial_adapter_sha256']==expected
    assert set(initial)==set(c['nonlinear_state_dict'])
    assert all(v.count_nonzero()==0 for key,v in initial.items() if '.up.' in key)
    for key in c['nonlinear_state_dict']:assert key.startswith('token_adapters.')
    history=json.loads((folder/'history.json').read_text());best=max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))
    assert best['epoch']==c['epoch']
    for row in history:
        assert row['scheduler_epoch']==row['successful_optimizer_updates']
        assert row['successful_optimizer_updates']+row['amp_skipped_updates']==row['epoch']*int(np.ceil(fit.sum()/16))
        reconstructed=row['train_supervised']+cfg['anchor_kl']*row['train_teacher_kl']+cfg['anchor_feature']*row['train_feature_mse']
        assert abs(reconstructed-row['train_loss'])<1e-5
    assert c['scheduler_epoch']==best['scheduler_epoch'] and c['successful_optimizer_updates']==best['successful_optimizer_updates']
    teacher=np.load(folder/'teacher_predictions.npz');indices=np.flatnonzero(fit)
    assert np.array_equal(teacher['fit_indices'],indices)
    cpu_teacher=collect_cpu(original,features,indices)[1]
    assert np.allclose(cpu_teacher,teacher['p'],atol=5e-5,rtol=5e-5)
    model=MultiScaleNonlinearAdapter(c,cfg).eval().cuda()
    expected_keys={key for key,p in model.backbone.named_parameters() if p.requires_grad}
    assert expected_keys==set(initial)
    loader=DataLoader(Pairs(frame.loc[stop],c['diseases']),batch_size=12,num_workers=4,pin_memory=True)
    p=predict(model,loader);stored=np.load(folder/'val_predictions.npz')['p']
    assert np.allclose(p,stored,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),stored.argmax(1))
    restored=metrics(frame.loc[stop,'label'],p)
    for name in ['accuracy','macro_f1','log_loss']:assert abs(restored[name]-c['val'][name])<1e-7
    actual=[];handles=[]
    for module in model.backbone.token_adapters:
        handles.append(module.register_forward_hook(lambda module,args,out:actual.append((module,args[0].float().cpu().numpy(),out.float().cpu().numpy()))))
    dataset=Pairs(frame.loc[stop].iloc[:2],c['diseases']);a=torch.stack([dataset[i][0] for i in range(2)]).cuda();b=torch.stack([dataset[i][1] for i in range(2)]).cuda()
    model(a,b)
    for handle in handles:handle.remove()
    errors=[]
    assert len(actual)==cfg['blocks']*2
    for module,tokens,delta in actual:
        independent=independent_adapter(module,tokens);errors.append(float(np.max(np.abs(independent-delta))))
        assert np.allclose(independent,delta,atol=5e-5,rtol=5e-5)
    result={'checkpoint_sha256':sha(folder/'model.pt'),'seed':seed,'epoch':c['epoch'],
        'scope_keys':sorted(initial),'maximum_independent_token_adapter_error':max(errors,default=0.),
        'teacher_cpu_restored':True,'stop_images_restored':len(p),'source_initialization_verified':True}
    del model,loader;torch.cuda.empty_cache();return result,p


def audit(frame,features,digest,reference,records,winner):
    checked=[]
    for phase,fit_ids,stop_ids,outer,sources in phase_plans(frame,reference):
        selected=records if phase=='validation' else ([] if winner is None else [records[0],winner])
        if not selected:continue
        fit=np.zeros(len(frame),dtype=bool);stop=fit.copy();fit[fit_ids]=True;stop[stop_ids]=True
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[ids][key]) for ids in [fit_ids,stop_ids,outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        for record in selected:
            ps=[]
            for seed,source in enumerate(sources):
                folder=(OUT/record['candidate_id'] if phase=='validation' else OUT/'crossvalidation'/record['candidate_id']/phase)/f'seed_{seed}'
                detail,p=audit_fit(folder,record['nonlinear_config'],seed,source,frame,features,digest,fit,stop)
                checked.append({'candidate_id':record['candidate_id'],'phase':phase,**detail})
                if phase!='validation':
                    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=MultiScaleNonlinearAdapter(c,c['nonlinear_config']).eval().cuda()
                    loader=DataLoader(Pairs(frame.iloc[outer],c['diseases']),batch_size=12,num_workers=4,pin_memory=True)
                    p=predict(model,loader);del model,loader;torch.cuda.empty_cache()
                ps.append(p)
            averaged=np.mean(ps,0)
            if phase=='validation':
                saved=np.load(OUT/record['candidate_id']/'val_ensemble.npz')['p'];target=record['val'];ids=stop_ids
            else:
                saved=np.load(OUT/'crossvalidation'/record['candidate_id']/phase/'outer_predictions.npz')['p']
                target=next(r['metrics'] for r in record['cv']['folds'] if r['phase']==phase);ids=outer
            assert np.allclose(averaged,saved,atol=5e-5,rtol=5e-5) and np.array_equal(averaged.argmax(1),saved.argmax(1))
            actual=metrics(frame.iloc[ids].label,averaged)
            for name in ['accuracy','macro_f1','log_loss']:assert abs(actual[name]-target[name])<1e-7
    assert len(checked)==(12 if winner is None else 30)
    destination=ROOT/'results/research_nonlinear_backbone_adapter'
    dump(destination/'verification.json',{'passed':True,'checked_seed_fits':len(checked),'fits':checked,
        'checks':['Allsources,backbone/initialdeltahashes,fit-onlystats andpatientgroups verified',
            'Everycompleteactualteacher independently recovered onCPU fromoriginalM5source',
            'Allactualtrainlosscomponents,successful/skippedAMPupdates,schedulerstates andselectedepochs verified',
            'Everyselectedmodel actualstop/outerimages andallensembles reforwarded andclassdecisions restored',
            'Everyactualnonlinearadapterdelta atbothresolutions independently rebuilt withNumPy/SciPy',
            'OriginalpatientCVplansandfulloutercoverage preserved; noexternalimagesorscores']})


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(WEIGHTS)==protocol['backbone_sha256']
    verification=json.loads((ROOT/'verification_nonlinear_backbone_adapter.json').read_text());assert verification['passed']
    for name,expected in verification['code_sha256'].items():assert sha(ROOT/name)==expected
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,features,digest=load_data('multiscale_vit')
    assert digest==protocol['manifest_sha256']
    for v in features.values():v[2091:]=np.nan
    fit=frame.split.eq('train').to_numpy();stop=frame.split.eq('val').to_numpy();records=[]
    for index,cfg in enumerate(protocol['configuration_plan']):
        group=OUT/f'{index:02d}_{cfg["name"]}'
        for seed,source in enumerate(reference['selected']['checkpoints']):train(source,frame,features,digest,fit,stop,cfg,seed,group/f'seed_{seed}')
        row=summarize_group(group,frame)
        initial=np.mean([np.load(group/f'seed_{seed}'/'initial_predictions.npz')['p'] for seed in range(3)],0)
        row.update(candidate_id=group.name,nonlinear_config=cfg,initial_live_val=metrics(frame.loc[stop,'label'],initial))
        records.append(row);dump(OUT/'leaderboard.json',records)
    eligible=[r for r in records if r['nonlinear_config']['blocks'] and r['val']['macro_f1']>r['initial_live_val']['macro_f1'] and r['val']['macro_f1']>=reference['selected']['val']['macro_f1']]
    winner=None
    if eligible:
        winner=max(eligible,key=lambda r:r['val']['macro_f1'])
        dump(OUT/'selection_before_cv.json',{'winner':winner,'checkpoint_sha256':{str(seed):sha(path) for seed,path in enumerate(winner['checkpoints'])},'external_test_evaluated':False})
        for row in [records[0],winner]:crossvalidate(row,reference,frame,features,digest)
    for row in records:row['development_gate_passed']=bool('cv' in row and row['nonlinear_config']['blocks'] and row['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and row['val']['macro_f1']>=reference['selected']['val']['macro_f1'])
    dump(OUT/'state.json',{'status':'auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records,winner)
    destination=ROOT/'results/research_nonlinear_backbone_adapter'
    passed=[r for r in records if r['development_gate_passed']]
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,
        'qualifying_adapters':[r['candidate_id'] for r in eligible],'new_seed_fits':12 if winner is None else 30,
        'requires_fresh_patient_confirmation':bool(passed),'external_test_evaluated':False})
    if passed:
        chosen=passed[0];dump(destination/'selection_before_repeat.json',{'candidate':chosen,
            'checkpoint_sha256':{str(seed):sha(path) for seed,path in enumerate(chosen['checkpoints'])},
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':12 if winner is None else 30,
        'development_gate_passers':len(passed),'external_test_evaluated':False})
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc(),'external_test_evaluated':False});raise


if __name__=='__main__':main()


"""Fit-only warmstarted selected-patch value adaptation with frozen original M8 heads and tail."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiment import ROOT,KEYS,normalize,loss_fn,metrics,dump,load_data
from kernel_residual_experiments import phase_plans,sha
from normal_large_residual_experiments import check_partitions
from register_patch_query_data import load_all
from register_patch_value import RegisterPatchValueModel
from register_patch_value_witness import tensors_for,weights_for,write_witness,audit_witness
from audit_register_patch_value import independent_probability
from audit_large_token_evidence import audit_training_witness as audit_original_witness,independent_probability as source_independent
from finalize_large_token_evidence import restore_fusion_sources

OUT=ROOT/'runs/register_patch_value'
DEST=ROOT/'results/research_register_patch_value'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name);assert sha(path if path.is_absolute() else ROOT/path)==expected,name


def restore(c,tail):
    source=torch.load(c['source_checkpoint'],map_location='cpu',weights_only=False);assert sha(c['source_checkpoint'])==c['source_checkpoint_sha256']
    model=RegisterPatchValueModel(source,tail,c['adapter_config'],20291900+c['seed'])
    expected={name for name,p in model.named_parameters() if p.requires_grad};assert set(c['value_state_dict'])==expected
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(c['value_state_dict'][name])
    return model.eval(),source


@torch.inference_mode()
def predict(model,source,features,bank,ids,device='cuda',independent=False):
    model=model.to(device).eval();result=[]
    for begin in range(0,len(ids),256):
        selected_ids=ids[begin:begin+256];base,selected,registers=tensors_for(features,bank,selected_ids,source,device)
        if independent:
            state={name:p.detach() for name,p in model.named_parameters() if p.requires_grad}
            raw={key:torch.from_numpy(features[key][selected_ids]).to(device) for key in features}
            tail_state={name:value for name,value in model.tail.state_dict().items()}
            out=independent_probability(source,tail_state,state,raw,selected,model.config)
        else:out=model(base,selected,registers)
        result.append(out['p'].cpu().numpy())
    return np.concatenate(result)


def fit_adapter(folder,source_path,tail,features,bank,frame,fit,stop,config,seed):
    assert not (folder/'model.pt').exists();folder.mkdir(parents=True,exist_ok=True)
    source=torch.load(source_path,map_location='cpu',weights_only=False);model=RegisterPatchValueModel(source,tail,config,20291900+seed).cuda().train()
    ids=fit[np.random.default_rng(20291900+seed).permutation(len(fit))[:256]]
    write_witness(folder/'training_input_witness.npz',model,source,features,bank,frame,fit,ids,config,seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    history=[];best=None;waiting=0;started=time.time();generator=np.random.default_rng(20291900+seed)
    frozen_before={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name not in ['value_down','value_up']}
    for epoch in range(61):
        total=0.;model.train()
        if epoch:
            order=fit[generator.permutation(len(fit))]
            for begin in range(0,len(order),256):
                chosen=order[begin:begin+256];base,selected,registers=tensors_for(features,bank,chosen,source,'cuda')
                y,disease,weights=weights_for(frame,fit,chosen,source,'cuda')
                optimizer.zero_grad(set_to_none=True);out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg'])
                assert torch.isfinite(task);task.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
                optimizer.step();total+=float(task.detach())*len(chosen)
        p=predict(model,source,features,bank,stop);score=metrics(frame.iloc[stop].label,p);key=(score['macro_f1'],-score['log_loss'])
        history.append({'epoch':epoch,'train_loss':total/len(fit),'val':score})
        dump(folder/'history.json',history)
        if best is None or key>best:
            best=key;waiting=0
            checkpoint={'adapter_config':config,'seed':seed,'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),
                'value_state_dict':{name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad},'epoch':epoch,'val':score}
            torch.save(checkpoint,folder/'model.pt');np.savez_compressed(folder/'val_predictions.npz',p=p)
        else:waiting+=1
        if epoch%5==0:print(json.dumps({'event':'patch_value_epoch','folder':str(folder),'epoch':epoch,'seconds':time.time()-started}),flush=True)
        if waiting>=12:break
    for name,expected in frozen_before.items():assert torch.equal(expected,model.state_dict()[name].cpu()),name
    dump(folder/'summary.json',{'configuration':config,'seed':seed,'epoch':checkpoint['epoch'],'val':checkpoint['val'],'seconds':time.time()-started,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'original_head_and_tail_frozen_before_after':True})
    del model;torch.cuda.empty_cache()


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame,features,bank,tail,digest=load_all();assert digest==protocol['manifest_sha256']
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());plans={};sources=[]
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        check_partitions(frame,fit,stop,outer);plans[phase]=(fit,stop,outer)
        dump(OUT/'split_plans'/(phase+'.json'),{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
        for seed in range(3):
            folder=ROOT/'runs/large_token_evidence/registers'/phase/f'seed_{seed}';source=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            assert source['manifest_sha256']==digest and set(source['diseases'])==set(frame.iloc[fit].disease)
            for key,value in features.items():
                x=value[fit];axes=tuple(range(x.ndim-1));assert np.allclose(source['stats'][key]['mean'],x.mean(axes),atol=1e-6,rtol=1e-6)
                assert np.allclose(source['stats'][key]['std'],np.maximum(x.std(axes),.05),atol=1e-6,rtol=1e-6)
            witness=audit_original_witness(source,folder,features,frame,fit)
            sources.append({'phase':phase,'seed':seed,'checkpoint':str(folder/'model.pt'),'checkpoint_sha256':sha(folder/'model.pt'),'actual_original_fulltask_witness':witness})
    frozen={'new_seed_fits_so_far':0,'split_plan_sha256':{str(path):sha(path) for path in (OUT/'split_plans').glob('*.json')},
        'source_geometry_sha256':{str(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz'):sha(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz') for phase in plans}}
    dump(OUT/'geometry_freeze_before_all_fits.json',frozen)
    for arm in protocol['configuration_plan']:
        config={key:arm[key] for key in ['kind','rank','alpha']}
        for phase,(fit,stop,outer) in plans.items():
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                source_path=ROOT/'runs/large_token_evidence/registers'/phase/f'seed_{seed}'/'model.pt'
                fit_adapter(OUT/arm['name']/phase/f'seed_{seed}',source_path,tail,features,bank,frame,fit,stop,config,seed)
    products={str(path):sha(path) for path in OUT.glob('*/*/seed_*/*') if path.is_file()}
    dump(OUT/'training_products_freeze_before_audit.json',{'input_artifact_sha256':products})
    records=[];checked=[]
    for arm in protocol['configuration_plan']:
        config={key:arm[key] for key in ['kind','rank','alpha']};folds=[]
        for phase,(fit,stop,outer) in plans.items():
            dump(OUT/'state.json',{'status':'independent_auditing','pid':os.getpid(),'arm':arm['name'],'phase':phase,'external_test_evaluated':False})
            ps=[];evaluation=stop if phase=='validation' else outer
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['adapter_config']==config and c['seed']==seed
                history=json.loads((folder/'history.json').read_text());assert max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))['epoch']==c['epoch']
                model,source=restore(c,tail);p=predict(model,source,features,bank,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.array_equal(p,saved)
                p=predict(model,source,features,bank,evaluation);manual=predict(model,source,features,bank,evaluation,independent=True)
                assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
                witness=audit_witness(folder/'training_input_witness.npz',source,tail,features,bank,frame,fit,config,seed)
                checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'training_witness_sha256':sha(folder/'training_input_witness.npz'),'witness':witness});ps.append(p)
                del model;torch.cuda.empty_cache()
            p=np.mean(ps,0);folder=OUT/arm['name']/phase;np.savez_compressed(folder/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));score=metrics(frame.iloc[evaluation].label,p)
            if phase=='validation':val=score
            else:folds.append({'phase':phase,'metrics':score});np.savez_compressed(folder/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
        row={'name':arm['name'],'candidate_id':arm['name'],'register_patch_value_configuration':arm,'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds]))}}
        row['development_gate_passed']=row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'];records.append(row)
    _,base,base_digest=load_data('multiscale_vit');assert base_digest==digest
    for value in base.values():value[2091:]=np.nan
    b_sources=restore_fusion_sources(frame,base,digest,plans);assert len(b_sources)==24
    mixtures=[];phases=[]
    for source in records:
        folds=[];name=source['name']+'_m8replace'
        for phase,(fit,stop,outer) in plans.items():
            ids=stop if phase=='validation' else outer
            banks=[np.load(path/phase/'predictions.npz') for path in [ROOT/'runs/normal_subspace/zero_control',ROOT/'runs/normal_subspace/normal_rank32_full',OUT/source['name']]]
            assert all(np.array_equal(bank['indices'],ids) for bank in banks)
            a,b,t=[bank['p'] for bank in banks];p=(.1875*a+.5625*b+.25*t).astype('float32')
            manual=(.1875*a.astype('float64')+.5625*b.astype('float64')+.25*t.astype('float64')).astype('float32')
            assert np.allclose(p,manual,atol=2e-7,rtol=2e-7) and np.array_equal(p.argmax(1),manual.argmax(1))
            folder=OUT/name/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1));score=metrics(frame.iloc[ids].label,p)
            if phase=='validation':val=score
            else:folds.append({'phase':phase,'metrics':score});np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            phases.append({'source':source['name'],'phase':phase,'images':len(ids)})
        row={**source,'name':name,'candidate_id':name,'source_register_patch_value_arm':source['name'],'m8_replacement_weight':.25,'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds]))}}
        row['development_gate_passed']=row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'];mixtures.append(row)
    verify_frozen(protocol)
    for path,expected in products.items():assert sha(path)==expected
    for group in ['split_plan_sha256','source_geometry_sha256']:
        for path,expected in frozen[group].items():assert sha(path)==expected
    assert len(checked)==48 and len(phases)==16
    all_records=records+mixtures;eligible=[r for r in all_records if r['development_gate_passed']]
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':all_records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,'frozen_register_source_heads_checked':12,'register_source_heads':sources,
        'reused_fusion_source_heads_checked':24,'source_heads':b_sources,'fixed_fusion_phases_checked':phases,'only_value_adapters_trained':True,'external_test_evaluated':False})
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1']);name=winner.get('source_register_patch_value_arm',winner['name'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'requires_fresh_patient_confirmation':True,'checkpoint_sha256':{str(OUT/name/phase/f'seed_{seed}'/'model.pt'):sha(OUT/name/phase/f'seed_{seed}'/'model.pt') for phase in plans for seed in range(3)},'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'all_original_sources_bank_tail_preprocessing_and_training_products_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    hypothesis=ROOT/'results/next_register_patch_value_hypothesis.json';draft=json.loads(hypothesis.read_text());dump(hypothesis,{**draft,'status':'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed','classification_fits':48,'scores_evaluated':8,'passing_development_gates':len(eligible)})
    print(json.dumps({'event':'patch_value_complete','passing_development_gates':len(eligible)}),flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

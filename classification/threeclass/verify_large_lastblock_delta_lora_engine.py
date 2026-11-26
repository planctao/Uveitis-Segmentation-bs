"""Actual shared trainer wiring, RAM task witnesses, patient plans and frozen decision boundaries."""
import json,os,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,dump
from register_multiquery_data import load_all
from large_lastblock_delta_lora_cache_v2 import build_cache,array_sha
from large_lastblock_delta_lora import LastBlockDeltaLoRA,make_templates
from large_lastblock_delta_lora_engine import write_witness,audit_witness,sha
from large_lastblock_delta_lora_plans import make_plans,assert_unused,OUTER_SEEDS,INNER_SEEDS
import large_lastblock_delta_lora_engine as shared
import large_lastblock_delta_lora_experiments as original
import large_lastblock_delta_lora_confirmation as fresh
OUT=ROOT/'runs/large_lastblock_delta_lora_engine_preflight'
DEST=ROOT/'verification_large_lastblock_delta_lora_engine.json'
def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256','package_implementation_sha256']:
        for name,expected in protocol[section].items():
            p=Path(name);assert sha(p if p.is_absolute() else ROOT/p)==expected,name
def main():
    assert not DEST.exists();started=time.time();torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    inventory=assert_unused(ROOT);frame,features,digest=load_all();plans=make_plans(frame,digest)
    assert len(plans)==6;plan_rows=[]
    for plan in plans:
        arrays={key:np.asarray(plan[key],dtype='int64') for key in ['fit_indices','stop_indices','outer_indices']}
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[value][key]) for value in arrays.values()]
            assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1,3))
        assert all(frame.iloc[value].split.eq('train').all() for value in arrays.values())
        plan_rows.append({'repeat':plan['repeat'],'fold':plan['fold'],'inner_split_seed':plan['inner_split_seed'],
            **{key:len(value) for key,value in arrays.items()},**{key+'_sha256':array_sha(value) for key,value in arrays.items()}})
    configs=fresh.requested_adapter_configs({'kind':'attention','rank':4,'alpha':8})
    assert configs=={'value':{'kind':'attention','rank':4,'alpha':8},'matched_token':{'kind':'token','rank':8,'alpha':16}}
    assert 72+18*len(configs)==108 and 36+18*len(configs)==72
    formulas=[];rng=np.random.default_rng(20294400)
    for weight in [.25,1.]:
        predictions={}
        for name in ['reference','subspace','large','token','value','matched_token']:
            values=[]
            for seed in range(3):
                p=rng.uniform(.01,1,(41,3)).astype('float32');p/=p.sum(1,keepdims=True);values.append(p)
            predictions[name]=values
        fast=fresh.combine(predictions,weight);independent=fresh.combine(predictions,weight,independent=True)
        for key in fast:
            assert np.allclose(fast[key],independent[key],atol=2e-7,rtol=2e-7),key
            assert np.array_equal(fast[key].argmax(1),independent[key].argmax(1)),key
        means={key:np.mean(value,0).astype('float64') for key,value in predictions.items()}
        for label,key in [('candidate','value'),('matched_token_control','matched_token')]:
            expected=.1875*means['reference']+.5625*means['subspace']+.25*means[key] if weight==.25 else means[key]
            assert np.allclose(fast[label],expected,atol=2e-7,rtol=2e-7)
        formulas.append({'replacement_weight':weight,'supervised_fits':108,'actual_task_witnesses':72,'passed':True})
    gates=[]
    for suffix in ['', '_m8replace']:
        for dominant in ['m8','token_control']:
            thresholds={'m8':(.86,.91),'token_control':(.85,.90)} if dominant=='m8' else {'m8':(.85,.90),'token_control':(.86,.91)}
            m8,control=thresholds['m8'],thresholds['token_control']
            limits={'minimum_cv_macro_f1':m8[0],'minimum_validation_macro_f1':m8[1]}
            for cv,val,expected in [(.861,.91,True),(.86,.91,False),(.859,.92,False),(.87,.909,False),(.87,.911,True)]:
                row={'name':'attention_qkv_rank4'+suffix,'large_lastblock_delta_lora_configuration':{'role':'sole_upgrade_candidate'},'cv':{'mean_macro_f1':cv},'val':{'macro_f1':val}}
                by_name={'token_residual_rank8'+suffix:{'cv':{'mean_macro_f1':control[0]},'val':{'macro_f1':control[1]}}}
                assert original.development_gate(row,limits,by_name)==expected
                gates.append({'suffix':suffix,'dominant':dominant,'cv':cv,'validation':val,'expected':expected,'passed':True})
            row={'name':'token_residual_rank8'+suffix,'large_lastblock_delta_lora_configuration':{'role':'matched_capacity_control_not_upgrade_candidate'},'cv':{'mean_macro_f1':1.},'val':{'macro_f1':1.}}
            assert not original.development_gate(row,limits,{})
            gates.append({'suffix':suffix,'dominant':dominant,'control_never_nominated':True,'passed':True})
    controls=['m8','m7','m6','reference','fixed_gaussian','matched_token_control']
    def fixture():
        return ([{'arm':'candidate','mean_macro_f1':.85}]+[{'arm':name,'mean_macro_f1':.84} for name in controls],
            {name:{'macro_f1_ci95':[.001,.02]} for name in controls})
    rows,paired=fixture();assert fresh.confirmation_gate(rows,paired,True)
    fresh_gates=[{'case':'positive_all_six','passed':True}]
    for name in controls:
        for mode in ['equal_mean','lower_mean','zero_ci','negative_ci']:
            rows,paired=fixture()
            if mode.endswith('mean'):next(r for r in rows if r['arm']==name)['mean_macro_f1']=.85 if mode=='equal_mean' else .86
            else:paired[name]['macro_f1_ci95'][0]=0 if mode=='zero_ci' else -.001
            assert not fresh.confirmation_gate(rows,paired,True)
            fresh_gates.append({'control':name,'case':mode,'passed':True})
    fit=np.flatnonzero(frame.split.eq('train'));ids=shared.first_ids(fit,0)
    parent=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());cache,metadata=build_cache(frame,ids,parent)
    prior=json.loads((ROOT/'verification_large_lastblock_delta_lora_cache_v2.json').read_text())
    assert metadata['arrays']==prior['metadata']['arrays'] and metadata['rotary_inputs']==prior['metadata']['rotary_inputs']
    templates=make_templates(parent);source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False);actual=[]
    for arm,config in configs.items():
        engine=fresh.engine(config)
        assert engine==(shared.fit_adapter,shared.restore,shared.predict,shared.audit_witness)
        assert original.fit_adapter is shared.fit_adapter and original.predict is shared.predict and original.restore is shared.restore and original.audit_witness is shared.audit_witness
        with tempfile.TemporaryDirectory(prefix='lastblock-delta-engine-') as directory:
            folder=Path(directory)
            fresh.freeze_fresh_warmstart(folder,source_path,features,cache,templates,fit,config,0)
            warm=json.loads((folder/'warmstart_freeze_before_adapter.json').read_text())
            assert warm['actual_images']==256 and warm['adapter_config']==config and warm['new_adapter_fits_so_far']==0
            model=LastBlockDeltaLoRA(source,templates,config,20294400).cuda().train()
            write_witness(folder/'task.npz',model,source,features,cache,frame,fit,config,0)
            witness=audit_witness(folder/'task.npz',source,templates,features,cache,frame,fit,config,0)
            with torch.no_grad():model.adapter_b.copy_(torch.cos(torch.arange(model.adapter_b.numel(),device='cuda').reshape_as(model.adapter_b)*.17)*1e-6)
            p=shared.predict(model,source,features,cache,ids);manual=shared.predict(model,source,features,cache,ids,independent=True)
            assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
            checkpoint={'adapter_config':config,'seed':0,'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),
                'trainable_state_dict':{n:parameter.detach().cpu().clone() for n,parameter in model.named_parameters() if parameter.requires_grad}}
            torch.save(checkpoint,folder/'model.pt')
            restored,_=shared.restore(torch.load(folder/'model.pt',map_location='cpu',weights_only=False),templates)
            assert np.array_equal(p,shared.predict(restored,source,features,cache,ids))
            actual.append({'arm':arm,'configuration':config,'actual_images':256,'actual_task_witness':witness,
                'exact_source_warmstart':True,'original_fresh_same_trainer_predict_restore_witness_entrypoints':True,
                'maximum_prediction_error':float(np.abs(p-manual).max()),'ordinary_cuda_restore_bitwise':True})
            del model,restored;torch.cuda.empty_cache()
            print(json.dumps({'event':'actual256engine_case_passed','arm':arm}),flush=True)
    verify(protocol)
    dump(DEST,{'passed':True,'seconds':time.time()-started,'plans':plan_rows,'outer_split_seeds':OUTER_SEEDS,'inner_seeds':INNER_SEEDS,
        'seed_inventory_sha256':{str(p):sha(p) for p in inventory},'supervised_fits_if_eligible':108,'actual_task_witnesses_if_eligible':72,
        'synthetic_formula_cases':formulas,'development_gate_cases':gates,'confirmation_gate_cases':fresh_gates,'actual_engine_cases':actual,
        'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],'package_implementation_sha256':protocol['package_implementation_sha256'],
        'engineering_protocol_sha256':sha(OUT/'protocol.json'),'classification_fits':0,'classification_scores':0,'engineering_optimizer_steps':0,'external_raw_images_read':0,
        'scope':'6patient+near-duplicate isolated plans,2fusion formulas,24original/25fresh gate cases,2actual256 shared-engine/warmstart/input-witness/fulltask/predict/restore cases. Uses existing original source for engineering; actual fresh source/geometry/task identities require eventual108fit/72task audit. No classifier fitting or performance evidence.'})
    print('Actual256shared-engine/control/plan proof passed; no classifier fits/scores.',flush=True)
if __name__=='__main__':
    dump(OUT/'state.json',{'status':'running_actual_engine_patient_gate_proof','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    try:
        main();dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    except BaseException:
        dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0});raise

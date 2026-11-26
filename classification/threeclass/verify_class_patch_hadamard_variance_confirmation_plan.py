"""Verify patient plans, gates, formulas and actual original/fresh dispersion engine wiring."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,dump,loss_fn
from kernel_residual_experiments import sha
from register_patch_query_data import load_all
from register_patch_query_features import array_sha
from class_patch_hadamard_variance_plans import make_plans,assert_unused,OUTER_SEEDS,INNER_SEEDS
import class_patch_hadamard_variance_experiments as original
import class_patch_hadamard_variance_confirmation as fresh
from class_patch_hadamard_variance_witness import tensors_for,weights_for,audit_witness,write_witness
from audit_class_patch_hadamard_variance import independent_probability
from audit_normal_tangent import complete_task

OUT=ROOT/'runs/class_patch_hadamard_variance_engine_preflight'


def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def main():
    destination=ROOT/'verification_class_patch_hadamard_variance_confirmation_plan.json';assert not destination.exists()
    started=time.time();torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    inventory=assert_unused(ROOT)
    frame,features,bank,tail,digest=load_all();plans=make_plans(frame,digest)
    assert len(plans)==6
    summaries=[]
    for plan in plans:
        arrays={k:np.asarray(plan[k],dtype='int64') for k in ['fit_indices','stop_indices','outer_indices']}
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[values][key]) for values in arrays.values()]
            assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1,3))
        summaries.append({'repeat':plan['repeat'],'fold':plan['fold'],'inner_split_seed':plan['inner_split_seed'],
            **{k:len(v) for k,v in arrays.items()},**{k+'_sha256':array_sha(v) for k,v in arrays.items()}})
    formula_cases=[];rng=np.random.default_rng(20294200)
    for pooling in ['uniform','source']:
        architecture={'pooling':pooling,'statistic':'log_hadamard_variance','matched_mean':'uniform_pool' if pooling=='uniform' else 'source_query_pool'}
        configs=fresh.requested_adapter_configs(architecture)
        assert configs=={'value':{'pooling':pooling,'statistic':'log_hadamard_variance'},'matched_mean':{'pooling':pooling,'statistic':'mean'},'matched_variance':{'pooling':pooling,'statistic':'log_variance'}}
        assert 72+18*len(configs)==126 and 36+18*len(configs)==90
        for weight in [.25,1.]:
            predictions={name:[] for name in ['reference','subspace','large','token','value','matched_mean','matched_variance']}
            for name in predictions:
                for seed in range(3):
                    p=rng.uniform(.01,1,(41,3)).astype('float32');p/=p.sum(1,keepdims=True);predictions[name].append(p)
            actual=fresh.combine(predictions,weight);manual=fresh.combine(predictions,weight,independent=True)
            means={name:np.mean(value,axis=0).astype('float64') for name,value in predictions.items()}
            for name,value in actual.items():
                if name in manual:assert np.allclose(value,manual[name],atol=2e-7,rtol=2e-7) and np.array_equal(value.argmax(1),manual[name].argmax(1)),name
            expected=(.1875*means['reference']+.5625*means['subspace']+.25*means['value']) if weight==.25 else means['value']
            control=(.1875*means['reference']+.5625*means['subspace']+.25*means['matched_mean']) if weight==.25 else means['matched_mean']
            assert np.allclose(actual['candidate'],expected,atol=2e-7,rtol=2e-7)
            assert np.allclose(actual['matched_mean_control'],control,atol=2e-7,rtol=2e-7)
            variance_control=(.1875*means['reference']+.5625*means['subspace']+.25*means['matched_variance']) if weight==.25 else means['matched_variance']
            assert np.allclose(actual['matched_variance_control'],variance_control,atol=2e-7,rtol=2e-7)
            formula_cases.append({'pooling':pooling,'replacement_weight':weight,'supervised_fits':126,'actual_task_witnesses':90,'passed':True})
    gate_cases=[]
    for pooling in ['uniform','source']:
        control='uniform_pool' if pooling=='uniform' else 'source_query_pool'
        variance='uniform_variance' if pooling=='uniform' else 'source_query_variance'
        for suffix in ['', '_m8replace']:
            architecture={'pooling':pooling,'statistic':'log_hadamard_variance','matched_mean':control,'matched_variance':variance}
            for dominant,(cv_limits,val_limits) in enumerate([([.86,.85,.84],[.91,.90,.89]),([.84,.86,.85],[.89,.91,.90]),([.85,.84,.86],[.90,.89,.91])]):
                policy={'minimum_cv_macro_f1':cv_limits[0],'minimum_validation_macro_f1':val_limits[0],
                    'frozen_mean_controls':[{'name':control+suffix,'cv':{'mean_macro_f1':cv_limits[1]},'val':{'macro_f1':val_limits[1]}}],
                    'frozen_variance_controls':[{'name':variance+suffix,'cv':{'mean_macro_f1':cv_limits[2]},'val':{'macro_f1':val_limits[2]}}]}
                cv_max=max(cv_limits);val_max=max(val_limits)
                for cv,val,expected in [(cv_max+.001,val_max,True),(cv_max,val_max,False),(cv_max-.001,val_max+.01,False),(cv_max+.01,val_max-.001,False),(cv_max+.01,val_max+.001,True)]:
                    row={'name':'hadamard_variance'+suffix,'class_patch_hadamard_variance_configuration':architecture,'cv':{'mean_macro_f1':cv},'val':{'macro_f1':val}}
                    assert original.development_gate(row,policy)==expected
                    gate_cases.append({'pooling':pooling,'suffix':suffix,'cv':cv,'validation':val,'expected':expected,'dominant_reference_or_control':dominant,'passed':True})
    controls=['m8','m7','m6','reference','fixed_gaussian','matched_mean_control','matched_variance_control']
    def confirmation_fixture():
        rows=[{'arm':'candidate','mean_macro_f1':.85}]+[{'arm':name,'mean_macro_f1':.84} for name in controls]
        comparisons={name:{'macro_f1_ci95':[.001,.02]} for name in controls}
        return rows,comparisons
    rows,comparisons=confirmation_fixture();assert fresh.confirmation_gate(rows,comparisons,True)
    confirmation_cases=[{'case':'positive_all_seven','passed':True}]
    for name in controls:
        for mode in ['equal_mean','lower_mean','zero_ci','negative_ci']:
            rows,comparisons=confirmation_fixture()
            if mode.endswith('mean'):next(r for r in rows if r['arm']==name)['mean_macro_f1']=.85 if mode=='equal_mean' else .86
            else:comparisons[name]['macro_f1_ci95'][0]=0 if mode=='zero_ci' else -.001
            assert not fresh.confirmation_gate(rows,comparisons,True)
            confirmation_cases.append({'control':name,'case':mode,'passed':True})
    fit=np.flatnonzero(frame.split.eq('train'));ids=fit[np.random.default_rng(20293500).permutation(len(fit))[:256]]
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
    raw={key:torch.from_numpy(value[ids]).cuda() for key,value in features.items()}
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    actual_cases=[]
    for pooling in ['uniform','source']:
        for statistic in ['log_hadamard_variance','mean','log_variance']:
            config={'pooling':pooling,'statistic':statistic}
            trainer,restorer,predictor,witness_audit=fresh.engine(config)
            assert trainer is original.fit_adapter and restorer is original.restore and predictor is original.predict and witness_audit is audit_witness
            with tempfile.TemporaryDirectory(prefix='hadamard-variance-engine-') as directory:
                folder=Path(directory)
                fresh.freeze_fresh_warmstart(folder,source_path,features,bank,tail,fit,config,0)
                warm=json.loads((folder/'warmstart_freeze_before_adapter.json').read_text())
                assert warm['adapter_config']==config and warm['actual_images']==256 and warm['new_adapter_fits_so_far']==0
                model=original.ClassPatchHadamardVarianceModel(source,tail,config,20293500).cuda().train()
                witness_path=folder/'first_task.npz';write_witness(witness_path,model,source,features,bank,frame,fit,ids,config,0)
                witness=witness_audit(witness_path,source,tail,features,bank,frame,fit,config,0)
                with torch.no_grad():
                    model.patch_readout.copy_(torch.cos(torch.arange(4096,device='cuda').reshape_as(model.patch_readout)*.17)*.002)
                    model.patch_bias.copy_(torch.tensor([.01,-.02],device='cuda'))
                actual=predictor(model,source,features,bank,ids)
                independent=predictor(model,source,features,bank,ids,independent=True)
                assert np.allclose(actual,independent,atol=5e-5,rtol=5e-5) and np.array_equal(actual.argmax(1),independent.argmax(1))
                parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
                model.zero_grad(set_to_none=True);out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg']);task.backward()
                state={n:p.detach().clone().requires_grad_() for n,p in parameters.items()}
                manual=independent_probability(source,{},state,raw,selected,config)
                independent_task=complete_task(manual,y,disease,weights,source['cfg']);independent_task.backward()
                gradient_errors={}
                for name,p in parameters.items():
                    assert torch.isfinite(p.grad).all() and p.grad.abs().max()>0
                    gradient_errors[name]=float((p.grad-state[name].grad).abs().max())
                    assert torch.allclose(p.grad,state[name].grad,atol=1e-6,rtol=3e-3)
                checkpoint={'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),'adapter_config':config,'seed':0,
                    'trainable_state_dict':{name:p.detach().cpu().clone() for name,p in parameters.items()}}
                checkpoint_path=folder/'model.pt';torch.save(checkpoint,checkpoint_path)
                restored,_=restorer(torch.load(checkpoint_path,map_location='cpu',weights_only=False),tail)
                replay=predictor(restored,source,features,bank,ids);assert np.array_equal(actual,replay)
                actual_cases.append({'pooling':pooling,'statistic':statistic,'actual_images':256,'exact_warmstart':True,
                    'actual_first_task':witness,'maximum_prediction_error':float(np.abs(actual-independent).max()),
                    'nonzero_complete_task_gradient_errors':gradient_errors,'ordinary_cuda_prediction_restore_bitwise':True})
                del model,restored,out,manual,state,parameters;torch.cuda.empty_cache()
    verify(protocol)
    dump(destination,{'passed':True,'manifest_sha256':digest,'engineering_protocol_sha256':sha(OUT/'protocol.json'),
        'plans':summaries,'outer_split_seeds':OUTER_SEEDS,'inner_seeds':INNER_SEEDS,'seed_inventory_sha256':{str(p):sha(p) for p in inventory},
        'supervised_fits':126,'actual_task_witnesses':90,'synthetic_formula_cases':formula_cases,
        'development_gate_cases':gate_cases,'confirmation_gate_cases':confirmation_cases,'actual_original_fresh_control_engine_cases':actual_cases,
        'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],
        'classification_fits':0,'classification_scores':0,'engineering_optimizer_steps':0,'external_images_read':0,'new_backbone_view_forwards':0,
        'seconds':time.time()-started,'scope':'Six isolated plans, four formulas,60 original/29fresh gates and six actual256 engine warmstart/prediction/fulltask/restore cases passed. No classifier fits or performance evidence; matching fresh source/control identities require actual eligible confirmation.'})
    print('Hadamard-variance126fit/90witness patient/control/engine plan passed; no classifier fits/scores.',flush=True)


if __name__=='__main__':
    import os,traceback
    dump(OUT/'state.json',{'status':'running_engine_plan_proof','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    try:
        main()
        dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    except BaseException:
        dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0})
        raise


"""Actual engine, matched-control, isolation, and decision checks before classifier fits."""
import copy
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from register_patch_query_features import array_sha
from register_patch_query_selection_data import load_all
from register_patch_attention_coupling import RegisterPatchAttentionCouplingModel, restore
from register_patch_attention_coupling_witness import write_witness
from register_patch_attention_coupling_experiments import predict, development_gate, fit_adapter
from register_patch_attention_coupling_confirmation import combine, freeze_fresh_warmstart, requested_adapter_configs, engine, confirmation_gate
from register_patch_attention_coupling_plans import make_plans, assert_unused, seed_allocations, OUTER_SEEDS, INNER_SEEDS

OUT = ROOT/'runs/register_patch_attention_coupling_engine_preflight'


def verify(protocol):
    for section in ['code_sha256', 'input_artifact_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path) == expected, name


def gate_cases(arms):
    def row(arm, fused=False, cv=.86, val=.91):
        return {'name': arm['name']+('_m8replace' if fused else ''),
            'register_patch_attention_coupling_configuration': arm,
            'cv': {'mean_macro_f1': cv}, 'val': {'macro_f1': val}}
    controls = {a['name']: {kind: {'cv_macro_f1': .85, 'validation_macro_f1': .90}
        for kind in ['standalone', 'fixed_fusion']} for a in arms if a['adapt_patches']}
    protocol = {'minimum_cv_macro_f1': .84, 'minimum_validation_macro_f1': .89,
        'frozen_reader_matched_controls': controls}
    by_name = {r['name']: r for r in [row(arms[0], f, .845, .895) for f in [False, True]]}
    cases = []
    for fused in [False, True]:
        reader = row(arms[0], fused)
        for cv, val, expected in [(.84,.90,False),(.85,.889,False),(.85,.89,True)]:
            sample = copy.deepcopy(reader);sample['cv']['mean_macro_f1']=cv;sample['val']['macro_f1']=val
            assert development_gate(sample, protocol, by_name) is expected
            cases.append({'arm': sample['name'], 'cv': cv, 'val': val, 'passed': expected})
        for arm in arms[1:]:
            sample = row(arm, fused)
            assert development_gate(sample, protocol, by_name)
            cases.append({'arm': sample['name'], 'condition': 'strictly_above_all_CV_and_validation_at_least_all', 'passed': True})
            # Each comparator must independently be capable of vetoing the candidate.
            for comparator in ['m8', 'reader', 'frozen_patch']:
                for metric in ['cv', 'validation']:
                    p = copy.deepcopy(protocol); rows=copy.deepcopy(by_name)
                    value = sample['cv']['mean_macro_f1'] if metric == 'cv' else sample['val']['macro_f1']+.001
                    if comparator == 'm8':
                        p['minimum_cv_macro_f1' if metric=='cv' else 'minimum_validation_macro_f1']=value
                    elif comparator == 'reader':
                        target=rows['attention_only'+('_m8replace' if fused else '')]
                        target['cv' if metric=='cv' else 'val']['mean_macro_f1' if metric=='cv' else 'macro_f1']=value
                    else:
                        p['frozen_reader_matched_controls'][arm['name']]['fixed_fusion' if fused else 'standalone']['cv_macro_f1' if metric=='cv' else 'validation_macro_f1']=value
                    assert not development_gate(sample, p, rows), (sample, comparator, metric)
                    cases.append({'arm': sample['name'], 'veto': comparator, 'metric': metric, 'passed': False})
    return cases


def main():
    destination = ROOT/'verification_register_patch_attention_coupling_confirmation_plan.json'
    assert not destination.exists()
    torch.set_num_threads(4)
    started = time.time()
    protocol = json.loads((OUT/'protocol.json').read_text());verify(protocol)
    prereg = json.loads((ROOT/'results/register_patch_attention_coupling_preregistration_before_query_outcome.json').read_text())
    arms = prereg['configuration_plan']
    assert prereg['confirmation_plan']['outer_seeds'] == OUTER_SEEDS
    assert prereg['confirmation_plan']['inner_seeds'] == INNER_SEEDS
    seed_records = assert_unused(ROOT)
    assert seed_allocations({'confirmation_plan': {'outer_seeds': OUTER_SEEDS}}) == set()
    assert seed_allocations({'outer_split_seeds': OUTER_SEEDS}) == set(OUTER_SEEDS)
    with tempfile.TemporaryDirectory(prefix='coupling-allocation-check-') as directory:
        root=Path(directory);folder=root/'runs/example';folder.mkdir(parents=True)
        path=folder/'protocol.json'
        path.write_text(json.dumps({'confirmation_plan':{'outer_seeds':OUTER_SEEDS}}));assert_unused(root)
        path.write_text(json.dumps({'outer_split_seeds':OUTER_SEEDS}))
        try: assert_unused(root)
        except AssertionError: pass
        else: raise AssertionError('Actual allocated patient seeds were accepted')
    frame, features, bank, tail, digest = load_all()
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = fit[np.random.default_rng(20292900).permutation(len(fit))[:256]]
    plans = make_plans(frame, digest)
    plan_cases = []
    for p in plans:
        parts=[frame.iloc[p[name]] for name in ['fit_indices','stop_indices','outer_indices']]
        assert all(part.split.eq('train').all() for part in parts)
        for key in ['patient_group','split_group']:
            sets=[set(part[key]) for part in parts]
            assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i+1,3))
        plan_cases.append({k:p[k] for k in ['repeat','fold','inner_split_seed']} |
            {name:len(p[name+'_indices']) for name in ['fit','stop','outer']} |
            {name+'_sha256':array_sha(np.asarray(p[name+'_indices'])) for name in ['fit','stop','outer']})
    generator=np.random.default_rng(20293400)
    values={}
    for name in ['reference','subspace','large','token','value','fixed_gaussian','matched_reader_only','matched_frozen_reader_patch']:
        values[name]=[]
        for seed in range(3):
            p=generator.random((256,3),dtype=np.float32);values[name].append(p/p.sum(1,keepdims=True))
    formula_cases=[]
    for coupled in [False,True]:
        selected={k:v for k,v in values.items() if coupled or not k.startswith('matched_')}
        for weight in [.25,1.]:
            fast=combine(selected,weight);manual=combine(selected,weight,independent=True)
            avg={k:np.mean(v,0).astype('float64') for k,v in selected.items()}
            expected={
                'm6':.25*avg['reference']+.75*avg['subspace'],
                'm7':.1875*avg['reference']+.5625*avg['subspace']+.25*avg['large'],
                'm8':.1875*avg['reference']+.5625*avg['subspace']+.25*avg['token'],
                'candidate':avg['value'] if weight==1. else .1875*avg['reference']+.5625*avg['subspace']+.25*avg['value']}
            for name in ['matched_reader_only','matched_frozen_reader_patch']:
                if coupled: expected[name+'_control']=avg[name] if weight==1. else .1875*avg['reference']+.5625*avg['subspace']+.25*avg[name]
                assert (name+'_control' in fast)==coupled
            for name in fast:
                assert np.allclose(fast[name],manual[name],atol=2e-7,rtol=2e-7)
                assert np.array_equal(fast[name].argmax(1),manual[name].argmax(1))
            for name,value in expected.items():
                assert np.array_equal(manual[name],value.astype('float32'))
            formula_cases.append({'coupled':coupled,'weight':weight,'independent_expanded_formula_arms':list(expected),
                'maximum_probability_error':float(np.abs(fast['candidate']-manual['candidate']).max())})
    decisions=gate_cases(arms)
    confirmation_decisions=[]
    for coupled in [False,True]:
        controls=['m8','m7','m6','reference','fixed_gaussian']+(['matched_reader_only_control','matched_frozen_reader_patch_control'] if coupled else [])
        rows=[{'arm':name,'mean_macro_f1':.84} for name in controls]+[{'arm':'candidate','mean_macro_f1':.85}]
        comparisons={name:{'macro_f1_ci95':[.001,.02]} for name in controls}
        assert confirmation_gate(rows,comparisons,coupled)
        for name in controls:
            for boundary in ['mean','ci']:
                r=copy.deepcopy(rows);c=copy.deepcopy(comparisons)
                if boundary=='mean':next(v for v in r if v['arm']==name)['mean_macro_f1']=.85
                else:c[name]['macro_f1_ci95'][0]=0.
                assert not confirmation_gate(r,c,coupled)
                confirmation_decisions.append({'coupled':coupled,'veto':name,'boundary':boundary,'passed':False})
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash=sha(source_path);source=torch.load(source_path,map_location='cpu',weights_only=False)
    warmstarts=[];prediction_cases=[];control_cases=[]
    keys=['kind','rank','alpha','selection','adapt_attention','adapt_patches']
    for arm in arms:
        config={k:arm[k] for k in keys}
        routed=requested_adapter_configs(arm)
        assert routed['value']==config and len(routed)==(3 if arm['adapt_patches'] else 1)
        with tempfile.TemporaryDirectory(prefix='coupling-engine-') as directory:
            folder=Path(directory)
            freeze_fresh_warmstart(folder,source_path,features,bank,tail,fit,config,0)
            warm=json.loads((folder/'warmstart_freeze_before_adapter.json').read_text())
            assert warm['adapter_config']==config and warm['first_batch_indices']==ids.tolist()
            warmstarts.append(warm)
            model=RegisterPatchAttentionCouplingModel(source,tail,config,20292900).cuda().eval()
            # Exercise restore/predict with both pathways away from their source state.
            with torch.no_grad():
                model.head.token_evidence.readout.bias.add_(.001)
                if config['adapt_patches']:model.value_up.fill_(.001)
            p=predict(model,source,features,bank,ids)
            manual=predict(model,source,features,bank,ids,independent=True)
            assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
            checkpoint={'source_checkpoint':str(source_path),'source_checkpoint_sha256':source_hash,'adapter_config':config,'seed':0,
                'trainable_state_dict':{n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad}}
            torch.save(checkpoint,folder/'restore_probe.pt')
            restored,_=restore(torch.load(folder/'restore_probe.pt',map_location='cpu',weights_only=False),tail)
            again=predict(restored,source,features,bank,ids)
            assert np.array_equal(p,again)
            prediction_cases.append({'arm':arm['name'],'actual_images':len(ids),'nonzero_state_restore_bitwise':True,
                'original_predict_independent_max_error':float(np.abs(p-manual).max())})
            del model,restored;torch.cuda.empty_cache()
        control_names=['value'] if not config['adapt_patches'] else ['matched_frozen_reader_patch']
        for control_name in control_names:
            c=routed[control_name]
            trainer,loader,predictor,auditor=engine(c)
            assert trainer is fit_adapter and loader is restore and predictor is predict
            with tempfile.TemporaryDirectory(prefix='coupling-control-') as directory:
                folder=Path(directory)
                freeze_fresh_warmstart(folder,source_path,features,bank,tail,fit,c,0)
                model=RegisterPatchAttentionCouplingModel(source,tail,c,20292900).cuda().train()
                path=folder/'training_input_witness.npz';write_witness(path,model,source,features,bank,frame,fit,ids,c,0)
                actual=auditor(path,source,tail,features,bank,frame,fit,c,0)
                assert actual['actual_images']==256
                control_cases.append({'source_arm':arm['name'],'control':control_name,'config':c,'actual_first_task_witness':actual})
                del model;torch.cuda.empty_cache()
    assert len(control_cases)==len(warmstarts)==len(prediction_cases)==4
    assert sha(source_path)==source_hash
    verify(protocol)
    dump(destination,{'passed':True,'manifest_sha256':digest,'engineering_protocol_sha256':sha(OUT/'protocol.json'),
        'outer_split_seeds':OUTER_SEEDS,'plans':plan_cases,'seed_allocation_rule_tested':True,
        'seed_inventory_sha256':{str(path):sha(path) for path in seed_records},
        'reader_only_winner_supervised_fits':90,'reader_only_winner_training_witnesses':54,
        'coupled_winner_supervised_fits':126,'coupled_winner_training_witnesses':90,
        'synthetic_formula_cases':formula_cases,'development_gate_cases':decisions,'confirmation_gate_cases':confirmation_decisions,
        'original_source_engine_preflight_cases':warmstarts,'actual_prediction_restore_engine_cases':prediction_cases,
        'actual_matched_control_engine_cases':control_cases,'matched_control_config_routing_verified':True,
        'source_checkpoint_sha256':{str(source_path):source_hash},'code_sha256':protocol['code_sha256'],
        'input_artifact_sha256':protocol['input_artifact_sha256'],
        'classification_fits':0,'classification_scores':0,'engineering_optimizer_steps':0,
        'external_images_read':0,'new_backbone_view_forwards':0,'seconds':time.time()-started,
        'limitation':'Actual cached training-image inference/backward checks, not classifier fitting or accuracy evidence. Fresh source identities, full tasks and controls must be replayed during any eligible confirmation.'})
    print('Six isolated plans, four formulas, original gates, four warmstarts/predict/restore and four actual256 controls passed; zero classifier fits/scores.',flush=True)


if __name__=='__main__':main()

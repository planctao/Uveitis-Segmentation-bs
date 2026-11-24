"""Fixed hierarchy-preserving soft disease targets; original M5 inference."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,load_data,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha,collect_cpu
from milestone2 import paired_oof
from taxonomy_smoothing import build_transition,SmoothingObjective
from verify_taxonomy_smoothing import independent_transition,independent_witness

OUT=ROOT/'runs/taxonomy_smoothing'
DEST=ROOT/'results/research_taxonomy_smoothing'


def configuration(reference,arm):
    return {**reference['selected']['config'],'taxonomy_smoothing_mode':arm['mode'],
        'taxonomy_smoothing_alpha':arm['alpha']}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_taxonomy_smoothing.json').read_text())['passed']
    for name,digest in protocol['code_sha256'].items():assert sha(ROOT/name)==digest
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for v in features.values():v[2091:]=np.nan
    records=[]
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            diseases=sorted(frame.iloc[fit].disease.unique())
            bank=build_transition(frame.iloc[fit],features['global'][fit],diseases,arm['mode'])
            (target/phase).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/phase/'fit_transition.npz',**bank,fit_indices=fit)
            ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                objective=SmoothingObjective(bank['transition'],folder/'first_objective_witness.npz')
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,training_loss=objective)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(collect_cpu(c,features,evaluation)[1])
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame);record.update(name=arm['name'],candidate_id=arm['name'],smoothing_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=bool(arm['alpha'] and record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'taxonomy_smoothing_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records)
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':chosen,
            'frozen_validation_checkpoint_sha256':{str(seed):sha(path) for seed,path in enumerate(chosen['checkpoints'])},
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit(frame,features,digest,reference,records):
    checked=[];witnesses=[];controls=[];transitions=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            tf=frame.iloc[fit];diseases=sorted(tf.disease.unique());mapping=tf.groupby('disease').label.first()
            bank_path=OUT/record['name']/phase/'fit_transition.npz';bank=dict(np.load(bank_path))
            assert np.array_equal(bank['fit_indices'],fit) and np.array_equal(bank['diseases'],diseases)
            expected,centers=independent_transition(tf,features['global'][fit],diseases,record['smoothing_configuration']['mode'])
            assert np.allclose(centers,bank['centers'],atol=1e-12,rtol=1e-12)
            assert np.allclose(expected,bank['transition'],atol=5e-8,rtol=5e-8)
            assert np.array_equal(bank['parents'],[int(mapping[d]) for d in diseases])
            assert np.array_equal(bank['patient_counts'],[int(tf.loc[tf.disease.eq(d),'patient_group'].nunique()) for d in diseases])
            transitions.append({'arm':record['name'],'phase':phase,'fit_transition_sha256':sha(bank_path)})
            ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==diseases
                assert c['cfg']=={**configuration(reference,record['smoothing_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                source=torch.load(sources[seed],map_location='cpu',weights_only=False)
                assert {k:tuple(v.shape) for k,v in c['state_dict'].items()}=={k:tuple(v.shape) for k,v in source['state_dict'].items()}
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stopping=collect_cpu(c,features,stop)[1];saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stopping,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stopping.argmax(1),saved.argmax(1))
                p=stopping if phase=='validation' else collect_cpu(c,features,outer)[1];ps.append(p)
                proof_path=folder/'first_objective_witness.npz';proof=np.load(proof_path)
                assert np.array_equal(proof['transition'],bank['transition'])
                assert float(proof['alpha'])==record['smoothing_configuration']['alpha']
                detail=independent_witness(proof);witnesses.append({'arm':record['name'],'phase':phase,'seed':seed,
                    'witness_sha256':sha(proof_path),**detail})
                if not record['smoothing_configuration']['alpha']:
                    original=collect_cpu(source,features,evaluation)[1]
                    assert np.allclose(p,original,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),original.argmax(1))
                    controls.append({'phase':phase,'seed':seed,'all_source_decisions_reproduced':True})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'taxonomy_smoothing_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==len(witnesses)==48 and len(controls)==12 and len(transitions)==16
    dump(DEST/'target_diagnostics.json',{'actual_first_step_witnesses':witnesses,'original_controls':controls,'transitions':transitions})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,
        'actual_training_witnesses_checked':48,'fit_transition_banks_checked':16,'original_control_heads_checked':12,'fits':checked,
        'checks':['All48actualfitstats,patientisolation,diseasemaps,schema andstop/outerprobabilities restored onCPU',
            'All16fit-onlypatientbalancedcentroids andwithinparenttransitionbanks independently rebuilt',
            'All48actualfirststeplogit/losswitnesses independently reconstructed',
            'All12alpha0controls reproduce originalM5heldoutdecisions',
            'Allensembles/metrics andoriginaltraincoverage reconstructed; noexternal scoring']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

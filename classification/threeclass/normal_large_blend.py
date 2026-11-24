"""Fixed small normal-L evidence mixtures with the retained M6 probabilities."""
import fcntl
import json
import os
import traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from experiment import ROOT,dump,metrics
from kernel_residual_experiments import sha,phase_plans
from milestone2 import paired_oof

OUT=ROOT/'runs/normal_large_blend'
DEST=ROOT/'results/research_normal_large_blend'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for key in ['code_sha256','frozen_artifacts','source_prediction_sha256','source_checkpoint_sha256']:
        for path,expected in protocol[key].items():assert sha(ROOT/path if not Path(path).is_absolute() else path)==expected


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv');assert digest==protocol['manifest_sha256']
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());records=[];checked=[]
    for arm in protocol['configuration_plan']:
        target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            dump(OUT/'state.json',{'status':'evaluating_fixed_development_mixtures','pid':os.getpid(),'arm':arm['name'],'phase':phase,'external_test_evaluated':False})
            ids=stop if phase=='validation' else outer;assert max(ids)<2091
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[indices][key]) for indices in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            saved=[np.load(ROOT/'runs/normal_subspace'/scope/phase/'predictions.npz') for scope in ['zero_control','normal_rank32_full']]
            saved.append(np.load(ROOT/'runs/normal_large_reference'/arm['feature']/phase/'predictions.npz'))
            for z in saved:
                assert np.array_equal(z['indices'],ids) and np.isfinite(z['p']).all() and np.all(z['p']>=0) and np.allclose(z['p'].sum(1),1,atol=1e-5)
            raw,normal,large=[z['p'] for z in saved];w=arm['large_weight']
            m6=(.25*raw+.75*normal).astype('float32');p=((1-w)*m6+w*large).astype('float32')
            retained=np.load(ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75'/phase/'predictions.npz')
            assert np.array_equal(retained['indices'],ids) and np.allclose(retained['p'],m6,atol=1e-7,rtol=1e-7)
            assert np.array_equal(retained['p'].argmax(1),m6.argmax(1))
            independent=((.25*(1-w))*raw.astype('float64')+(.75*(1-w))*normal.astype('float64')+w*large.astype('float64')).astype('float32')
            assert np.allclose(p,independent,atol=2e-7,rtol=2e-7) and np.array_equal(p.argmax(1),independent.argmax(1))
            folder=target/phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1));row={'phase':phase,'metrics':metrics(frame.iloc[ids].label,p)};dump(folder/'metrics.json',row)
            control=OUT/'m6'/phase;control.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(control/'outer_predictions.npz',indices=ids,p=m6,pred=m6.argmax(1))
            dump(control/'metrics.json',{'phase':phase,'metrics':metrics(frame.iloc[ids].label,m6)})
            if phase=='validation':validation=row['metrics']
            else:
                np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append(row)
            checked.append({'candidate':arm['name'],'phase':phase,'images':len(ids),'probability_sha256':sha(folder/'predictions.npz'),'maximum_independent_probability_error':float(np.max(np.abs(p-independent)))})
        record={'name':arm['name'],'candidate_id':arm['name'],'blend_configuration':arm,'val':validation,
            'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))},
            'new_seed_fits':0,'reused_neural_heads_per_fold':9,'paired_original_oof_vs_m6':paired_oof(OUT/'m6',target,frame)}
        record['development_gate_passed']=bool(record['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and validation['macro_f1']>=protocol['minimum_validation_macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_large_blend_candidate','candidate':arm,'val_f1':validation['macro_f1'],'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    assert len(checked)==32
    # Independently reconstruct every reported score from its physical saved probabilities.
    for record in records:
        scores=[];seen=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            saved=np.load(OUT/record['name']/phase/'predictions.npz');ids=stop if phase=='validation' else outer
            assert np.array_equal(saved['indices'],ids) and np.array_equal(saved['pred'],saved['p'].argmax(1))
            actual=metrics(frame.iloc[ids].label,saved['p']);expected=record['val'] if phase=='validation' else next(r['metrics'] for r in record['cv']['folds'] if r['phase']==phase)
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-expected[key])<1e-12
            if phase!='validation':scores.append(actual['macro_f1']);seen.extend(ids.tolist())
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
    verify_frozen(protocol);eligible=[r for r in records if r['development_gate_passed']]
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':0,'reused_source_heads':72,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'actual_seed_heads_checked':0,'reused_source_heads_checked':72,'fixed_candidate_scores':8,'fixed_mixture_phases_checked':checked,'new_seed_fits':0,
        'checks':['Freeze8mixturesbeforeevaluation;4cachedlargearms xweights.1/.25withM6;0newmodelfits',
            'All24sourcepredictionarrays and72parentheadhashes protected; priorformalheads independentlyaudited',
            'All32mixturephasesindependentlyreconstructedwiththreeseparatefloat64components,classdecisions/scoresandouterpatientcoverageverified;nodevelopmentexternalsused']})
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'source_prediction_sha256':protocol['source_prediction_sha256'],'source_checkpoint_sha256':protocol['source_checkpoint_sha256'],'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'frozen_artifacts':protocol['frozen_artifacts'],'source_prediction_sha256':protocol['source_prediction_sha256'],'source_checkpoint_sha256':protocol['source_checkpoint_sha256'],'all_code_and_source_hashes_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':0,'reused_source_heads':72,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

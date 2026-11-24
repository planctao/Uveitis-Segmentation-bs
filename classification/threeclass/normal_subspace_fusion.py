"""Four frozen probability mixtures of audited M5 and normal-subspace models."""
import fcntl
import json
import os
from pathlib import Path
import traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from experiment import ROOT,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof

OUT=ROOT/'runs/normal_subspace_fusion'
DEST=ROOT/'results/research_normal_subspace_fusion'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in protocol['frozen_artifacts'].items():assert sha(path)==expected


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    source=ROOT/'results/research_normal_subspace'
    assert json.loads((source/'verification.json').read_text())['passed']
    assert json.loads((source/'freeze_integrity_after_training.json').read_text())['passed']
    comparison=json.loads((source/'comparison.json').read_text())
    parents={name:next(r for r in comparison['candidates'] if r['name']==name) for name in protocol['parents']}
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');assert sha(ROOT/'metadata/manifest.csv')==protocol['manifest_sha256']
    records=[];checked=[]
    for weight in protocol['subspace_weights']:
        name=f'normal_full_weight_{str(weight).replace(".","p")}';target=OUT/name;folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            ids=stop if phase=='validation' else outer
            a=np.load(ROOT/'runs/normal_subspace'/protocol['parents'][0]/phase/'predictions.npz')
            b=np.load(ROOT/'runs/normal_subspace'/protocol['parents'][1]/phase/'predictions.npz')
            assert np.array_equal(a['indices'],ids) and np.array_equal(b['indices'],ids)
            assert a['p'].shape==b['p'].shape==(len(ids),3)
            assert np.isfinite(a['p']).all() and np.isfinite(b['p']).all()
            p=((1-weight)*a['p']+weight*b['p']).astype('float32')
            independent=np.stack([np.add((1-weight)*a['p'][:,i].astype('float64'),weight*b['p'][:,i].astype('float64')) for i in range(3)],1)
            assert np.allclose(p,independent,atol=1e-7,rtol=1e-7) and np.array_equal(p.argmax(1),independent.argmax(1))
            assert np.allclose(p.sum(1),1,atol=1e-6)
            folder=target/phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            row={'phase':phase,'metrics':metrics(frame.iloc[ids].label,p)};dump(folder/'metrics.json',row)
            if phase=='validation':validation=row['metrics']
            else:
                np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append(row)
            checked.append({'weight':weight,'phase':phase,'images':len(ids),'source_probability_hashes':[sha(ROOT/'runs/normal_subspace'/arm/phase/'predictions.npz') for arm in protocol['parents']]})
        record={'candidate_id':name,'name':name,'subspace_weight':weight,'val':validation,
            'base_config':parents[protocol['parents'][0]]['config'],'subspace_config':parents[protocol['parents'][1]]['config'],
            'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
                'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))}}
        record['paired_original_oof_vs_reference']=paired_oof(ROOT/'runs/normal_subspace'/protocol['parents'][0],target,frame)
        record['development_gate_passed']=bool(record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and validation['macro_f1']>=reference['selected']['val']['macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_subspace_fusion_complete','weight':weight,'val_f1':validation['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    assert len(checked)==16
    verify_frozen(protocol)
    dump(DEST/'comparison.json',{'protocol':protocol,'candidates':records,'new_seed_fits':0,'reused_parent_fits':24,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'mixture_phases_checked':checked,'new_seed_fits':0,
        'checks':['Previouslyauditedparentmodelsandprobabilityfilesallfrozenbeforeanyfourmixtureevaluations',
            'Everydevelopmentrowidentity,finiteprobabilityandfloat64independentmixtureclassdecisionchecked',
            'OriginalpatientCVcoverageandpairedpatientbootstrap; noexternaloutcomes']})
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'frozen_parent_artifacts':protocol['frozen_artifacts'],
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_fusion.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_artifacts':protocol['frozen_artifacts'],'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':0,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

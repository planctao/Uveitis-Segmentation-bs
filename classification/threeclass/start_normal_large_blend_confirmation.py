"""Freeze a selected normal-L/M6 mixture before two unused patient repeats."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan


def main():
    family='normal_large_blend_confirmation';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    result=ROOT/'results/research_normal_large_blend'
    selection_record=result/'selection_before_repeat.json'
    selection=json.loads(selection_record.read_text());selected=selection['candidate']
    comparison=json.loads((result/'comparison.json').read_text());verified=json.loads((result/'verification.json').read_text())
    assert verified['passed'] and verified['actual_seed_heads_checked']==0 and verified['reused_source_heads_checked']==72
    assert len(verified['fixed_mixture_phases_checked'])==32
    assert json.loads((ROOT/'runs/normal_large_blend/state.json').read_text())['status']=='completed'
    assert selected==max([r for r in comparison['candidates'] if r['development_gate_passed']],key=lambda r:r['cv']['mean_macro_f1'])
    assert selected['cv']['mean_macro_f1']>.8270827413505814 and selected['val']['macro_f1']>=.8904827120381046
    for key in ['source_prediction_sha256','source_checkpoint_sha256']:
        for path,expected in selection[key].items():assert sha(path)==expected
    arm=selected['blend_configuration'];weight=arm['large_weight'];assert weight in [.1,.25]
    choice={'blend_arm':arm,'new_branch_weight':weight,'m6_weight':1-weight,
        'reference_weight':.25*(1-weight),'normal_base_weight':.75*(1-weight)}
    repeats=[20280701,20280711];old_protocols=sorted(ROOT.glob('runs/*/protocol.json'))
    for p in old_protocols:assert all(str(seed) not in p.read_text() for seed in repeats),f'Previously used split seed in {p}'
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    original=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[original];plans=[]
    for ri,repeat in enumerate(repeats):
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=repeat);seen=[]
        for fold,(inner_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
            possible=frame.iloc[original[inner_local]];outer=original[outer_local]
            groups=possible.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
            inner,stopping=train_test_split(groups.index,test_size=.2,random_state=20280724+ri*10+fold,stratify=groups)
            fit=np.zeros(len(frame),bool);stop=fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]]=True;stop[possible.index[possible.split_group.isin(stopping)]]=True
            check_plan(frame,fit,stop,outer,original);seen.extend(outer.tolist())
            assert frame.loc[fit&frame.label.eq(2)].patient_group.nunique()>32
            path=out/'split_plans'/f'repeat_{repeat}_fold_{fold}.json'
            dump(path,{'repeat':repeat,'fold':fold,'manifest_sha256':digest,'inner_split_seed':20280724+ri*10+fold,
                'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist(),'outer_indices':outer.tolist()});plans.append(path)
        assert sorted(seen)==original.tolist()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw=reference['selected']['config'];normal={**raw,'normal_rank':32,'normal_strength':1.}
    large_cfg={**normal,'normal_large_feature':arm['feature']}
    source_protocol=json.loads((ROOT/'runs/normal_large_blend/protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts','source_prediction_sha256','source_checkpoint_sha256']:
        for path,expected in source_protocol[key].items():assert sha(ROOT/path if not Path(path).is_absolute() else path)==expected
    artifacts=[selection_record,ROOT/'metadata/manifest.csv',ROOT/'results/milestone5/selection_before_test.json',
        ROOT/'results/milestone6/package.json',ROOT/'results/milestone6/selection_before_test.json',
        ROOT/'runs/normal_large_blend/protocol.json',ROOT/'runs/normal_large_reference/protocol.json',
        ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json',ROOT/'verification_normal_large_reference.json']
    large_source=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts += [Path(large_source['feature_storage']),Path(large_source['feature_storage']).with_suffix('.json'),ROOT/'runs/dinov3_large/protocol.json']
    artifacts += [result/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [Path(path) for path in source_protocol['source_checkpoint_sha256']]
    artifacts += [Path(path) for path in source_protocol['source_prediction_sha256']]
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            path=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(path)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['normal_large_blend_confirmation.py','start_normal_large_blend_confirmation.py','normal_large_reference.py','dinov3_large_data.py',
        'normal_large_reference_experiments.py','audit_normal_large.py','audit_normal_tangent.py',
        'normal_subspace.py','normal_subspace_experiments.py','verify_normal_subspace.py','experiment.py',
        'repeat_patient_cv.py','resolution_repeat_cv.py','kernel_residual_experiments.py',
        'patient_gaussian.py','audit_gaussian_evidence.py','repeated_oof.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest,
        'selection_record':str(selection_record.relative_to(ROOT)),'selected_candidate':selected['candidate_id'],
        'selected_configuration':choice,'configurations':{'reference':raw,'subspace':normal,'large':large_cfg},
        'outer_split_seeds':repeats,'inner_split_rule':'20280724+10*repeat_index+fold;20% of possible fitting patient/near-duplicate groups',
        'fresh_seed_audit':{str(p):sha(p) for p in old_protocols},'split_plan_sha256':{str(p):sha(p) for p in plans},
        'arms':['reference','subspace','large','fixed_gaussian','m6','candidate'],'planned_new_seed_fits':54,'planned_generative_fits':18,
        'hypothesis':'Confirm the highest qualifying development-CV fixed normal-L/M6 mixture. Refit raw B, normal B and selected normal L independently in six fresh patient plans, three seeds per branch; retain the selected feature layout and all three mixture weights.',
        'reference_comparators':'Fresh M5; fixed .9M5+.1patient-balanced coarseGaussian,shrinkage.1,temperature sqrt256; M6=.25M5+.75normalB.',
        'confirmation_gate':'Mean6fold Macro-F1 strictly higher than EACH of M6/M5/fixedGaussian AND repeated paired patient bootstrap95% Macro-F1 lower endpoint>0 versus EACH. No automatic external testing.',
        'external_test_policy':'Only original1717 trainingimages form fit/stop/outer. Originalvalidation unused for confirmation. External feature rows NaN; no external outcomes or predictions loaded.',
        'audit':'Freeze all6patientplans before anyfit; freeze two distinct normal PCA geometries before each fold nine heads. Independently rebuild54heads,18large fulltask witnesses,all stopping/outer predictions,12normalspaces,18Gaussian moments and the nested three-component candidate mixture.',
        'frozen_artifacts':{str(p):sha(p) for p in artifacts},'code_sha256':{name:sha(ROOT/name) for name in code}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_large_blend_confirmation.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_large_blend_confirmation.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':54,'planned_generative_fits':18,'outer_split_seeds':repeats}),flush=True)


if __name__=='__main__':main()

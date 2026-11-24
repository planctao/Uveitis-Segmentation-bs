"""Freeze one selected mixture and two unused patient partitions before confirmation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan


def main():
    family='axis_mixture_confirmation';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    selection_record=ROOT/'results/research_axis_reference_mixtures/selection_before_repeat.json'
    selected=json.loads(selection_record.read_text())['candidate']
    assert selected['development_gate_passed']
    choice=selected['configuration'];assert choice['arm']=='gaussian_scale10' and choice['augmented_weight']==.1
    repeats=[20271101,20271111]
    old_protocols=sorted(ROOT.glob('runs/*/protocol.json'))
    for p in old_protocols:
        assert all(str(seed) not in p.read_text() for seed in repeats),f'Previously used split seed in {p}'
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    original=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[original];plans=[]
    for ri,repeat in enumerate(repeats):
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=repeat)
        seen=[]
        for fold,(inner_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
            possible=frame.iloc[original[inner_local]];outer=original[outer_local]
            groups=possible.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
            inner,stopping=train_test_split(groups.index,test_size=.2,random_state=20271124+ri*10+fold,stratify=groups)
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]]=True
            stop[possible.index[possible.split_group.isin(stopping)]]=True
            check_plan(frame,fit,stop,outer,original);seen.extend(outer.tolist())
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique()>32
            plan={'repeat':repeat,'fold':fold,'manifest_sha256':digest,'inner_split_seed':20271124+ri*10+fold,
                'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist(),'outer_indices':outer.tolist()}
            path=out/'split_plans'/f'repeat_{repeat}_fold_{fold}.json';dump(path,plan);plans.append(path)
        assert sorted(seen)==original.tolist()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw=reference['selected']['config'];normal={**raw,'normal_rank':32,'normal_strength':1.}
    augmented={**normal,'normal_axis_mode':'gaussian','normal_axis_scale':.1}
    artifacts=[selection_record,ROOT/'results/milestone5/selection_before_test.json',ROOT/'results/milestone6/package.json']
    for research in ['axis_reference_mixtures','normal_axis_augmentation']:
        artifacts += [ROOT/'results'/('research_'+research)/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    source_protocol=json.loads((ROOT/'runs/axis_reference_mixtures/protocol.json').read_text())
    artifacts += [Path(p) for p in source_protocol['source_checkpoints']]
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            p=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(p)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(p)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['axis_mixture_confirmation.py','start_axis_mixture_confirmation.py','normal_subspace.py',
        'normal_subspace_experiments.py','verify_normal_subspace.py','normal_axis_augmentation.py','normal_axis_experiments.py',
        'experiment.py','repeat_patient_cv.py','resolution_repeat_cv.py','kernel_residual_experiments.py',
        'patient_gaussian.py','audit_gaussian_evidence.py','repeated_oof.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest,
        'selection_record':str(selection_record.relative_to(ROOT)),'selected_candidate':selected['candidate_id'],
        'selected_configuration':choice,'configurations':{'reference':raw,'subspace':normal,'augmented':augmented},
        'outer_split_seeds':repeats,'inner_split_rule':'20271124+10*repeat_index+fold;20% of possible fitting patient/near-duplicate groups',
        'fresh_seed_audit':{str(p):sha(p) for p in old_protocols},
        'split_plan_sha256':{str(p):sha(p) for p in plans},
        'arms':['reference','subspace','augmented','fixed_gaussian','m6','candidate'],
        'planned_new_seed_fits':54,'planned_generative_fits':18,
        'hypothesis':'Confirm the frozen .9 M6+.1 small-Gaussian-axis-augmented normal branch after original CV and validation gates. Retrain raw M5, full normal-reference and noise-augmented reference independently within all6 new fit/stop/outer plans;3seeds each. Do not alter augmentation or mixture after results.',
        'reference_comparators':'Fresh M5; fixed.9M5+.1patient-balanced coarseGaussian with shrinkage.1,temperature sqrt256; and M6=.25M5+.75normal. Same raw source reused in all controls and mixtures, no extra fit claims.',
        'confirmation_gate':'Mean6fold Macro-F1 strictly higher than EACH of M6/M5/fixedGaussian and repeated paired patient-group bootstrap95CI Macro-F1 lower endpoint>0 versus EACH. Only then eligible for frozen inference review. No automatic external testing.',
        'external_test_policy':'Only original1717training images form fit/stop/outer. Original validation is not reused for new fitting or confirmation selection. External feature rows setNaN; no external outcomes or predictions loaded.',
        'audit':'Freeze6patient/duplicate plans before any fit. For everyfold freeze equal-normal-patient PCA, banks and provenance before its9heads. Check54actual selected checkpoints, all stop/outer probabilities through actual models and independent NumPy projection+plainM5,18first-training-batch augmentation witnesses,6normalPCA spans and18Gaussian patient moments. Verify complete outer coverage, M6/candidate mixtures and paired repeated comparisons.',
        'frozen_artifacts':{str(p):sha(p) for p in artifacts},'code_sha256':{name:sha(ROOT/name) for name in code}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'axis_mixture_confirmation.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'axis_mixture_confirmation.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'pid':p.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':54,'planned_generative_fits':18,'outer_split_seeds':repeats}),flush=True)


if __name__=='__main__':main()

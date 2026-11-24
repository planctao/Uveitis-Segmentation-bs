"""Freeze the selected group-risk mixture and two unused patient split repeats."""
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
    family='normal_group_risk_confirmation';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    selection_record=ROOT/'results/research_normal_group_risk/selection_before_repeat.json'
    selection=json.loads(selection_record.read_text());selected=selection['candidate']
    comparison=json.loads((ROOT/'results/research_normal_group_risk/comparison.json').read_text())
    verified=json.loads((ROOT/'results/research_normal_group_risk/verification.json').read_text())
    assert verified['passed'] and verified['actual_seed_heads_checked']==verified['actual_training_witnesses_checked']==48
    assert selected==max([r for r in comparison['candidates'] if r['development_gate_passed']],key=lambda r:r['cv']['mean_macro_f1'])
    assert selected['cv']['mean_macro_f1']>.8270827413505814 and selected['val']['macro_f1']>=.8904827120381046
    assert selected['source_risk_arm']=='groups2_risk0p1' and selected['subspace_weight']==.75
    for path,expected in selection['checkpoint_sha256'].items():assert sha(path)==expected
    for path,expected in selection['geometry_and_plans_sha256'].items():assert sha(path)==expected
    choice={'risk_arm':selected['risk_configuration'],'new_branch_weight':.75,'reference_weight':.25}
    repeats=[20271201,20271211];old_protocols=sorted(ROOT.glob('runs/*/protocol.json'))
    for p in old_protocols:assert all(str(seed) not in p.read_text() for seed in repeats),f'Previously used split seed in {p}'
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    original=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[original];plans=[]
    for ri,repeat in enumerate(repeats):
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=repeat);seen=[]
        for fold,(inner_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
            possible=frame.iloc[original[inner_local]];outer=original[outer_local]
            groups=possible.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
            inner,stopping=train_test_split(groups.index,test_size=.2,random_state=20271224+ri*10+fold,stratify=groups)
            fit=np.zeros(len(frame),bool);stop=fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]]=True;stop[possible.index[possible.split_group.isin(stopping)]]=True
            check_plan(frame,fit,stop,outer,original);seen.extend(outer.tolist())
            assert frame.loc[fit&frame.label.eq(2)].patient_group.nunique()>32
            path=out/'split_plans'/f'repeat_{repeat}_fold_{fold}.json'
            dump(path,{'repeat':repeat,'fold':fold,'manifest_sha256':digest,'inner_split_seed':20271224+ri*10+fold,
                'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist(),'outer_indices':outer.tolist()});plans.append(path)
        assert sorted(seen)==original.tolist()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw=reference['selected']['config'];normal={**raw,'normal_rank':32,'normal_strength':1.}
    risk={**normal,'normal_risk_groups':2,'normal_risk_weight':.1};assert risk==selected['config']
    source_protocol=json.loads((ROOT/'runs/normal_group_risk/protocol.json').read_text())
    proof_path=ROOT/'verification_normal_group_risk.json';proof=json.loads(proof_path.read_text());assert proof['passed']
    for name,expected in source_protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected
    artifacts=[selection_record,ROOT/'results/milestone5/selection_before_test.json',ROOT/'results/milestone6/package.json',
        ROOT/'runs/normal_group_risk/protocol.json',proof_path,ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json']
    for research in ['normal_group_risk','normal_atlas']:
        artifacts += [ROOT/'results'/('research_'+research)/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [Path(p) for p in source_protocol['source_checkpoints']]
    for row in verified['fits']:
        folder=ROOT/'runs/normal_group_risk'/row['arm']/row['phase']/f'seed_{row["seed"]}'
        for name,key in [('model.pt','checkpoint_sha256'),('training_input_witness.npz','training_witness_sha256')]:
            assert sha(folder/name)==row[key];artifacts.append(folder/name)
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            path=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(path)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['normal_group_risk_confirmation.py','start_normal_group_risk_confirmation.py','normal_group_risk.py',
        'normal_group_risk_experiments.py','audit_normal_group_risk.py','audit_normal_tangent.py','normal_atlas.py',
        'normal_subspace.py','normal_subspace_experiments.py','verify_normal_subspace.py','experiment.py',
        'repeat_patient_cv.py','resolution_repeat_cv.py','kernel_residual_experiments.py',
        'patient_gaussian.py','audit_gaussian_evidence.py','repeated_oof.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest,
        'selection_record':str(selection_record.relative_to(ROOT)),'selected_candidate':selected['candidate_id'],
        'selected_configuration':choice,'configurations':{'reference':raw,'subspace':normal,'risk':risk},
        'outer_split_seeds':repeats,'inner_split_rule':'20271224+10*repeat_index+fold;20% of possible fitting patient/near-duplicate groups',
        'fresh_seed_audit':{str(p):sha(p) for p in old_protocols},'split_plan_sha256':{str(p):sha(p) for p in plans},
        'arms':['reference','subspace','risk','fixed_gaussian','m6','candidate'],'planned_new_seed_fits':54,'planned_generative_fits':18,
        'hypothesis':'Confirm the frozen .75 group-risk/.25 M5 mixture selected on original patientCV and validation. Retrain rawM5,fullnormalreference,and selectedgroup-riskMLP independently within sixnewpatient fit/stop/outerplans,3seeds each. Keep groupgeometry,riskweight and probability mixture fixed.',
        'reference_comparators':'Fresh M5; fixed.9M5+.1patient-balanced coarseGaussian withshrinkage.1,temperature sqrt256; M6=.25M5+.75normal. Shared raw source reused without additional fit claims.',
        'confirmation_gate':'Mean6fold Macro-F1 strictlyhigher thanEACHofM6/M5/fixedGaussian AND repeatedpairedpatientbootstrap95%Macro-F1lowerendpoint>0versusEACH. Noautomaticexternaltesting.',
        'external_test_policy':'Only original1717 trainingimages form fit/stop/outer. Originalvalidation unused for confirmation fitting or selection. External feature rows NaN; no external outcomes/predictions loaded.',
        'audit':'Freeze all6patient/near-duplicate plans beforeanyfit. Beforeeachfold9heads freeze normalpatientPCA andgroupgeometry. Independentlycheck54selectedheads,allstop/outerprobabilities,18group-risk firsttaskwitnesses,allnormal/groupbuffers,6normalspaces and18Gaussianpatientmoments. Rebuild M6/candidate mixtures,outercoverage andjointrepeatedpairedpatient comparisons.',
        'frozen_artifacts':{str(p):sha(p) for p in artifacts},'code_sha256':{name:sha(ROOT/name) for name in code}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_group_risk_confirmation.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_group_risk_confirmation.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':54,'planned_generative_fits':18,'outer_split_seeds':repeats}),flush=True)


if __name__=='__main__':main()

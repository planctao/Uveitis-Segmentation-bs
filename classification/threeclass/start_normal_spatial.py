"""Freeze the normal-reference spatial reference study and launch its fixed48head protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    family='normal_spatial';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    verified=ROOT/'verification_normal_spatial.json';proof=json.loads(verified.read_text());assert proof['passed']
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected
    previous=json.loads((ROOT/'runs/normal_decorrelation/state.json').read_text());assert previous['status']=='completed' and previous['passing_development_gates']==0
    assert json.loads((ROOT/'results/research_normal_decorrelation/verification.json').read_text())['passed']
    parent=json.loads((ROOT/'runs/normal_moment_geometry/protocol.json').read_text())
    hypothesis=ROOT/'results/next_normal_spatial_hypothesis.json';planned=json.loads(hypothesis.read_text())
    artifacts=[ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json',
        ROOT/'metadata/manifest.csv',ROOT/'results/milestone6/package.json']
    artifacts += [ROOT/'results/research_normal_group_risk_confirmation'/n for n in ['comparison.json','verification.json','freeze_integrity_after_confirmation.json']]
    artifacts += [ROOT/'results/research_normal_group_risk'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/normal_spatial_diagnostics.json',ROOT/'normal_spatial_diagnostics.py',ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt']
    artifacts += [ROOT/'results/research_normal_decorrelation'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_normal_atlas'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_normal_evidence_attention'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_fine_normal_reference'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_normal_tangent_adversarial'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz' for phase in ['validation','fold_0','fold_1','fold_2']]
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            path=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(path)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['normal_spatial_reference.py','normal_spatial_experiments.py','verify_normal_spatial.py',
        'start_normal_spatial.py','audit_normal_spatial.py','audit_normal_tangent.py','verify_normal_subspace.py','normal_subspace.py','experiment.py','kernel_residual_experiments.py','milestone2.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':parent['manifest_sha256'],
        'reference':parent['reference'],'reference_sha256':parent['reference_sha256'],'preflight_sha256':sha(verified),
        'source_checkpoints':parent['source_checkpoints'],'frozen_parent_artifacts':{str(p):sha(p) for p in artifacts},
        'prior_hypothesis_sha256':sha(hypothesis),'prior_hypothesis':planned,
        'configuration_plan':planned['configuration_plan'],'planned_new_seed_fits':48,'predeclared_candidate_scores':8,
        'fixed_fusion_weight':.75,'minimum_cv_macro_f1':.8270827413505814,'minimum_validation_macro_f1':.8904827120381046,
        'hypothesis':planned['hypothesis'],'geometry':planned['geometry'],'architecture':planned['architecture'],
        'geometry_freeze':'All16 arm/phase normal patient geometries and16 patient plans hashed before all MLP fits.',
        'selection':planned['selection'],'audit':planned['audit'],
        'external_test_policy':'Only2091 development rows; external feature arrays NaN before fitting. No external outcomes/predictions loaded for selection.',
        'code_sha256':{name:sha(ROOT/name) for name in code}}
    assert proof['manifest_sha256']==sha(ROOT/'metadata/manifest.csv')==protocol['manifest_sha256']
    for name,expected in protocol['source_checkpoints'].items():assert sha(name)==expected
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_spatial.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_spatial_experiments.py')],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':str((out/'protocol.json').relative_to(ROOT)),
        'protocol_sha256':sha(out/'protocol.json'),'pid':process.pid,'formal_fits_started':'Inspect actual checkpoints'})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48,'external_test_evaluated':False}),flush=True)


if __name__=='__main__':main()

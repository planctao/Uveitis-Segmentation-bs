"""Freeze the centered spatial contrast study before its 48 actual head fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    family='large_spatial_contrast';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    verified=ROOT/'verification_large_spatial_contrast.json';proof=json.loads(verified.read_text());assert proof['passed']
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected
    previous=json.loads((ROOT/'runs/normal_large_reference_confirmation/state.json').read_text());assert previous['status']=='completed' and not previous['confirmation_gate_passed']
    assert json.loads((ROOT/'results/research_normal_large_reference_confirmation/verification.json').read_text())['passed']
    parent=json.loads((ROOT/'runs/normal_moment_geometry/protocol.json').read_text())
    hypothesis=ROOT/'results/next_large_spatial_contrast_hypothesis.json';planned=json.loads(hypothesis.read_text())
    artifacts=[ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json',ROOT/'metadata/manifest.csv',ROOT/'results/milestone6/package.json',
        ROOT/'results/normal_branch_diagnostics.json',ROOT/'normal_branch_diagnostics.py',ROOT/'runs/normal_large_reference/base_dual_large_dual/validation/seed_0/model.pt',ROOT/'verification_large_spatial_contrast.json']
    large=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts += [ROOT/'runs/dinov3_large/protocol.json',Path(large['feature_storage']),Path(large['feature_storage']).with_suffix('.json')]
    for family in ['normal_large_reference','normal_large_reference_confirmation']:
        artifacts += [ROOT/'results'/('research_'+family)/n for n in ['comparison.json','verification.json']]
    source_heads=dict(parent['source_checkpoints'])
    for phase in ['validation','fold_0','fold_1','fold_2']:
        for seed in range(3):
            path=ROOT/'runs/dinov3_large/base_dual_large_dual'/phase/f'seed_{seed}/model.pt'
            source_heads[str(path)]=sha(path)
    for prior in ['normal_spatial_consistency','normal_spatial','normal_group_risk_confirmation','normal_coordinate_confirmation','normal_task_gradient_confirmation']:
        for name in ['comparison.json','verification.json']:
            artifacts.append(ROOT/'results'/('research_'+prior)/name)
    artifacts += [ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz' for phase in ['validation','fold_0','fold_1','fold_2']]
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            path=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(path)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['normal_spatial_contrast.py','large_spatial_contrast_experiments.py','verify_large_spatial_contrast.py',
        'start_large_spatial_contrast.py','audit_normal_spatial_contrast.py','dinov3_large_data.py','audit_normal_tangent.py','verify_normal_subspace.py',
        'normal_subspace.py','experiment.py','kernel_residual_experiments.py','milestone2.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':parent['manifest_sha256'],
        'reference':parent['reference'],'reference_sha256':parent['reference_sha256'],'preflight_sha256':sha(verified),
        'source_checkpoints':source_heads,'frozen_parent_artifacts':{str(p):sha(p) for p in artifacts},
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
    with (ROOT/'large_spatial_contrast.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'large_spatial_contrast_experiments.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':str((out/'protocol.json').relative_to(ROOT)),
        'protocol_sha256':sha(out/'protocol.json'),'pid':process.pid,'formal_fits_started':'Inspect actual checkpoints'})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48,'external_test_evaluated':False}),flush=True)


if __name__=='__main__':main()

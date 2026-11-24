"""Freeze independent/shared normal cross-backbone geometry before scoring."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    out=ROOT/'runs/normal_cross_geometry';storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/normal_cross_geometry')
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    current=json.loads((ROOT/'results/current_research_reference.json').read_text());assert current['retained_development_model']=='M7'
    for key in ['package','selection','inference_verification']:assert sha(ROOT/current[key])==current[key+'_sha256']
    parent=ROOT/'results/research_normal_large_interaction';state=json.loads((ROOT/'runs/normal_large_interaction/state.json').read_text())
    assert state['status']=='completed' and state['passing_development_gates']==0
    verification=json.loads((parent/'verification.json').read_text());assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==48
    assert not any(r['development_gate_passed'] for r in json.loads((parent/'comparison.json').read_text())['candidates'])
    proof_path=ROOT/'verification_normal_cross_geometry.json';proof=json.loads(proof_path.read_text());assert proof['passed']
    assert len(proof['cases'])==4 and all(row['batch_size']==256 and row['actual_cuda_full_task_witness']['full_task_restored'] for row in proof['cases'])
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected
    hypothesis=ROOT/'results/next_normal_cross_geometry_hypothesis.json';planned=json.loads(hypothesis.read_text())
    reference=ROOT/'results/milestone5/selection_before_test.json';large=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts=[ROOT/'metadata/manifest.csv',reference,ROOT/'results/milestone6/package.json',ROOT/'results/milestone6/selection_before_test.json',
        ROOT/'results/current_research_reference.json',ROOT/current['package'],ROOT/current['selection'],ROOT/current['inference_verification'],
        ROOT/'features/multiscale_vit.npz',ROOT/'features/multiscale_vit.json',Path(large['feature_storage']),Path(large['feature_storage']).with_suffix('.json'),
        ROOT/'runs/dinov3_large/protocol.json',ROOT/'runs/normal_large_residual/protocol.json',ROOT/'runs/residual_m7_confirmation/protocol.json',ROOT/'runs/normal_large_interaction/protocol.json',proof_path]
    artifacts += [parent/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    for phase in ['validation','fold_0','fold_1','fold_2']:
        artifacts.append(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
        for branch in ['zero_control','normal_rank32_full']:
            artifacts += [ROOT/'runs/normal_subspace'/branch/phase/f'seed_{seed}/model.pt' for seed in range(3)]
        artifacts += [ROOT/'runs/normal_large_reference/large_dual'/phase/f'seed_{seed}/model.pt' for seed in range(3)]
    m7=json.loads((ROOT/current['package']).read_text())
    for seed in range(3):
        for branch in ['reference','subspace','large']:
            path=ROOT/'results/milestone7_normal_large_blend/frozen_package'/f'seed_{seed}'/(branch+'.pt')
            assert sha(path)==m7['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    code=['normal_cross_geometry.py','audit_normal_cross_geometry.py','verify_normal_cross_geometry.py',
        'normal_cross_geometry_experiments.py','start_normal_cross_geometry.py','normal_large_reference.py',
        'normal_large_residual.py','normal_large_residual_experiments.py','normal_subspace.py','verify_normal_subspace.py',
        'audit_normal_tangent.py','experiment.py','dinov3_large_data.py','kernel_residual_experiments.py','milestone2.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'reference':str(reference.relative_to(ROOT)),'retained_model':'M7','configuration_plan':planned['configuration_plan'],
        'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
        'development_plan':'Original train/validation, and original patient3fold seed20261125 with stopping-group seeds20261124+fold. All16 separate independent/shared B/L geometries and16plans frozen before anyhead fit.',
        'hypothesis':planned['hypothesis'],'geometry':planned['geometry'],'architecture':planned['architecture'],'planned_new_seed_fits':48,'predeclared_scores':8,
        'fixed_fusions':'Each branch and fixed .75newbranch+.25M5; no extra fits for fusion.',
        'selection':'Highest qualifying original CV with CV strictly>M7 and validation>=M7 frozen before two unusedpatient3fold repeats. Higher sixfoldmeanF1 and joint pairedpatient95%lowerendpoint>0 versus EACHofM7/M6/M5/fixedGaussian. No automatic external evaluation.',
        'audit':'Independently recover all48actualCUDA firstfulltasks, trainednormalbuffers and allstopping/outerprobabilities with NumPy separate projections+ordinaryMLP, all16normalspaces from independent Gram spectrum/singular equations, originalepochs/stats/maps and16fusionphases. Audit originalGPU validationensemble separately from CPU restoration.',
        'external_test_policy':'Only2091developmentrows; externalfeaturerowsNaN; no external predictions or outcomes loaded.',
        'frozen_artifacts':{str(path):sha(path) for path in artifacts},'code_sha256':{name:sha(ROOT/name) for name in code}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_cross_geometry.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_cross_geometry_experiments.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':'runs/normal_cross_geometry/protocol.json','protocol_sha256':sha(out/'protocol.json'),'pid':process.pid})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48}),flush=True)


if __name__=='__main__':main()

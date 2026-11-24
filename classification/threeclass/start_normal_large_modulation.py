"""Freeze four L-conditioned hidden-channel arms after the preceding study is terminal."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    previous = ROOT/'results/research_temporal_backbone_confirmation'
    state = json.loads((ROOT/'runs/temporal_backbone_confirmation/state.json').read_text())
    assert state['status'] == 'completed'
    result = json.loads((previous/'comparison.json').read_text())
    verified = json.loads((previous/'verification.json').read_text())
    assert verified['passed'] and verified['actual_seed_heads_checked'] == 72
    assert verified['actual_training_witnesses_checked'] == 54
    assert json.loads((previous/'freeze_integrity_after_confirmation.json').read_text())['passed']
    registry = ROOT/'results/research_reference_registry.json'
    reference_path = ROOT/'results/current_research_reference.json'
    if registry.exists():
        active = json.loads(registry.read_text())
        reference_path = ROOT/active['active_reference']
        assert sha(reference_path) == active['active_reference_sha256']
    current = json.loads(reference_path.read_text())
    assert current['retained_development_model'] == ('M8' if result['confirmation_gate_passed'] else 'M7')
    for key in ['package', 'selection', 'inference_verification']:
        assert sha(ROOT/current[key]) == current[key+'_sha256']
    if result['confirmation_gate_passed']:
        assert json.loads((ROOT/'results/milestone8_temporal_backbone/independent_package_audit.json').read_text())['passed']
    proof_path = ROOT/'verification_normal_large_modulation.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == len(proof['actual_cuda_full_task_witnesses']) == 4
    assert all(row['batch_size'] == 256 for row in proof['cases'])
    assert all(row['witness']['full_task_restored'] for row in proof['actual_cuda_full_task_witnesses'])
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    out = ROOT/'runs/normal_large_modulation'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/normal_large_modulation')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out/'protocol.json').exists()
    hypothesis = ROOT/'results/next_normal_large_modulation_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    reference = ROOT/'results/milestone5/selection_before_test.json'
    large = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts = [ROOT/'metadata/manifest.csv', reference, ROOT/'results/milestone6/package.json',
        ROOT/'results/milestone6/selection_before_test.json', reference_path, proof_path,
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json',
        Path(large['feature_storage']), Path(large['feature_storage']).with_suffix('.json'),
        ROOT/'runs/dinov3_large/protocol.json', ROOT/'runs/temporal_backbone_confirmation/protocol.json']
    artifacts += [ROOT/current[key] for key in ['package', 'selection', 'inference_verification']]
    artifacts += [previous/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_confirmation.json']]
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        artifacts.append(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
        for branch in ['zero_control', 'normal_rank32_full']:
            artifacts += [ROOT/'runs/normal_subspace'/branch/phase/f'seed_{seed}/model.pt' for seed in range(3)]
        artifacts += [ROOT/'runs/normal_large_reference/large_dual'/phase/f'seed_{seed}/model.pt' for seed in range(3)]
    package = json.loads((ROOT/current['package']).read_text())
    package_folder = (ROOT/current['package']).parent
    for name, expected in package['model_sha256'].items():
        branch, seed = name.rsplit('_seed', 1)
        path = package_folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
        assert sha(path) == expected; artifacts.append(path)
    code = ['normal_large_modulation.py', 'audit_normal_large_modulation.py', 'verify_normal_large_modulation.py',
        'normal_large_modulation_experiments.py', 'finalize_normal_large_modulation.py', 'start_normal_large_modulation.py',
        'normal_large_residual.py', 'normal_large_residual_experiments.py', 'normal_subspace.py',
        'verify_normal_subspace.py', 'audit_normal_tangent.py', 'experiment.py', 'dinov3_large_data.py',
        'kernel_residual_experiments.py', 'milestone2.py']
    retained = current['retained_development_model']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'reference': str(reference.relative_to(ROOT)), 'retained_reference': str(reference_path.relative_to(ROOT)),
        'retained_model': retained, 'configuration_plan': planned['configuration_plan'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'development_plan': 'Original train/validation and original patient3fold20261125; stopping seeds20261124+fold. Four separate B/L geometries and patient plans frozen before all fits.',
        'hypothesis': planned['hypothesis'], 'architecture': planned['architecture'], 'planned_new_seed_fits': 48,
        'predeclared_scores': 8, 'fixed_fusions': 'Each branch and fixed .75branch+.25rawB. No fusion fits.',
        'selection': f'Freeze highest qualifying original CV strictly>{retained} with validation>={retained}, then two unusedpatient3fold repeats. Higher sixfoldmeanF1 and pairedpatient95%lower>0 versus EACH retainedreference/M7/M6/M5/fixedGaussian as applicable. No automaticexternaltest.',
        'audit': 'Restore all48 actual CUDA fulltasks/allparametergradients independently, four separate B/L spaces, fitstats/maps/selectedepochs, allstop/outerprobabilities and16 fixedfusionphases. Independent NumPy projections and functional tanh modulation, original GPU validationensemble separately checked.',
        'external_test_policy': 'Only2091developmentrows; externalfeaturerowsNaN. No external predictions or outcomes loaded.',
        'frozen_artifacts': {str(path): sha(path) for path in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_large_modulation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'normal_large_modulation_experiments.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': 'runs/normal_large_modulation/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid, 'retained_reference': retained})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__': main()

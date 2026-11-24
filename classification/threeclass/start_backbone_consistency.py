"""Freeze complementary-view training after the completed EMA study fails gates."""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    out = ROOT/'runs/backbone_consistency'; assert not (out/'protocol.json').exists()
    current = json.loads((ROOT/'results/current_research_reference.json').read_text())
    assert current['retained_development_model'] == 'M7'
    parent = ROOT/'results/research_weight_averaging'
    state = json.loads((ROOT/'runs/weight_averaging/state.json').read_text())
    assert state['status'] == 'completed' and state['passing_development_gates'] == 0
    comparison = json.loads((parent/'comparison.json').read_text())
    verification = json.loads((parent/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_trajectories_checked'] == 12
    assert verification['actual_selected_models_checked'] == 36 and not any(r['development_gate_passed'] for r in comparison['candidates'])
    proof_path = ROOT/'verification_backbone_consistency.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == 4
    assert all(r['cuda_task']['complete_objective_and_all_parameter_gradients_restored'] for r in proof['cases'])
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    hypothesis_path = ROOT/'results/next_backbone_consistency_hypothesis.json'
    hypothesis = json.loads(hypothesis_path.read_text()); assert hypothesis['scores_evaluated'] == 0
    artifacts = [proof_path, ROOT/'metadata/manifest.csv', ROOT/'results/current_research_reference.json',
        ROOT/'results/milestone5/selection_before_test.json', ROOT/'results/milestone6/package.json', ROOT/'results/milestone6/selection_before_test.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json', ROOT/'runs/weight_averaging/protocol.json',
        ROOT/'runs/backbone_dropout/protocol.json', ROOT/'runs/backbone_dropout_confirmation/protocol.json']
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.append(path)
    large_source = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts += [Path(large_source['feature_storage']), Path(large_source['feature_storage']).with_suffix('.json'), ROOT/'runs/dinov3_large/protocol.json']
    artifacts += [parent/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_backbone_dropout'/name for name in ['comparison.json', 'verification.json']]
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        artifacts.append(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
        for branch in ['normal_subspace/zero_control', 'normal_subspace/normal_rank32_full', 'normal_large_reference/large_dual']:
            for seed in range(3): artifacts.append(ROOT/'runs'/branch/phase/f'seed_{seed}/model.pt')
    code = ['backbone_consistency.py', 'audit_backbone_consistency.py', 'verify_backbone_consistency.py',
        'backbone_consistency_experiments.py', 'start_backbone_consistency.py', 'normal_cross_geometry.py',
        'audit_normal_cross_geometry.py', 'normal_large_reference.py', 'normal_large_residual.py',
        'normal_large_residual_experiments.py', 'normal_subspace.py', 'verify_normal_subspace.py',
        'audit_normal_tangent.py', 'experiment.py', 'dinov3_large_data.py', 'kernel_residual_experiments.py', 'milestone2.py']
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/backbone_consistency')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'reference': 'results/milestone5/selection_before_test.json', 'retained_model': 'M7',
        'configuration_plan': hypothesis['configuration_plan'], 'hypothesis': hypothesis['hypothesis'], 'architecture': hypothesis['architecture'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'planned_new_seed_fits': 48, 'predeclared_scores': 8,
        'development_plan': 'Original train/validation and original patient3fold20261125, innerstop20261124+fold. All16geometrycopies and16plans frozen before anyfit; four unique geometry estimates.',
        'fixed_fusions': 'Each branch and fixed .75branch+.25rawB; no fusion training or weight tuning.',
        'training': 'Original fullview task plus lambda times leaf-weighted mean KL(detached fullcoarse teacher || each complementaryview). Weights.05/.2; structuredbackbone views vs complementaryrandomchannel controls. Masks tied over allstreams/tokens, views replay original fullview dropout under fork_rng without advancing nextsource RNG.',
        'audit': 'All48actualCUDA complete objectives and allparametergradients independently reconstructed, including detachedteacher and two shared-dropout masks; all16fit-only normalgeometries by independentGram spectra, fitstats/diseasemaps/epochs/stop/outer probabilities and16fixedfusion phases verified.',
        'selection': 'Highest qualifying originalCV strictly>M7 and validation>=M7 frozen before two unusedpatient3fold repeats. Highermean6foldF1 and jointpairedpatient95% lowerendpoint>0 versus EACH M7/M6/M5/fixedGaussian. No automaticexternaltest.',
        'external_test_policy': 'Only2091developmentrows; all externalfeature rowsNaN. No externalpredictions/outcomes loaded.',
        'frozen_artifacts': {str(p): sha(p) for p in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'backbone_consistency.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'backbone_consistency_experiments.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis_path, {**hypothesis, 'status': 'running', 'preflight': 'verification_backbone_consistency.json',
        'protocol': 'runs/backbone_consistency/protocol.json', 'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__': main()

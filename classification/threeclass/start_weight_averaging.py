"""Launch only after the frozen backbone-dropout confirmation fails all gates."""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    out = ROOT/'runs/weight_averaging'; assert not (out/'protocol.json').exists()
    state = json.loads((ROOT/'runs/backbone_dropout_confirmation/state.json').read_text())
    assert state['status'] == 'completed' and not state['confirmation_gate_passed']
    confirm = ROOT/'results/research_backbone_dropout_confirmation'
    check = json.loads((confirm/'verification.json').read_text())
    assert check['passed'] and check['actual_seed_heads_checked'] == 72 and check['actual_training_witnesses_checked'] == 36
    assert not json.loads((confirm/'comparison.json').read_text())['confirmation_gate_passed']
    current = json.loads((ROOT/'results/current_research_reference.json').read_text())
    assert current['retained_development_model'] == 'M7'
    hypothesis_path = ROOT/'results/next_weight_averaging_hypothesis.json'
    hypothesis = json.loads(hypothesis_path.read_text())
    assert hypothesis['scores_evaluated'] == 0 and len(hypothesis['configuration_plan']) == 6
    preflight_path = ROOT/'verification_weight_averaging.json'; preflight = json.loads(preflight_path.read_text())
    assert preflight['passed'] and preflight['actual_cuda_complete_task']['batch_size'] == 256
    for name, expected in preflight['code_sha256'].items(): assert sha(ROOT/name) == expected
    artifacts = [preflight_path, ROOT/'results/diagnostic_training_trajectory.json', ROOT/'metadata/manifest.csv',
        ROOT/'results/current_research_reference.json', ROOT/'results/milestone5/selection_before_test.json',
        ROOT/'results/milestone6/package.json', ROOT/'results/milestone6/selection_before_test.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json', ROOT/'runs/backbone_dropout_confirmation/protocol.json']
    artifacts += [confirm/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_confirmation.json']]
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.append(path)
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        for branch in ['normal_subspace/zero_control', 'normal_subspace/normal_rank32_full', 'normal_large_reference/large_dual']:
            base = ROOT/'runs'/branch/phase; artifacts.append(base/'predictions.npz')
            for seed in range(3):
                artifacts.append(base/f'seed_{seed}/model.pt')
                if branch == 'normal_subspace/normal_rank32_full': artifacts.append(base/f'seed_{seed}/history.json')
    code = ['weight_averaging.py', 'trajectory_train.py', 'audit_weight_averaging.py', 'verify_weight_averaging.py',
        'weight_averaging_experiments.py', 'start_weight_averaging.py', 'experiment.py', 'normal_subspace.py',
        'normal_subspace_experiments.py', 'verify_normal_subspace.py', 'audit_normal_tangent.py',
        'normal_large_residual_experiments.py', 'kernel_residual_experiments.py']
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/weight_averaging')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'reference': 'results/milestone5/selection_before_test.json', 'retained_model': 'M7',
        'hypothesis': hypothesis['hypothesis'], 'configuration_plan': hypothesis['configuration_plan'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'architecture': 'Exact normalB attentionMLP and original complete task. Two epochEMA parameter states at decays.8/.95 initialized from actual source epoch5, updated once per epoch. Fixed buffers never averaged. Same three seeds, original patient3fold and fitting-only normalPCA32.',
        'trajectory_control': 'Only raw stopping curve controls duration. Every actual epoch parameter vector retained. Additional averaged-model evaluation is CPU/CUDA RNG neutral. Raw histories/durations/selected parameters must match historical normalB controls.',
        'planned_new_seed_fits': 12, 'planned_saved_selected_models': 36, 'predeclared_scores': 6,
        'counting': '12 optimizer trajectories, each producing raw/EMA.8/EMA.95 selected weights. These are36 checkpoint states, not36 independent fits. Original rawB/normalB/normalL source ensembles reused.',
        'selection': 'Highest qualifying originalCV strictly>M7 with validation>=M7 frozen before two unusedpatient3fold repeats; require highermean6foldF1 and jointpairedpatient95%lowerendpoint>0 against EACH M7/M6/M5/fixedGaussian. No automatic external test.',
        'audit': 'All4geometries/4plans frozen before12trajectories; raw controls matched to originals; all36 selected vectors reconstructed with independent explicit temporal scalarweights from actualsource snapshots; all12actualCUDA complete losses/parametergradients replayed; allstop/outer probabilities and24fixedcandidate phases checked.',
        'external_test_policy': 'Development only; feature rows outside2091 setNaN. No external predictions or outcomes loaded.',
        'frozen_artifacts': {str(p): sha(p) for p in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'weight_averaging.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'weight_averaging_experiments.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis_path, {**hypothesis, 'status': 'running', 'protocol': 'runs/weight_averaging/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 12,
        'saved_selected_models': 36}), flush=True)


if __name__ == '__main__': main()

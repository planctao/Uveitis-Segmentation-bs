"""Freeze the normal-tangent adversarial protocol before all geometry and head fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    family = 'normal_tangent_adversarial'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite a frozen protocol or restart its process.'
    verified = ROOT / 'verification_normal_tangent_adversarial.json'
    preflight = json.loads(verified.read_text()); assert preflight['passed']
    for name, expected in preflight['code_sha256'].items(): assert sha(ROOT / name) == expected
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    hypothesis = ROOT / 'results/next_normal_tangent_adversarial_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    code = ['normal_tangent_adversarial.py', 'normal_tangent_experiments.py',
            'verify_normal_tangent_adversarial.py', 'start_normal_tangent_adversarial.py',
            'audit_normal_tangent.py', 'normal_subspace.py', 'verify_normal_subspace.py', 'experiment.py',
            'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json']
    for family_name in ['axis_reference_mixtures', 'axis_mixture_confirmation']:
        integrity='freeze_integrity_after_confirmation.json' if family_name=='axis_mixture_confirmation' else 'freeze_integrity_after_training.json'
        artifacts += [ROOT / 'results' / ('research_' + family_name) / name
                      for name in ['comparison.json', 'verification.json', integrity]]
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    package = json.loads((ROOT / 'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference', 'subspace']:
            path = ROOT / 'results/milestone6/frozen_package' / f'seed_{seed}' / f'{branch}.pt'
            assert sha(path) == package['model_sha256'][f'{branch}_seed{seed}']
            artifacts.append(path)
    artifacts += sorted((ROOT / 'results/milestone6/reproduction_code').glob('*.py'))
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'prior_hypothesis_sha256': sha(hypothesis), 'prior_hypothesis': planned,
        'configuration_plan': planned['configuration_plan'],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 8, 'fixed_fusion_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': planned['hypothesis'], 'geometry': planned['geometry'], 'rng': planned['rng'],
        'architecture': planned['architecture'],
        'geometry_freeze': 'Freeze all4fit-only normal-patient geometries and4split plans before any head fitting. Original MLP initialization and Torch/CUDA RNG stay matched; only clean forward advances dropout RNG; perturbed forward replays masks in fork_rng.',
        'selection': planned['selection'],
        'audit': 'Zeroepsilon recovers original fulltask,predictions,parameter gradients and globalRNG. Independent full coarse/fine/patient-contrastive objective and matched-dropout finite differences verify FGSM. Four actual256imageGPUtasks, updates, recovery and witnesses audited. All48first batches must recover CUDA permutation, initial parameter SHA256, fit-only inputs, saved CPU/CUDA dropout state, complete losses/coordinate gradients/directions and attacks with independent NumPy projection plus plainM5. Restore all48selected checkpoints and stop/outer probabilities, patient coverage, frozen geometry and16fixed mixtures.',
        'external_test_policy': 'Only development rows0:2091; external feature arrays set NaN before fitting. No external outcomes or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items(): assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'normal_tangent_adversarial.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'normal_tangent_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': str((out / 'protocol.json').relative_to(ROOT)),
        'protocol_sha256': sha(out / 'protocol.json'), 'pid': process.pid,
        'formal_fits_started': 'Inspect state and actual model checkpoints for completed count'})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

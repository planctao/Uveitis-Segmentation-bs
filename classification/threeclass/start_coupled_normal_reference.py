"""Freeze the coupled normal-reference protocol before all geometry and head fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    family = 'coupled_normal_reference'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite a frozen protocol or restart its process.'
    verified = ROOT / 'verification_coupled_normal_reference.json'
    preflight = json.loads(verified.read_text()); assert preflight['passed']
    for name, expected in preflight['code_sha256'].items(): assert sha(ROOT / name) == expected
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    hypothesis = ROOT / 'results/next_coupled_normal_reference_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    code = ['coupled_normal_reference.py', 'coupled_normal_experiments.py',
            'verify_coupled_normal_reference.py', 'start_coupled_normal_reference.py',
            'scale_normal_geometry.py', 'normal_subspace.py', 'experiment.py',
            'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json']
    for family_name in ['scale_normal_geometry', 'bayesian_normal_reference']:
        artifacts += [ROOT / 'results' / ('research_' + family_name) / name
                      for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
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
        'hypothesis': planned['hypothesis'], 'geometry': planned['geometry'],
        'architecture': planned['architecture'],
        'geometry_freeze': 'Freeze all8 phase/mode geometry artifacts and4patient split plans with SHA256 before any head fitting; original MLP initialization and RNG unchanged.',
        'selection': planned['selection'],
        'audit': 'Independent patient loops/Gram spectra reconstruct normal means, RMS scales and independent/joint PCA; duplication invariance, regional contrast before clipping, cross-stream dependence, original RNG/zero-strength identity and four real256-image full-task gradient/update/recovery checks. Restore all48 actual checkpoints and stopping/outer probabilities using independent NumPy image-level correction plus plain M5; check geometry, complete patient coverage and16fixed fusion phases.',
        'external_test_policy': 'Only development rows0:2091; external feature arrays set NaN before fitting. No external outcomes or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items(): assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'coupled_normal_reference.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'coupled_normal_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': str((out / 'protocol.json').relative_to(ROOT)),
        'protocol_sha256': sha(out / 'protocol.json'), 'pid': process.pid,
        'formal_fits_started': 'Inspect state and actual model checkpoints for completed count'})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

"""Freeze the complete development-only protocol before dispatching any fit."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    family = 'bounded_normal_projection'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite an existing frozen protocol or restart its process.'
    verified = ROOT / 'verification_bounded_normal_projection.json'
    assert json.loads(verified.read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    code = ['bounded_normal_projection.py', 'bounded_normal_experiments.py',
            'verify_bounded_normal_projection.py', 'start_bounded_normal_projection.py',
            'normal_subspace.py', 'verify_normal_subspace.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json',
                 ROOT / 'results/research_shared_normal_head/comparison.json',
                 ROOT / 'results/research_shared_normal_head/verification.json',
                 ROOT / 'results/research_shared_normal_head/freeze_integrity_after_training.json']
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'configuration_plan': [{'name': f'{shape}_bound{int(factor)}', 'shape': shape, 'factor': factor}
                               for shape in ['hard', 'smooth'] for factor in [1., 2.]],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 8, 'fixed_fusion_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Fixed full normal-PCA suppression is still the confirmed reference. Diagnostic-span protection and shared-view learning failed; normal-axis augmentation did not satisfy both development gates. Test a bounded-removal alternative that removes normal-scale axis components while retaining larger residual coordinates. Large coordinates are not assumed to represent lesions.',
        'geometry': 'Unchanged original fit-only standardization and equal-normal-patient centroid PCA32 in all four streams. Each normal-axis deviation is sqrt(sample eigenvalue). All statistics refit per phase using only fitting patients.',
        'architecture': 'For coordinates z=(x-mu)B and limit k*normalDeviation, subtract clamp(z,-limit,limit)B^T or limit*tanh(z/limit)B^T, then M5 clipping and full original attention/taxonomy task. Four arms: hard/smooth by k=1/2. No extra trainable parameters. Four architectures, 48 actual heads; each also scored with predeclared .75 new branch/.25 M5 fixed probability mixture, eight scores and no fusion fits.',
        'selection': 'Original CV strictly > .8270827413505814 and validation >= .8904827120381046; freeze highest passing CV candidate before previously unused patient partitions against M6, M5 and fixed Gaussian. No automatic external evaluation.',
        'audit': 'Preflight independently checks every transform, zero-removal identity and large-bound full-PCA limit, bounded monotone residuals, actual 256-image full-task gradients/update/strict recovery. Check all 48 actual checkpoints, original schemas, geometry/deviation buffers, fit statistics, selected epochs, stop/outer probabilities, full patient coverage and all 16 fusion phases; freeze hashes before and after training.',
        'external_test_policy': 'Only 2091 development rows; external feature arrays set to NaN before fitting. No external outcomes, raw images or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items():
        assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'bounded_normal_projection.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'bounded_normal_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

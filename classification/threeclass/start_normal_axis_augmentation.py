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
    family = 'normal_axis_augmentation'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite an existing frozen protocol or restart its process.'
    verified = ROOT / 'verification_normal_axis_augmentation.json'
    assert json.loads(verified.read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    code = ['normal_axis_augmentation.py', 'normal_axis_experiments.py',
            'verify_normal_axis_augmentation.py', 'start_normal_axis_augmentation.py',
            'normal_subspace.py', 'verify_normal_subspace.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json',
                 ROOT / 'results/research_diagnostic_normal_projection/comparison.json',
                 ROOT / 'results/research_diagnostic_normal_projection/verification.json',
                 ROOT / 'results/research_diagnostic_normal_projection/freeze_integrity_after_training.json']
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'configuration_plan': [{'name': f'{mode}_scale{int(scale*100)}', 'mode': mode, 'scale': scale}
                               for mode in ['coherent', 'gaussian'] for scale in [.1, .3]],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 8, 'fixed_fusion_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Normal-centroid nuisance suppression improved patient development CV, but alternative moments and safeguarded diagnostic projections failed. Test whether exposure to small normal-subspace perturbations during training teaches robust diagnostic evidence. Compare coherent normal-patient donor differences to independent Gaussian draws with matched marginal sample covariance.',
        'geometry': 'Fit-only original normalization and equal-patient normal centroid PCA32, unchanged from the successful normal branch. Project normal patient centers into these axes. Refit geometry independently for every phase; no stopping or outer patients in donor banks.',
        'architecture': 'Base normal-suppressed M5 input, then train-only additive rank32 nuisance perturbation and clipping. Coherent mode draws distinct uniform normal patient donors a,b and uses (coordinates[a]-coordinates[b])/sqrt(2), sharing donors across all four streams and the image perturbation across region tokens. Gaussian mode independently draws each stream with deviation sqrt(sample eigenvalue). Fixed scales .1/.3. Four new architectures, 48 actual heads; fixed .75 new branch / .25 M5 fusion scores add no fits. Evaluation uses only the deterministic normal-suppressed network.',
        'selection': 'Original CV strictly > .8270827413505814 and validation >= .8904827120381046; freeze highest passing CV candidate before previously unused patient partitions against M6, M5 and fixed Gaussian. No automatic external evaluation.',
        'audit': 'Full preflight real-task loss/update/recovery; exact coherent-pair sample covariance checked. All 48 actual first-training-batch witnesses independently reconstruct fit-only inputs, private RNG donor/Gaussian draws and outputs. Check every geometry/bank/deviation, source schema, fit statistics, disease map, selected epoch and stop/outer probability on CPU, and all fixed mixtures. Verify frozen hashes before and after training.',
        'external_test_policy': 'Only 2091 development rows; external feature arrays set to NaN before fitting. No external outcomes, raw images or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items():
        assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'normal_axis_augmentation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'normal_axis_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

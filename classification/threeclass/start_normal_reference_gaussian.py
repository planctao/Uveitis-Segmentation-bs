"""Freeze the normal-reference Gaussian study before fitting any evidence model."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from normal_reference_gaussian_experiments import source_path


def main():
    family = 'normal_reference_gaussian'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family; storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out / 'protocol.json').exists()
    assert json.loads((ROOT / 'verification_patient_gaussian.json').read_text())['passed']
    code = ['normal_reference_gaussian.py', 'normal_reference_gaussian_experiments.py', 'start_normal_reference_gaussian.py',
            'normal_subspace.py', 'patient_gaussian.py', 'audit_gaussian_evidence.py', 'verify_normal_subspace.py',
            'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json', ROOT / 'results/milestone6/selection_before_test.json',
                 ROOT / 'verification_patient_gaussian.json', ROOT / 'results/research_scale_normal_geometry/comparison.json',
                 ROOT / 'results/research_scale_normal_geometry/verification.json',
                 ROOT / 'results/research_scale_normal_geometry/freeze_integrity_after_training.json']
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        for arm in ['zero_control', 'normal_rank32_full']:
            artifacts.append(ROOT / 'runs/normal_subspace' / arm / phase / 'predictions.npz')
    reference = ROOT / 'results/milestone5/selection_before_test.json'; artifacts.append(reference)
    checkpoints = [source_path(arm, phase, seed) for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']
                   for seed in range(3) for arm in ['raw', 'normal']]
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reference': str(reference.relative_to(ROOT)),
        'manifest_sha256': sha(ROOT / 'metadata/manifest.csv'),
        'source_checkpoints': {str(path): sha(path) for path in checkpoints},
        'frozen_artifacts': {str(path): sha(path) for path in artifacts},
        'code_sha256': {name: sha(ROOT / name) for name in code},
        'configuration_plan': [{'name': f'{grain}_shrink{int(shrink * 100)}', 'granularity': grain, 'shrinkage': shrink}
                               for grain in ['coarse', 'subtype'] for shrink in [.1, .3]],
        'planned_new_seed_fits': 0, 'planned_generative_fits': 48, 'reused_source_heads': 24,
        'minimum_cv_macro_f1': .8270827413505814, 'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Existing Gaussian evidence improved the raw M5 embedding in development. Test whether corresponding evidence in the frozen normal-reference embedding complements the confirmed M6 probability mixture. No source MLP is trained or selected again.',
        'architecture': 'For each of three frozen normal-reference MLPs fit patient-balanced coarse or subtype Gaussian evidence with shared covariance and fixed shrinkage .1/.3. Temperature=sqrt(256). Average three evidences. Final probability=.9*(.25 meanRaw+.75 meanNormal)+.1 meanNormalGaussian. Four predeclared candidates, 48 generative fits, zero new MLP/backbone fits.',
        'selection': 'Require original patient CV strictly greater than .8270827413505814 and validation >= .8904827120381046; freeze highest passing CV candidate before previously unused patient-partition confirmation against M6/M5/fixed Gaussian. No automatic external scoring.',
        'audit': 'Rebuild normal PCA and restore every actual raw/normal source stop/fit/outer embedding with actual model plus independent NumPy projection/plain M5. All 48 Gaussian moments independently rebuilt by per-patient loops; direct covariance quadratic scores verified on every stop/outer row. Rebuild all mixtures and original M6 scores, complete patient coverage, checkpoint/code hashes before and after fitting.',
        'external_test_policy': 'Only development indices; external features set NaN before research. No external outcomes loaded.'}
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_fitting', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'normal_reference_gaussian.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'normal_reference_gaussian_experiments.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'new_seed_fits': 0, 'planned_generative_fits': 48}), flush=True)


if __name__ == '__main__':
    main()

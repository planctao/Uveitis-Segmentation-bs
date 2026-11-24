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
    family = 'diagnostic_normal_projection'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite an existing frozen protocol or restart its process.'
    verified = ROOT / 'verification_diagnostic_normal_geometry.json'
    assert json.loads(verified.read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    code = ['diagnostic_normal_geometry.py', 'diagnostic_normal_experiments.py',
            'verify_diagnostic_normal_geometry.py', 'start_diagnostic_normal_projection.py',
            'normal_subspace.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json',
                 ROOT / 'results/research_normal_moment_geometry/comparison.json',
                 ROOT / 'results/research_normal_moment_geometry/verification.json',
                 ROOT / 'results/research_normal_moment_geometry/freeze_integrity_after_training.json']
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'configuration_plan': [{'name': f'{protection}_rank{rank}', 'protection': protection, 'rank': rank}
                               for protection in ['coarse', 'disease'] for rank in [8, 32]],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 8, 'fixed_fusion_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Normal patient-centroid suppression improved development generalization, while image/within-patient moments and selective streams failed. Test whether safeguarding patient-balanced coarse or disease mean contrasts prevents suppression of useful diagnostic evidence. This is an empirical supervised safeguard, not a claim of causal separation or methodological novelty.',
        'geometry': 'Original fit-only normalization/clipping. Average regions within each image. Fit equal-patient normal center and coarse/disease means using fit patients only. SVD the abnormal-minus-normal mean contrasts, retaining eigenvalues above max(1e-6*largest,1e-12). Residualize centered normal patient centers against this diagnostic span, then SVD residuals. Suppress leading 8/32 supported orthonormal residual directions in all four streams. Zero-pad unsupported directions. Coordinates along the protected span are preserved before the final M5 clipping.',
        'architecture': 'Original M5 taxonomy attention MLP with fixed safeguarded normal projection; unchanged complete loss, weights, hyperparameters, patient plans and seeds. Four new architectures, 12 actual heads each. Independently also score each with fixed .75 projection / .25 M5 probability mixture. No fusion fits.',
        'selection': 'Require original CV strictly > .8270827413505814 and validation >= .8904827120381046. Freeze highest passing CV candidate before confirmation on previously unused patient partitions against M6, M5 and fixed Gaussian. No automatic external evaluation.',
        'audit': 'Independent patient loops, small Gram eigenproblems and least-squares residualization recover every geometry and protected contrast. Check duplication invariance, degenerate rank, real task gradients/update/recovery, full original parameter schemas, fit statistics, selected epochs, all stop/outer probabilities and all mixtures. Freeze hashes verified before and after training.',
        'external_test_policy': 'Only 2091 development rows; external feature arrays set to NaN before fitting. No external outcomes, raw images or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items():
        assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'diagnostic_normal_projection.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'diagnostic_normal_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

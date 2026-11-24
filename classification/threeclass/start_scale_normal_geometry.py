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
    family = 'scale_normal_geometry'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite an existing frozen protocol or restart its process.'
    verified = ROOT / 'verification_scale_normal_geometry.json'
    assert json.loads(verified.read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    code = ['scale_normal_geometry.py', 'scale_normal_experiments.py',
            'verify_scale_normal_geometry.py', 'start_scale_normal_geometry.py',
            'normal_subspace.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json',
                 ROOT / 'results/research_bounded_normal_projection/comparison.json',
                 ROOT / 'results/research_bounded_normal_projection/verification.json',
                 ROOT / 'results/research_bounded_normal_projection/freeze_integrity_after_training.json']
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'configuration_plan': [{'name': f'{mode}_rank{rank}', 'mode': mode, 'rank': rank}
                               for mode in ['separate', 'tied'] for rank in [16, 32]],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 8, 'fixed_fusion_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Joint 1536-dimensional normal-patient PCA is the successful reference; bounded suppression failed to retain both M6 development gates. Test a scale-structured alternative: independent 768-dimensional normal bases per resolution, or shared axes pooled from the two centered resolutions. This imposes a different structural constraint without changing the downstream MLP.',
        'geometry': 'Actual channel halves verified against original 320/448 feature banks. Original fit-only normalization, region averaging within image, then equal normal patient centroids. Center each scale separately. Separate mode estimates each sample covariance; tied mode averages the two sample covariances via stacked residual rows/sqrt(2*(N-1)). Retain leading supported16/32 directions per scale, zero-pad unsupported eigenvalues <=max(1e-6*largest,1e-12). Every normal statistic refit per phase using fitting patients only.',
        'architecture': 'Project each normalized 768-dimensional scale half independently using its normal center and fixed basis, concatenate and apply M5 clipping and original full taxonomy attention MLP. Tied mode uses exactly the same basis in both halves. Total block-diagonal suppression rank32/64. Four architectures48actualheads; predeclared .75 new/.25 M5 mixtures add four scores and no fits.',
        'selection': 'Original CV strictly > .8270827413505814 and validation >= .8904827120381046. Freeze highest passing CV candidate before previously unused patient partitions against M6, M5 and fixed Gaussian. No automatic external evaluation.',
        'audit': 'Actual feature-half identities, independent patient loops/Gram eigenproblems, duplicated normal-image invariance, zero-strength original RNG identity and degenerate ranks checked before training. Four real256-imagefull-task gradient/update/restore checks. Audit all48 actual checkpoint schemas, fit statistics, geometry buffers, selected epochs, stop/outer probabilities, complete patient coverage and16fusion phases with independent per-scale NumPy transforms plus plain M5.',
        'external_test_policy': 'Only 2091 development rows; external feature arrays set to NaN before fitting. No external outcomes, raw images or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items():
        assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'scale_normal_geometry.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scale_normal_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

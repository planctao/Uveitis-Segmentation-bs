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
    family = 'shared_normal_head'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    assert not (out / 'protocol.json').exists(), 'Never overwrite an existing frozen protocol or restart its process.'
    verified = ROOT / 'verification_shared_normal_head.json'
    assert json.loads(verified.read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_moment_geometry/protocol.json').read_text())
    code = ['shared_normal_head.py', 'shared_normal_experiments.py',
            'verify_shared_normal_head.py', 'start_shared_normal_head.py',
            'normal_subspace.py', 'verify_normal_subspace.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json',
                 ROOT / 'results/research_normal_axis_augmentation/comparison.json',
                 ROOT / 'results/research_normal_axis_augmentation/verification.json',
                 ROOT / 'results/research_normal_axis_augmentation/freeze_integrity_after_training.json']
    artifacts += [ROOT / 'runs/normal_subspace/zero_control' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'preflight_sha256': sha(verified), 'source_checkpoints': parent['source_checkpoints'],
        'frozen_parent_artifacts': {str(path): sha(path) for path in artifacts},
        'configuration_plan': [{'name': 'shared_control', 'level': 'coarse', 'consistency': 0.},
                               {'name': 'shared_coarse01', 'level': 'coarse', 'consistency': .1},
                               {'name': 'shared_disease01', 'level': 'disease', 'consistency': .1},
                               {'name': 'shared_disease05', 'level': 'disease', 'consistency': .5}],
        'planned_new_seed_fits': 48, 'predeclared_candidate_scores': 4, 'fixed_normal_view_weight': .75,
        'minimum_cv_macro_f1': .8270827413505814,
        'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'The successful M6 mixes independently trained raw and normal-reference MLPs. Test whether sharing every MLP parameter while supervising both views retains complementary evidence with fewer unique head parameters. A zero-consistency control isolates the shared two-view task from coarse/disease probability agreement.',
        'geometry': 'Unchanged fit-only normalization and normal-patient-centroid PCA32, refit per phase. Only normal fitting patients define the projection. Raw and projected paths use the same standardized cached DINO features.',
        'architecture': 'One original M5 attention MLP parameter set evaluates raw and normal-reference inputs. Training task=.25 complete raw M5 loss+.75 complete projected M5 loss, including original fine contrastive terms separately. Optional patient-weighted symmetric KL over coarse or disease probabilities. Four fixed configurations: zero-consistency shared control; coarse KL .1; disease KL .1/.5. Inference always .25 raw+.75 normal-reference probabilities. 48 actual heads and four scores, no additional parent M5 mixture.',
        'selection': 'Original CV strictly > .8270827413505814 and validation >= .8904827120381046; freeze highest passing CV candidate before previously unused patient partitions against M6, M5 and fixed Gaussian. No automatic external evaluation.',
        'audit': 'Preflight verifies full weighted two-view task and NumPy coarse/disease KL, both view gradients, actual 256-image update and strict restore. All actual checkpoints must have exactly one original trainable parameter schema plus normal buffers. Independently reconstruct each raw and projected path with plain M5 plus NumPy projector; verify all stopping/outer probabilities, ensembles, fit statistics and geometry hashes.',
        'external_test_policy': 'Only 2091 development rows; external feature arrays set to NaN before fitting. No external outcomes, raw images or prediction files loaded for research selection.',
        'code_sha256': {name: sha(ROOT / name) for name in code}}
    for name, expected in protocol['source_checkpoints'].items():
        assert sha(name) == expected
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'shared_normal_head.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'shared_normal_experiments.py')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_seed_fits': 48, 'external_test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()

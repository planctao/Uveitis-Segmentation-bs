"""Freeze both cascade orders on concatenated raw/normal-reference embeddings."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    family = 'dual_reference_cascades'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    out = ROOT / 'runs' / family; storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out / 'protocol.json').exists()
    assert json.loads((ROOT / 'verification_conditional_experts.json').read_text())['passed']
    parent = json.loads((ROOT / 'runs/normal_reference_gaussian/protocol.json').read_text())
    for path, expected in parent['source_checkpoints'].items():
        assert sha(path) == expected
    code = ['dual_reference_cascade_experiments.py', 'start_dual_reference_cascades.py', 'conditional_experts.py',
            'normal_reference_gaussian.py', 'normal_reference_gaussian_experiments.py', 'normal_subspace.py',
            'verify_normal_subspace.py', 'audit_gaussian_evidence.py', 'patient_gaussian.py',
            'factorized_experiments.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']
    artifacts = [ROOT / 'results/milestone6/package.json', ROOT / 'results/milestone6/selection_before_test.json',
                 ROOT / parent['reference'], ROOT / 'verification_conditional_experts.json',
                 ROOT / 'results/research_normal_reference_gaussian/comparison.json',
                 ROOT / 'results/research_normal_reference_gaussian/verification.json',
                 ROOT / 'results/research_normal_reference_gaussian/freeze_integrity_after_training.json',
                 ROOT / 'results/research_conditional_experts/comparison.json',
                 ROOT / 'results/research_conditional_experts/verification.json']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reference': parent['reference'],
        'manifest_sha256': parent['manifest_sha256'], 'source_checkpoints': parent['source_checkpoints'],
        'frozen_artifacts': {str(path): sha(path) for path in artifacts},
        'code_sha256': {name: sha(ROOT / name) for name in code},
        'configuration_plan': [{'name': f'{order}_hidden{hidden}', 'order': order, 'hidden': hidden}
                               for order in ['cascade_i', 'cascade_n'] for hidden in [0, 64]],
        'binary_training': {'dropout': .6, 'lr': .002, 'wd': .1, 'epochs': 120, 'patience': 18},
        'planned_new_binary_head_fits': 96, 'planned_new_cascade_system_fits': 48, 'reused_source_heads': 24,
        'minimum_cv_macro_f1': .8270827413505814, 'minimum_validation_macro_f1': .8904827120381046,
        'hypothesis': 'Raw and normal-reference M6 branches contain complementary representations. Test whether independent clinical conditional heads over their concatenated 512D embeddings improve either requested hard cascade. This extends the previous raw-M5 conditional experts with a distinct normal-reference representation.',
        'architecture': 'For each seed concatenate frozen raw256D and normal-reference256D embeddings, standardize on corresponding fitting patients only. Train two independent binary heads for each requested order. Four fixed candidates: infection-first/normal-first by linear/64-GELU-dropout. Frozen source networks never receive downstream gradients.',
        'conditional_tasks': 'Infection-first: stage1 infectious versus noninfectious+normal; stage2 noninfectious versus normal using only true eligible fit/stop rows. Normal-first: stage1 abnormal versus normal; stage2 infectious versus noninfectious using true abnormal fit/stop rows. Recompute patient-balanced binary-task weights, exclude ineligible stage2 rows.',
        'stopping_and_routing': 'Each binary head early-stops on eligible stopping-patient binary macroF1, then logloss. Average scalar gates over three seeds. Both routing thresholds fixed .5; no threshold tuning. Report stage1, true-eligible stage2 and complete hard-route metrics independently. Ranking uses hard-routed macroF1.',
        'selection': 'Require hard-routed original patient CV > .8270827413505814 and validation >= .8904827120381046. Freeze highest passing CV candidate, all source and selected binary-model hashes before previously unused patient-partition confirmation against M6, M5 and fixed Gaussian. No automatic external scoring.',
        'audit': 'Existing binary task algebra, fitting and serialization verifier retained. Rebuild all 24 frozen source fit/stop/outer embeddings with independent NumPy projection plus plain M5, verify fit-only geometry and source statistics. Rebuild 512D normalization, every conditional mask/weight, selected binary epoch, all 96 stop/outer predictions with independent NumPy linear/GELU algebra and all stage/route metrics; verify code/source hashes before and after fitting.',
        'external_test_policy': 'Development indices only; external features set NaN. No external labels/outcomes/predictions or images loaded.'}
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_binary_head_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'dual_reference_cascades.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'dual_reference_cascade_experiments.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
                      'planned_new_binary_head_fits': 96, 'planned_new_cascade_system_fits': 48}), flush=True)


if __name__ == '__main__':
    main()

"""Freeze the highest qualifying residual/M7 mixture and six entirely unused plans."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan


def main():
    family = 'residual_m7_confirmation'
    out = ROOT / 'runs' / family
    assert not (out / 'protocol.json').exists()
    result = ROOT / 'results/research_residual_m7_mixtures'
    selection_record = result / 'selection_before_repeat.json'
    selected = json.loads(selection_record.read_text())['candidate']
    comparison = json.loads((result / 'comparison.json').read_text())
    verified = json.loads((result / 'verification.json').read_text())
    assert verified['passed'] and verified['actual_seed_heads_checked'] == 0
    assert verified['reused_source_heads_checked'] == 84 and len(verified['fixed_mixture_phases_checked']) == 32
    assert json.loads((ROOT / 'runs/residual_m7_mixtures/state.json').read_text())['status'] == 'completed'
    assert selected == max([row for row in comparison['candidates'] if row['development_gate_passed']],
                           key=lambda row: row['cv']['mean_macro_f1'])
    current = json.loads((ROOT / 'results/current_research_reference.json').read_text())
    assert current['retained_development_model'] == 'M7'
    assert selected['cv']['mean_macro_f1'] > current['original_cv_macro_f1']
    assert selected['val']['macro_f1'] >= current['validation_macro_f1']
    preflight = ROOT / 'verification_residual_m7_confirmation_preflight.json'
    proof = json.loads(preflight.read_text())
    assert proof['passed'] and len(proof['four_phases']) == 4
    for name, expected in proof['code_sha256'].items():
        assert sha(ROOT / name) == expected
    source_protocol = json.loads((ROOT / 'runs/residual_m7_mixtures/protocol.json').read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in source_protocol[key].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT / path) == expected
    choice = selected['mixture_configuration']
    residual_arm = next(arm for arm in json.loads((ROOT / 'runs/normal_large_residual/protocol.json').read_text())['configuration_plan']
                        if arm['name'] == choice['residual_arm'])
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']
    normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    large_cfg = {**normal, 'normal_large_feature': 'large_dual'}
    residual_cfg = {**normal, 'normal_large_feature': 'base_dual_large_dual',
                    'large_residual_rank': residual_arm['rank'], 'large_residual_scope': residual_arm['scope']}
    repeats = [20280801, 20280811]
    inner_seeds = [20280824 + ri*10 + fold for ri in range(2) for fold in range(3)]
    old_protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in old_protocols:
        assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), f'Previously reserved or used seed in {path}'
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    digest = sha(ROOT / 'metadata/manifest.csv')
    original = np.flatnonzero(frame.split.eq('train'))
    tf = frame.iloc[original]
    plans = []
    for ri, repeat in enumerate(repeats):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        seen = []
        for fold, (inner_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[inner_local]]
            outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda values: values.mode().iloc[0])
            inner_seed = 20280824 + ri*10 + fold
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=inner_seed, stratify=groups)
            fit = np.zeros(len(frame), bool)
            stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original)
            seen.extend(outer.tolist())
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            path = out / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json'
            dump(path, {'repeat': repeat, 'fold': fold, 'manifest_sha256': digest, 'inner_split_seed': inner_seed,
                'fit_indices': np.flatnonzero(fit).tolist(), 'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()})
            plans.append(path)
        assert sorted(seen) == original.tolist()
    artifacts = [preflight, selection_record, ROOT / 'metadata/manifest.csv', ROOT / 'results/current_research_reference.json',
        ROOT / 'results/milestone5/selection_before_test.json', ROOT / 'results/milestone6/package.json',
        ROOT / 'results/milestone6/selection_before_test.json', ROOT / 'runs/residual_m7_mixtures/protocol.json',
        ROOT / 'features/multiscale_vit.npz', ROOT / 'features/multiscale_vit.json']
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT / current[key]
        assert sha(path) == current[key+'_sha256']
        artifacts.append(path)
    large_source = json.loads((ROOT / 'runs/dinov3_large/protocol.json').read_text())
    artifacts += [Path(large_source['feature_storage']), Path(large_source['feature_storage']).with_suffix('.json'),
                  ROOT / 'runs/dinov3_large/protocol.json']
    artifacts += [result / name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    artifacts += [Path(path) for key in ['source_checkpoint_sha256', 'source_prediction_sha256'] for path in source_protocol[key]]
    code = ['residual_m7_confirmation.py', 'start_residual_m7_confirmation.py', 'normal_large_blend_confirmation.py',
        'normal_large_reference.py', 'normal_large_reference_experiments.py', 'audit_normal_large.py',
        'normal_large_residual.py', 'normal_large_residual_experiments.py', 'audit_normal_large_residual.py',
        'dinov3_large_data.py', 'audit_normal_tangent.py', 'normal_subspace.py', 'normal_subspace_experiments.py',
        'verify_normal_subspace.py', 'experiment.py', 'repeat_patient_cv.py', 'resolution_repeat_cv.py',
        'kernel_residual_experiments.py', 'patient_gaussian.py', 'audit_gaussian_evidence.py', 'repeated_oof.py']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'selection_record': str(selection_record.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'selected_configuration': choice, 'configurations': {'reference': raw, 'subspace': normal,
            'large': large_cfg, 'residual': residual_cfg}, 'outer_split_seeds': repeats,
        'inner_split_rule': '20280824+10*repeat_index+fold;20% of possible fitting patient/near-duplicate groups',
        'fresh_seed_audit': {str(p): sha(p) for p in old_protocols},
        'split_plan_sha256': {str(p): sha(p) for p in plans},
        'arms': ['reference', 'subspace', 'large', 'residual', 'fixed_gaussian', 'm6', 'm7', 'candidate'],
        'planned_new_seed_fits': 72, 'planned_generative_fits': 18,
        'hypothesis': 'Confirm the highest qualifying originalCV residual/M7 mixture. All four branches refit independently in six unused plans, three seeds each. Candidate weights and all training configurations fixed before fitting.',
        'reference_comparators': 'M5;fixed.9M5+.1patientGaussian;M6=.25rawB+.75normalB;M7=.1875rawB+.5625normalB+.25normalL.',
        'confirmation_gate': 'Mean sixfold MacroF1 strictly higher and repeated paired patient bootstrap95% MacroF1 lower endpoint>0 versus EACH of M7/M6/M5/fixedGaussian. No automatic external evaluation.',
        'external_test_policy': 'Only original1717 trainingimages form fit/stop/outer; original validation unused, all external feature rows NaN. No external predictions or outcomes loaded.',
        'audit': 'All six plans and12 normalspaces frozen before72fits. Independently reconstruct all72heads,36actual CUDA fulltasks, allstop/outer probabilities,12normalspaces,18Gaussianmoments, eight ensembles and four pairedcomparisons.',
        'frozen_artifacts': {str(p): sha(p) for p in artifacts}, 'code_sha256': {name: sha(ROOT / name) for name in code}}
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'residual_m7_confirmation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'residual_m7_confirmation.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'),
        'planned_new_seed_fits': 72, 'outer_split_seeds': repeats}), flush=True)


if __name__ == '__main__':
    main()

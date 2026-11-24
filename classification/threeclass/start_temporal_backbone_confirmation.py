"""Freeze the highest qualifying temporal/backbone candidate and six unused plans."""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan


def main():
    out = ROOT/'runs/temporal_backbone_confirmation'
    assert not (out/'protocol.json').exists()
    source = ROOT/'results/research_temporal_backbone_mixtures'
    selection_path = source/'selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    comparison = json.loads((source/'comparison.json').read_text())
    verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == 0
    assert verification['reused_source_heads_checked'] == 60 and len(verification['fixed_mixture_phases_checked']) == 12
    assert selected == max([r for r in comparison['candidates'] if r['development_gate_passed']],
        key=lambda r: r['cv']['mean_macro_f1'])
    assert selected['candidate_id'] in ['dropout_weight0p1', 'dropout_weight0p25', 'dropout_weight0p5']
    current = json.loads((ROOT/'results/current_research_reference.json').read_text())
    assert current['retained_development_model'] == 'M7'
    assert selected['cv']['mean_macro_f1'] > current['original_cv_macro_f1']
    assert selected['val']['macro_f1'] >= current['validation_macro_f1']
    assert json.loads((ROOT/'runs/temporal_backbone_mixtures/state.json').read_text())['status'] == 'completed'
    proof_path = ROOT/'verification_temporal_backbone_confirmation.json'
    proof = json.loads(proof_path.read_text()); assert proof['passed'] and len(proof['cases']) == 12
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    parent = json.loads((ROOT/'runs/temporal_backbone_mixtures/protocol.json').read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in parent[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']; normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    configurations = {'reference': raw, 'subspace': normal,
        'large': {**normal, 'normal_large_feature': 'large_dual'},
        'dropout': json.loads((ROOT/'results/research_backbone_dropout/selection_before_repeat.json').read_text())['candidate']['config']}
    repeats = [20281001, 20281011]
    inner_seeds = [20281024+10*ri+fold for ri in range(2) for fold in range(3)]
    old_protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in old_protocols:
        assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), str(path)
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/temporal_backbone_confirmation')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = sha(ROOT/'metadata/manifest.csv')
    original = np.flatnonzero(frame.split.eq('train')); tf = frame.iloc[original]; plans = []
    for ri, repeat in enumerate(repeats):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat); seen = []
        for fold, (inner_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[inner_local]]; outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda values: values.mode().iloc[0])
            inner_seed = 20281024+ri*10+fold
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=inner_seed, stratify=groups)
            fit = np.zeros(len(frame), bool); stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original); seen.extend(outer.tolist())
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            path = out/'split_plans'/f'repeat_{repeat}_fold_{fold}.json'
            dump(path, {'repeat': repeat, 'fold': fold, 'manifest_sha256': digest, 'inner_split_seed': inner_seed,
                'fit_indices': np.flatnonzero(fit).tolist(), 'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()})
            plans.append(path)
        assert sorted(seen) == original.tolist()
    artifacts = [proof_path, selection_path, ROOT/'metadata/manifest.csv', ROOT/'results/current_research_reference.json',
        ROOT/'results/milestone5/selection_before_test.json', ROOT/'results/milestone6/package.json',
        ROOT/'results/milestone6/selection_before_test.json', ROOT/'runs/temporal_backbone_mixtures/protocol.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json']
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.append(path)
    large_source = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts += [Path(large_source['feature_storage']), Path(large_source['feature_storage']).with_suffix('.json'),
        ROOT/'runs/dinov3_large/protocol.json']
    artifacts += [source/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    artifacts += [Path(p) for key in ['source_checkpoint_sha256', 'source_prediction_sha256'] for p in parent[key]]
    artifacts += [ROOT/'verification_weight_averaging.json', ROOT/'verification_backbone_dropout.json']
    code = ['temporal_backbone_confirmation.py', 'start_temporal_backbone_confirmation.py', 'verify_temporal_backbone_confirmation.py',
        'backbone_dropout.py', 'backbone_dropout_experiments.py', 'audit_backbone_dropout.py',
        'weight_averaging.py', 'trajectory_train.py', 'audit_weight_averaging.py',
        'normal_cross_geometry.py', 'audit_normal_cross_geometry.py',
        'normal_large_blend_confirmation.py', 'normal_large_reference.py', 'normal_large_reference_experiments.py',
        'audit_normal_large.py', 'dinov3_large_data.py', 'audit_normal_tangent.py', 'normal_subspace.py',
        'normal_subspace_experiments.py', 'verify_normal_subspace.py', 'experiment.py', 'repeat_patient_cv.py',
        'resolution_repeat_cv.py', 'kernel_residual_experiments.py', 'patient_gaussian.py', 'audit_gaussian_evidence.py', 'repeated_oof.py']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'selection_record': str(selection_path.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'selected_configuration': {'weight': selected['mixture_configuration']['dropout_weight'],
            'ema_source': parent['ema_source'], 'dropout_source': parent['dropout_source']},
        'configurations': configurations, 'outer_split_seeds': repeats,
        'inner_split_rule': '20281024+10*repeat_index+fold;20% of possible fit patient/near-duplicate groups',
        'fresh_seed_audit': {str(p): sha(p) for p in old_protocols}, 'split_plan_sha256': {str(p): sha(p) for p in plans},
        'arms': ['reference', 'subspace', 'ema', 'large', 'dropout', 'fixed_gaussian', 'm6', 'm7', 'candidate'],
        'planned_new_seed_fits': 72, 'planned_generative_fits': 18,
        'planned_used_source_weight_states': 90, 'planned_saved_selected_weight_states': 108,
        'counting': 'Four optimizer branches x6folds x3seeds =72trajectories. NormalB trajectory also producesEMA.8/EMA.95; onlyrawnormalB andEMA.8entercandidate. 90usedsource states and108savedselected states are not108fits.',
        'hypothesis': 'Confirm the highest qualifying originalCV temporal/backbone mixture. Refit all four optimizerbranches within each of sixunusedpatientplans, re-estimating normalBEMA solely from that fit trajectory. Fixedmixtureweight and completeoriginaltask; no alternatecandidate chosen from confirmation outcomes.',
        'reference_comparators': 'M5;fixed.9M5+.1patientGaussian;M6=.25rawB+.75normalB;M7=.1875rawB+.5625normalB+.25normalL.',
        'confirmation_gate': 'Mean sixfold MacroF1 strictly higher AND joint paired patient95% lower endpoint>0 versus EACH of M7/M6/M5/fixedGaussian. No automatic external evaluation.',
        'external_test_policy': 'Only original1717 training rows form fit/stop/outer; validation unused, all external rows NaN. No external predictions or outcomes loaded.',
        'audit': 'Freeze all6plans/18geometries before72optimizertrajectories. Hash allactualsource epochvectors beforeaudit; recover108savedselectedweightstates/90usedsources and54actualCUDA tasks (18normalBfullgradients,18normalL,18dropout), allstop/outer probabilities,18Gaussianmoments, nine ensembles/fourpairedcomparisons. Explicit scalarweights reconstruct all54normalB/EMAselected states.',
        'frozen_artifacts': {str(p): sha(p) for p in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'temporal_backbone_confirmation.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'temporal_backbone_confirmation.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    hypothesis_path = ROOT/'results/next_temporal_backbone_mixtures_hypothesis.json'
    hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'fresh_patient_confirmation_running', 'scores_evaluated': 3,
        'passing_development_gates': len([r for r in comparison['candidates'] if r['development_gate_passed']]), 'selected_candidate': selected['candidate_id'],
        'confirmation_protocol': str((out/'protocol.json').relative_to(ROOT)),
        'confirmation_protocol_sha256': sha(out/'protocol.json'), 'confirmation_pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'),
        'planned_new_seed_fits': 72, 'outer_split_seeds': repeats}), flush=True)


if __name__ == '__main__': main()

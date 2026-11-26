"""Freeze a multistream candidate, M8 controls and unused patient plans."""
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
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--family', required=True, choices=['large_token_interaction', 'large_token_cross_attention', 'register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry', 'base_token_evidence', 'register_common_mode', 'register_cosine', 'register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout'])
    family = parser.parse_args().family; suffix = 'pair' if family=='large_token_interaction' else 'cross'
    out = ROOT/'runs'/(family+'_confirmation'); assert not (out/'protocol.json').exists()
    source = ROOT/'results'/('research_'+family); selection_path = source/'selection_before_repeat.json'
    selected = json.loads(selection_path.read_text())['candidate']
    comparison = json.loads((source/'comparison.json').read_text()); verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
    assert selected.get('normalB_replacement_weight', selected.get('m7_replacement_weight', 1.)) in [.25, .5625, 1.]
    architecture_key = family+'_configuration' if family in ['register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry', 'base_token_evidence', 'register_common_mode', 'register_cosine', 'register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout'] else 'token_'+suffix+'_configuration'
    source_key = 'source_'+family+'_arm' if family in ['register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry', 'base_token_evidence', 'register_common_mode', 'register_cosine', 'register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout'] else 'source_token_'+suffix+'_arm'
    arm = selected[architecture_key]
    assert selected.get(source_key, selected['name']) == arm['name']
    registry = json.loads((ROOT/'results/research_reference_registry.json').read_text()); current_path=ROOT/registry['active_reference']
    assert sha(current_path) == registry['active_reference_sha256']; current=json.loads(current_path.read_text())
    assert current['retained_development_model'] == 'M8'
    assert selected['cv']['mean_macro_f1'] > current['original_cv_macro_f1']
    assert selected['val']['macro_f1'] >= current['validation_macro_f1']
    assert json.loads((ROOT/'runs'/family/'state.json').read_text())['status'] == 'completed'
    proof_path = ROOT/('verification_'+family+'_confirmation.json'); proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == 4
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    parent_path = ROOT/'runs'/family/'protocol.json'; parent = json.loads(parent_path.read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in parent[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']; normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    configurations = {'reference': raw, 'subspace': normal,
        'large': {**normal, 'normal_large_feature': 'large_dual'},
        'token': {**normal, 'normal_large_feature': 'large_dual', 'token_evidence_stream':'registers'}, 'multistream': selected['config']}
    seed_base = {'large_token_interaction':20290100, 'large_token_cross_attention':20290200, 'register_depth':20290300, 'register_multiquery':20290400, 'register_position_calibration':20290500, 'register_geometry':20290600, 'base_token_evidence':20290700, 'register_common_mode':20290800, 'register_cosine':20290900, 'register_scale_factorization':20291000, 'register_static_context':20291100, 'register_projection_freezing':20291200, 'register_classifier_context':20291300, 'register_routing_information':20291400, 'register_recurrent_refinement':20291500, 'register_conditional_readout':20291600}[family]
    repeats = [seed_base+1, seed_base+11]; inner_seeds = [seed_base+24+10*ri+fold for ri in range(2) for fold in range(3)]
    old_protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in old_protocols: assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), str(path)
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/(family+'_confirmation')
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
            inner_seed = seed_base+24+ri*10+fold
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
    artifacts = {Path(path) for path in parent['frozen_artifacts']}
    artifacts.update([proof_path, selection_path, parent_path, current_path, ROOT/'results/current_research_reference.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json', ROOT/'verification_large_token_evidence.json'])
    artifacts.update(source/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json'])
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.add(path)
    selected_record = json.loads(selection_path.read_text())
    for key in ['checkpoint_sha256', 'geometry_and_plans_sha256']:
        for path, expected in selected_record[key].items(): assert sha(path) == expected; artifacts.add(Path(path))
    code = sorted(set(parent['code_sha256']) | {
        'multistream_confirmation.py', 'start_multistream_confirmation.py', 'verify_multistream_confirmation.py',
        'large_token_confirmation.py', 'audit_large_token_evidence.py', 'large_token_evidence_experiments.py',
        'normal_large_blend_confirmation.py', 'repeat_patient_cv.py', 'resolution_repeat_cv.py', 'patient_gaussian.py',
        'audit_gaussian_evidence.py', 'repeated_oof.py', 'audit_normal_large.py', 'normal_large_reference_experiments.py',
        'normal_subspace_experiments.py', 'verify_normal_subspace.py'})
    protocol = {'family': family, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'selection_record': str(selection_path.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'selected_configuration': {'architecture': arm, 'replacement_weight': selected.get('normalB_replacement_weight', selected.get('m7_replacement_weight',1.))},
        'configurations': configurations, 'outer_split_seeds': repeats,
        'inner_split_rule': str(seed_base+24)+'+10*repeat_index+fold;20% of possible fit patient/near-duplicate groups',
        'fresh_seed_audit': {str(p): sha(p) for p in old_protocols}, 'split_plan_sha256': {str(p): sha(p) for p in plans},
        'arms': ['reference', 'subspace', 'large', 'token', 'multistream', 'fixed_gaussian', 'm6', 'm7', 'm8', 'candidate'],
        'planned_new_seed_fits': 90, 'planned_generative_fits': 18,
        'hypothesis': 'Confirm the highest qualifying originalCV token candidate. Refit rawB,normalB,originalnormalL,registerL,multistreamL on sixunusedpatientplans; all preprocessing and normalspaces fit strictly within fittingpatients. Candidate fixed selectedstandalone or predeclaredL/B replacement formula; do not reselect standalone/alternate arms from confirmation outcomes.',
        'reference_comparators': 'M5;fixed.9M5+.1patientGaussian;M6=.25rawB+.75normalB;M7=.1875rawB+.5625normalB+.25normalL;M8=.1875rawB+.5625normalB+.25registerL.',
        'confirmation_gate': 'Mean sixfold MacroF1 strictly higher AND paired patient95% lower endpoint>0 versus EACH M8/M7/M6/M5/fixedGaussian. No automatic external evaluation.',
        'external_test_policy': 'Only original1717 trainingrows form fit/stop/outer; validation unused; all externalfeature rows NaN. No external outcomes or predictions loaded.',
        'audit': 'Freeze all6plans/24geometries before90optimizertrajectories; freeze products before independentaudit. Restore all90selectedheads/statistics/maps/epochs/stop/outerprobs, all54actualCUDA normalL/registerL/candidate complete augmentedtasks/alltrainablegradients,18Gaussian moments, tenensembles/fivepairedcomparisons. Independentlyexpand allfixedfusion arithmetic in float64.',
        'frozen_artifacts': {str(p): sha(p) for p in sorted(artifacts)}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/(family+'_confirmation.log')).open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'multistream_confirmation.py'), '--family', family], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    hypothesis_path = ROOT/'results'/('next_'+family+'_hypothesis.json'); hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'fresh_patient_confirmation_running', 'selected_candidate': selected['candidate_id'],
        'confirmation_protocol': str((out/'protocol.json').relative_to(ROOT)), 'confirmation_protocol_sha256': sha(out/'protocol.json'), 'confirmation_pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 90, 'outer_split_seeds': repeats}), flush=True)


if __name__ == '__main__': main()

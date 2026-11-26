"""Independently recompute saved development metrics after a patch study finishes."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FAMILIES = ['register_patch_joint', 'register_patch_gated_value',
    'register_patch_readout_adaptation', 'inflammatory_pair_specialist']
PROTOCOL = ROOT/'runs/completed_patch_family_review/protocol.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def independent_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability)
    assert len(labels) and probability.shape == (len(labels), 3)
    assert np.isin(labels, [0, 1, 2]).all() and np.isfinite(probability).all()
    assert (probability >= 0).all() and np.allclose(probability.sum(1), 1, atol=1e-6, rtol=1e-6)
    decision = probability.argmax(1)
    matrix = np.bincount(3*labels+decision, minlength=9).reshape(3, 3)
    true_count, predicted_count = matrix.sum(1), matrix.sum(0)
    diagonal = np.diag(matrix)
    f1 = np.divide(2*diagonal, true_count+predicted_count,
        out=np.zeros(3, dtype=float), where=true_count+predicted_count > 0)
    recall = np.divide(diagonal, true_count, out=np.zeros(3, dtype=float), where=true_count > 0)
    bounded = np.clip(probability.astype('float64'), 1e-7, 1)
    bounded /= bounded.sum(1, keepdims=True)
    return {'n': len(labels), 'accuracy': float(diagonal.sum()/len(labels)),
        'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[true_count > 0].mean()),
        'log_loss': float(-np.log(bounded[np.arange(len(labels)), labels]).mean()),
        'confusion_matrix': matrix.tolist()}


def verify_metric(expected, actual):
    assert expected['n'] == actual['n']
    assert expected['confusion_matrix'] == actual['confusion_matrix']
    for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss']:
        assert abs(expected[key]-actual[key]) < 1e-12, (key, expected[key], actual[key])


def review(family):
    assert family in FAMILIES
    run, result = ROOT/'runs'/family, ROOT/'results'/('research_'+family)
    state = json.loads((run/'state.json').read_text())
    assert state['status'] == 'completed', 'Only completed studies can be reviewed; never read active checkpoints.'
    runner_protocol = json.loads(PROTOCOL.read_text())
    assert sha(__file__) == runner_protocol['code_sha256']
    for name, expected in runner_protocol['frozen_artifacts'].items():
        assert sha(name) == expected, name
    inputs = {str(PROTOCOL): sha(PROTOCOL)}

    def check(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        if str(path) in inputs:
            assert inputs[str(path)] == digest
        inputs[str(path)] = digest

    def document(path):
        check(path)
        return json.loads(Path(path).read_text())

    manifest = ROOT/'metadata/manifest.csv'
    check(manifest, runner_protocol['manifest_sha256'])
    frame = pd.read_csv(manifest)
    protocol = document(run/'protocol.json')
    assert protocol['manifest_sha256'] == runner_protocol['manifest_sha256']
    comparison = document(result/'comparison.json')
    verification = document(result/'verification.json')
    integrity = document(result/'freeze_integrity_after_training.json')
    assert comparison['protocol'] == protocol
    assert verification['passed'] and integrity['passed']
    assert state['new_seed_fits'] == comparison['new_seed_fits'] == 48
    assert verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == len(verification['fits']) == 48
    assert verification['frozen_register_source_heads_checked'] == len(verification['register_source_heads']) == 12
    assert verification['reused_fusion_source_heads_checked'] == len(verification['source_heads']) == 24
    assert len(verification['fixed_fusion_phases_checked']) == 16
    assert not comparison['external_test_evaluated'] and not verification['external_test_evaluated']
    for section in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name)
            check(path if path.is_absolute() else ROOT/path, expected)
    products = document(run/'training_products_freeze_before_audit.json')['input_artifact_sha256']
    assert len(products) == 240
    for name, expected in products.items():
        check(name, expected)
    phases, baselines = {}, {}
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        plan = document(run/'split_plans'/(phase+'.json'))
        arrays = {name: np.asarray(plan[name], dtype=int) for name in ['fit_indices', 'stop_indices', 'outer_indices']}
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[arrays[name]][key]) for name in arrays]
            assert all(not groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)])
        assert frame.iloc[arrays['fit_indices']].split.eq('train').all()
        if phase == 'validation':
            assert np.array_equal(arrays['fit_indices'], np.flatnonzero(frame.split.eq('train')))
            assert np.array_equal(arrays['stop_indices'], np.flatnonzero(frame.split.eq('val')))
            assert len(arrays['outer_indices']) == 0
        else:
            assert all(frame.iloc[ids].split.eq('train').all() for ids in arrays.values())
        phases[phase] = arrays['stop_indices'] if phase == 'validation' else arrays['outer_indices']
        baselines[phase] = {}
        for name, folder in [('rawB', 'normal_subspace/zero_control'), ('normalB', 'normal_subspace/normal_rank32_full'), ('sourceL', 'large_token_evidence/registers')]:
            path = ROOT/'runs'/folder/phase/'predictions.npz'
            check(path)
            with np.load(path) as saved:
                assert np.array_equal(saved['indices'], phases[phase])
                baselines[phase][name] = saved['p'].copy()
    outer = np.concatenate([ids for phase, ids in phases.items() if phase != 'validation'])
    assert np.array_equal(np.sort(outer), np.flatnonzero(frame.split.eq('train')))
    checked_fits = set()
    for row in verification['fits']:
        key = (row['arm'], row['phase'], row['seed'])
        assert key not in checked_fits
        checked_fits.add(key)
        folder = run/row['arm']/row['phase']/f'seed_{row["seed"]}'
        check(folder/'model.pt', row['checkpoint_sha256'])
        check(folder/'training_input_witness.npz', row['training_witness_sha256'])
        witness = row['witness']
        assert witness['actual_images'] == 256
        assert witness['original_cuda_outputs_sha256_restored'] and witness['frozen_head_tail_verified']
    rows, saved_probabilities = [], {}
    candidate_names = [arm['name'] for arm in protocol['configuration_plan']]
    assert len(candidate_names) == 4
    assert len(comparison['candidates']) == 8
    assert {row['name'] for row in comparison['candidates']} == set(candidate_names+[name+'_m8replace' for name in candidate_names])
    for candidate in comparison['candidates']:
        name, fold_scores = candidate['name'], []
        for phase, ids in phases.items():
            path = run/name/phase/'predictions.npz'
            check(path)
            with np.load(path) as saved:
                assert np.array_equal(saved['indices'], ids)
                p = saved['p'].copy()
                assert np.array_equal(saved['pred'], p.argmax(1))
            actual = independent_metrics(frame.iloc[ids].label, p)
            if phase == 'validation':
                verify_metric(candidate['val'], actual)
                val = actual
            else:
                source = next(row for row in candidate['cv']['folds'] if row['phase'] == phase)
                verify_metric(source['metrics'], actual)
                fold_scores.append(actual['macro_f1'])
            saved_probabilities[(name, phase)] = p
        mean_f1 = float(np.mean(fold_scores))
        assert abs(mean_f1-candidate['cv']['mean_macro_f1']) < 1e-12
        passed = mean_f1 > protocol['minimum_cv_macro_f1'] and val['macro_f1'] >= protocol['minimum_validation_macro_f1']
        assert passed == candidate['development_gate_passed']
        rows.append({'candidate': name, 'cv_macro_f1': mean_f1, 'validation_macro_f1': val['macro_f1'],
            'validation_accuracy': val['accuracy'], 'development_gate_passed': passed})
    for name in candidate_names:
        for phase in phases:
            a, b = baselines[phase]['rawB'], baselines[phase]['normalB']
            adapted = saved_probabilities[(name, phase)]
            fused = saved_probabilities[(name+'_m8replace', phase)]
            expected = (.1875*a+.5625*b+.25*adapted).astype('float32')
            assert np.array_equal(fused, expected)
            independent = (.1875*a.astype('float64')+.5625*b.astype('float64')+.25*adapted.astype('float64')).astype('float32')
            assert np.allclose(fused, independent, atol=2e-7, rtol=2e-7)
            assert np.array_equal(fused.argmax(1), independent.argmax(1))
            if family == 'inflammatory_pair_specialist':
                assert np.array_equal(adapted[:, 2], baselines[phase]['sourceL'][:, 2])
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert state['passing_development_gates'] == len(eligible)
    selection = run.parent.parent/'results'/('research_'+family)/'selection_before_repeat.json'
    if eligible:
        chosen = document(selection)
        assert chosen['requires_fresh_patient_confirmation']
        assert chosen['candidate'] == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
        for name, expected in chosen['checkpoint_sha256'].items():
            check(name, expected)
    else:
        assert not selection.exists()
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    return {'review_passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'family': family, 'classification_fits': 48, 'actual_task_witnesses': 48,
        'training_products_rechecked': 240, 'saved_candidate_phases_metrics_recomputed': 32,
        'patient_and_duplicate_group_plans_checked': 4, 'fixed_fusion_phases_recomputed': 16,
        'gate_passers': len(eligible), 'rows': rows, 'input_sha256': inputs,
        'scope_limit': 'Independent arithmetic reconstruction of saved development predictions/metrics and frozen-byte/accounting checks. Existing study-specific semantic model/task verification is checked, not rerun. No new candidates, weights, thresholds, model fitting, prediction inference, reference promotion or external scoring.',
        'new_model_fits': 0, 'new_model_predictions': 0, 'new_candidate_scores': 0,
        'external_test_evaluated': False, 'code_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES, required=True)
    args = parser.parse_args()
    target = ROOT/'results'/('research_'+args.family)/'independent_saved_output_review.json'
    assert not target.exists(), 'Preserve prior reviews; version any recovery.'
    result = review(args.family)
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'review_passed': True, 'family': args.family,
        'gate_passers': result['gate_passers'], 'saved_metric_phases_checked': 32,
        'input_records_checked': len(result['input_sha256'])}), flush=True)


if __name__ == '__main__':
    main()

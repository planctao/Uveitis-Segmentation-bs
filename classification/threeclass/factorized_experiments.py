"""Matched flat and two-order coarse/conditional-disease factorization."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, train_one, summarize_group, metrics, dump, tune_thresholds, routed, leaf_probs
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof

OUT = ROOT / 'runs/factorized_taxonomy'


@torch.inference_mode()
def collect_cpu(checkpoint, features, indices):
    model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    ps, gs = [], []
    for begin in range(0, len(indices), 256):
        selected = indices[begin:begin+256]
        x, _ = normalize({key: value[selected] for key, value in features.items()}, None, checkpoint['stats'])
        out = model({key: torch.from_numpy(value) for key, value in x.items()})
        assert torch.allclose(out['p'], out['factorized_disease_probability'] @ model.subtype_to_class, atol=1e-6)
        assert torch.allclose(out['factorized_disease_probability'].sum(1), torch.ones(len(selected)), atol=1e-6)
        ps.append(out['p'].numpy());gs.append(out['factorized_gates'].numpy())
    return np.concatenate(ps), np.concatenate(gs)


def stage_metrics(labels, gates, order, thresholds):
    y = np.asarray(labels)
    first_target = y == 0 if order == 'cascade_i' else y != 2
    eligible = y != 0 if order == 'cascade_i' else y != 2
    second_target = y == 1 if order == 'cascade_i' else y == 0
    pred = routed(gates, order, thresholds)
    return {'thresholds': thresholds, 'images': len(y),
        'stage1_accuracy': float(np.mean((gates[:, 0] >= thresholds[0]) == first_target)),
        'oracle_stage2_images': int(eligible.sum()),
        'oracle_stage2_accuracy': float(np.mean((gates[eligible, 1] >= thresholds[1]) == second_target[eligible])),
        'end_to_end': metrics(y, leaf_probs(gates, order), pred),
        'policy': 'Average each scalar gate across seeds, route with stopping-only thresholds. Oracle stage2 uses the true eligible subset; end-to-end includes routing errors.'}


def config_for(reference, arm):
    return {**reference['selected']['config'], 'taxonomy_factorized': True,
        'factorization_order': arm['order'], 'factorization_conditional_aux': arm['conditional_aux']}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_factorized_taxonomy.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    records = []
    for arm in protocol['configuration_plan']:
        config = config_for(reference, arm)
        group = OUT / arm['name'] / 'validation'
        phases = phase_plans(frame, reference)
        for seed in range(3):
            dump(OUT / 'state.json', {'status': 'formal_training', 'pid': os.getpid(), 'arm': arm['name'], 'seed': seed, 'external_test_evaluated': False})
            train_one(frame, features, digest, {**config, 'seed': seed}, group / f'seed_{seed}')
        record = summarize_group(group, frame)
        record.update(name=arm['name'], candidate_id='factorized_' + arm['name'], factorization=arm)
        fold_results = []
        for phase, fitting, stopping, outer, _ in phases:
            target = OUT / arm['name'] / phase
            evaluation = stopping if phase == 'validation' else outer
            fit_mask = np.zeros(len(frame), dtype=bool);stop_mask = fit_mask.copy()
            fit_mask[fitting] = True;stop_mask[stopping] = True
            probabilities, stopping_probabilities, gates, stopping_gates = [], [], [], []
            for seed in range(3):
                if phase != 'validation':
                    dump(OUT / 'state.json', {'status': 'patient_cv_training', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                    train_one(frame, features, digest, {**config, 'seed': seed}, target / f'seed_{seed}', fit_mask, stop_mask)
                c = torch.load(target / f'seed_{seed}/model.pt', map_location='cpu', weights_only=False)
                p, g = collect_cpu(c, features, evaluation)
                probabilities.append(p);gates.append(g)
                if phase == 'validation':stopping_probabilities.append(p);stopping_gates.append(g)
                else:
                    p, g = collect_cpu(c, features, stopping)
                    stopping_probabilities.append(p);stopping_gates.append(g)
            p, g = np.mean(probabilities, 0), np.mean(gates, 0)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1), gates=g)
            result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            if arm['order'] != 'flat':
                thresholds = tune_thresholds(frame.iloc[stopping].label, np.mean(stopping_gates, 0), arm['order'])
                result['hard_route'] = stage_metrics(frame.iloc[evaluation].label, g, arm['order'], thresholds)
                result['hard_route_default05'] = stage_metrics(frame.iloc[evaluation].label, g, arm['order'], [.5, .5])
            dump(target / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1), gates=g)
                fold_results.append(result)
            else:
                assert abs(result['metrics']['macro_f1'] - record['val']['macro_f1']) < 1e-12
                if 'hard_route' in result:record['hard_route_validation'] = result['hard_route']
        record['cv'] = {'mean_macro_f1': float(np.mean([row['metrics']['macro_f1'] for row in fold_results])),
            'sd_macro_f1': float(np.std([row['metrics']['macro_f1'] for row in fold_results], ddof=1)), 'folds': fold_results}
        source_cv = Path(reference['selected']['source_cv_summary']).parent
        record['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / arm['name'], frame)
        record['development_gate_passed'] = record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'factorized_arm_complete', 'arm': arm['name'], 'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1']}), flush=True)
    destination = ROOT / 'results/research_factorized_taxonomy'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'candidates': records, 'new_seed_fits': 48, 'external_test_evaluated': False})
    audit_results(frame, features, digest, reference, protocol, records)
    eligible = [row for row in records if row['development_gate_passed']]
    if eligible:dump(destination / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def audit_results(frame, features, digest, reference, protocol, records):
    checked = []
    for arm, record in zip(protocol['configuration_plan'], records):
        plans = phase_plans(frame, reference)
        seen, scores = [], []
        for phase, fitting, stopping, outer, _ in plans:
            evaluation = stopping if phase == 'validation' else outer
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[ids][key]) for ids in [fitting, stopping, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            tf = frame.iloc[fitting]
            probabilities, gates, stop_gates = [], [], []
            for seed in range(3):
                folder = OUT / arm['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == {**config_for(reference, arm), 'seed': seed}
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                mapping = tf.groupby('disease').label.first()
                assert c['dims']['disease_to_class'] == [int(mapping[name]) for name in c['diseases']]
                for key, values in features.items():
                    z = values[fitting];axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p, g = collect_cpu(c, features, stopping)
                expected = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, expected, atol=5e-5, rtol=5e-5)
                assert np.array_equal(p.argmax(1), expected.argmax(1))
                stop_gates.append(g)
                if phase == 'validation':probabilities.append(p);gates.append(g)
                else:
                    p, g = collect_cpu(c, features, outer);probabilities.append(p);gates.append(g)
                checked.append({'arm': arm['name'], 'phase': phase, 'seed': seed})
            p, g = np.mean(probabilities, 0), np.mean(gates, 0)
            saved = np.load(OUT / arm['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation)
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5) and np.allclose(saved['gates'], g, atol=5e-5, rtol=5e-5)
            assert np.array_equal(saved['pred'], p.argmax(1))
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            stored = json.loads((OUT / arm['name'] / phase / 'metrics.json').read_text())
            for key in ['macro_f1', 'accuracy', 'log_loss']:assert abs(actual[key] - stored['metrics'][key]) < 1e-12
            if arm['order'] != 'flat':
                thresholds = tune_thresholds(frame.iloc[stopping].label, np.mean(stop_gates, 0), arm['order'])
                assert thresholds == stored['hard_route']['thresholds']
                for key in ['hard_route', 'hard_route_default05']:
                    result = stage_metrics(frame.iloc[evaluation].label, g, arm['order'], thresholds if key == 'hard_route' else [.5, .5])
                    assert result == stored[key]
            if phase != 'validation':scores.append(actual['macro_f1']);seen.extend(evaluation.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(record['cv']['mean_macro_f1'] - np.mean(scores)) < 1e-12
    assert len(checked) == 48
    dump(ROOT / 'results/research_factorized_taxonomy/verification.json', {'passed': True,
        'manifest_sha256': digest, 'actual_seed_fits_checked': 48, 'fits': checked,
        'checks': ['Every flat and two-stage factorization configuration receives all12actual seed fits',
            'Patient and duplicate isolation, fit-only statistics and disease mappings independently rebuilt',
            'Every actual model strictly restored and full stopping/outer predictions independently recovered on CPU',
            'Fine probabilities recover coarse marginals and sum to1 for every checked batch',
            'All soft ensemble metrics and original outer coverage reconstructed',
            'Both hard routing orders, stopping-only thresholds and oracle stage2 subset sizes/accuracies independently reconstructed']})
    print('All48factorized heads and bothhardroutingorders audited', flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in ['factorized_experiments.py', 'factorized_head.py', 'experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()

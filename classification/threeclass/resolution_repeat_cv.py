"""Confirm a fixed probability expert on two previously unused patient partitions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, predict, train_one, metrics, dump
from repeat_patient_cv import audit

OUT = ROOT / 'runs/resolution_repeat_cv'
ARMS = ['milestone5_reference', 'reference75_raw768_25', 'reference75_up768_25']


def paired(reference, new, frame, repeats):
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    tf = frame.iloc[original]
    _, group = np.unique(tf.split_group, return_inverse=True)
    count = group.max() + 1

    def confusion(arm):
        c = np.zeros((count, 3, 3), dtype=np.int64)
        for repeat in repeats:
            sources = [np.load(OUT / arm / f'repeat_{repeat}/fold_{fold}/outer_predictions.npz') for fold in range(3)]
            indices = np.concatenate([s['indices'] for s in sources])
            probabilities = np.concatenate([s['p'] for s in sources])
            order = indices.argsort()
            assert np.array_equal(indices[order], original)
            c += np.bincount(group * 9 + tf.label.to_numpy() * 3 + probabilities[order].argmax(1),
                             minlength=count * 9).reshape(count, 3, 3)
        return c

    def scores(c):
        diagonal = np.diagonal(c, axis1=-2, axis2=-1)
        denominator = c.sum(-1) + c.sum(-2)
        f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                       where=denominator > 0).mean(-1)
        accuracy = diagonal.sum(-1) / c.sum((-2, -1))
        return accuracy, f1

    a, b = confusion(reference), confusion(new)
    samples = np.random.default_rng(20261224).integers(0, count, size=(2000, count))
    aa, af = scores(a[samples].sum(1))
    ba, bf = scores(b[samples].sum(1))
    pa, pf = scores(a.sum(0))
    qa, qf = scores(b.sum(0))
    return {'accuracy_difference': float(qa - pa), 'macro_f1_difference': float(qf - pf),
        'accuracy_ci95': np.quantile(ba - aa, [.025, .975]).tolist(),
        'macro_f1_ci95': np.quantile(bf - af, [.025, .975]).tolist(),
        'bootstrap_unit': 'Patient split group; all images and both repeated predictions jointly resampled.'}


def check_plan(frame, fit, stop, outer, original):
    groups = [set(np.flatnonzero(fit)), set(np.flatnonzero(stop)), set(outer)]
    assert set.union(*groups) == set(original)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for key in ['patient_group', 'split_group']:
        partitions = [set(frame.iloc[sorted(g)][key]) for g in groups]
        assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2]
        assert not partitions[1] & partitions[2]


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    selected_path = ROOT / 'results/research_probability_fusion/selection_before_repeat.json'
    assert hashlib.sha256(selected_path.read_bytes()).hexdigest() == protocol['selection_record_sha256']
    winner = json.loads(selected_path.read_text())['candidate']
    assert winner['candidate_id'] == protocol['selected_candidate']
    assert winner['donor_weight'] == protocol['donor_weight'] == .25
    assert winner['donor_feature'] == 'vitb_raw768' and protocol['arms'] == ARMS
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    donors = json.loads((ROOT / 'runs/raw_highres/leaderboard.json').read_text())
    configs = {reference['feature']: reference['config']}
    configs.update({r['feature']: r['config'] for r in donors if r['feature'] in protocol['source_features']})
    frame, base_features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    banks = {'multiscale_vit': base_features}
    for feature in ['vitb_raw768', 'vitb_up768']:
        other, values, actual_digest = load_data(feature)
        assert digest == actual_digest and np.array_equal(frame.path, other.path)
        banks[feature] = values
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    tf = frame.iloc[original]
    for ri, repeat in enumerate(protocol['outer_split_seeds']):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        for fold, (fit_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            outer = original[outer_local]
            possible = frame.iloc[original[fit_local]]
            patients = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(patients.index, test_size=.2,
                random_state=20261224 + ri * 10 + fold, stratify=patients)
            fit = np.zeros(len(frame), dtype=bool)
            stop = fit.copy()
            fit[possible.index] = possible.split_group.isin(fit_ids)
            stop[possible.index] = possible.split_group.isin(stop_ids)
            check_plan(frame, fit, stop, outer, original)
            plan = {'manifest_sha256': digest, 'repeat': repeat, 'fold': fold,
                    'fit_indices': np.flatnonzero(fit).tolist(),
                    'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()}
            plan_path = OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json'
            if plan_path.exists():
                assert json.loads(plan_path.read_text()) == plan
            else:
                dump(plan_path, plan)
            source_probabilities = {}
            for feature in protocol['source_features']:
                target = OUT / 'sources' / feature / f'repeat_{repeat}/fold_{fold}'
                features = banks[feature]
                saved_path = target / 'outer_predictions.npz'
                if saved_path.exists():
                    saved = np.load(saved_path)
                    assert np.array_equal(saved['indices'], outer)
                    source_probabilities[feature] = saved['p']
                    continue
                predictions = []
                for seed in range(3):
                    dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'feature': feature,
                         'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                    destination = target / f'seed_{seed}'
                    train_one(frame, features, digest, {**configs[feature], 'seed': seed}, destination, fit, stop)
                    c = torch.load(destination / 'model.pt', map_location='cpu', weights_only=False)
                    audit(c, frame, features, fit, stop, outer, digest)
                    model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval().cuda()
                    model.load_state_dict(c['state_dict'], strict=True)
                    normalized, _ = normalize({k: v[outer] for k, v in features.items()}, None, c['stats'])
                    x = {k: torch.from_numpy(v).cuda() for k, v in normalized.items()}
                    p, gates = predict(model, x)
                    assert gates.shape[1] == 0
                    predictions.append(p)
                    del model, x
                    torch.cuda.empty_cache()
                p = np.mean(predictions, 0)
                source_probabilities[feature] = p
                np.savez_compressed(saved_path, indices=outer, p=p, pred=p.argmax(1))
            base = source_probabilities['multiscale_vit']
            combinations = [base, .75 * base + .25 * source_probabilities['vitb_raw768'],
                            .75 * base + .25 * source_probabilities['vitb_up768']]
            for arm, p in zip(ARMS, combinations):
                target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / 'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
                report = {'repeat': repeat, 'fold': fold, 'metrics': metrics(frame.iloc[outer].label, p)}
                dump(target / 'metrics.json', report)
                print(json.dumps({'event': 'resolution_repeat_fold', 'arm': arm, **report}), flush=True)
    verify_and_report(frame, banks, configs, digest, protocol)


def verify_and_report(frame, banks, configs, digest, protocol):
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    checked = []
    for repeat in protocol['outer_split_seeds']:
        seen = []
        for fold in range(3):
            plan = json.loads((OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json').read_text())
            assert plan['manifest_sha256'] == digest
            fit = np.zeros(len(frame), dtype=bool)
            stop = fit.copy()
            fit[plan['fit_indices']] = True
            stop[plan['stop_indices']] = True
            outer = np.asarray(plan['outer_indices'], dtype=int)
            check_plan(frame, fit, stop, outer, original)
            source = {}
            for feature in protocol['source_features']:
                target = OUT / 'sources' / feature / f'repeat_{repeat}/fold_{fold}'
                for seed in range(3):
                    c = torch.load(target / f'seed_{seed}/model.pt', map_location='cpu', weights_only=False)
                    audit(c, frame, banks[feature], fit, stop, outer, digest)
                    assert c['cfg'] == {**configs[feature], 'seed': seed}
                    assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
                    checked.append({'feature': feature, 'repeat': repeat, 'fold': fold, 'seed': seed})
                saved = np.load(target / 'outer_predictions.npz')
                assert np.array_equal(saved['indices'], outer)
                source[feature] = saved['p']
                assert np.isfinite(saved['p']).all() and np.allclose(saved['p'].sum(1), 1, atol=1e-5)
            base = source['multiscale_vit']
            combined = [base, .75 * base + .25 * source['vitb_raw768'],
                        .75 * base + .25 * source['vitb_up768']]
            for arm, p in zip(ARMS, combined):
                target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                saved = np.load(target / 'outer_predictions.npz')
                assert np.array_equal(saved['indices'], outer) and np.array_equal(saved['p'], p)
                assert np.array_equal(saved['pred'], p.argmax(1))
                actual = metrics(frame.iloc[outer].label, p)
                recorded = json.loads((target / 'metrics.json').read_text())['metrics']
                for key in ['macro_f1', 'accuracy', 'log_loss']:
                    assert abs(actual[key] - recorded[key]) < 1e-12
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
    assert len(checked) == protocol['planned_new_seed_fits'] == 54
    rows = []
    for arm in ARMS:
        folds = [json.loads((OUT / arm / f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
                 for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        row = {'arm': arm, 'folds': folds,
               'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
               'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)),
               'mean_accuracy': float(np.mean([f['metrics']['accuracy'] for f in folds]))}
        if arm != ARMS[0]:
            row['paired_vs_reference'] = paired(ARMS[0], arm, frame, protocol['outer_split_seeds'])
        rows.append(row)
    rows[1]['paired_vs_matched_upsampling'] = paired(ARMS[2], ARMS[1], frame, protocol['outer_split_seeds'])
    passed = rows[1]['mean_macro_f1'] > rows[0]['mean_macro_f1'] and rows[1]['paired_vs_reference']['macro_f1_ci95'][0] > 0
    destination = ROOT / 'results/research_resolution_repeat'
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'checked_seed_fits': len(checked), 'actual_models': checked,
        'checks': ['Only original training patients in every plan', 'Fit, stop and outer patients and duplicate groups disjoint',
                   'Actual checkpoint normalization and disease mappings fit inner training only',
                   'Every training image appears exactly once per repeat',
                   'Every fixed composite equals its corresponding source probabilities exactly',
                   'All reported metrics reconstructed from saved probabilities']})
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'arms': rows,
        'confirmation_gate_passed': passed, 'new_seed_fits': len(checked), 'external_test_evaluated': False,
        'decision': 'Eligible for implementation and model-freezing review; no automatic external scoring.' if passed
                    else 'Keep milestone5; fixed expert did not pass predeclared repeated-patient confirmation.'})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': len(checked),
         'confirmation_gate_passed': passed, 'external_test_evaluated': False})
    print(json.dumps({'confirmation_gate_passed': passed, 'arms': [{k: r[k] for k in ['arm', 'mean_macro_f1', 'sd_macro_f1']} for r in rows]}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

"""New patient partitions confirm a frozen coarse Gaussian evidence recipe."""
import fcntl
import hashlib
import json
import os
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, train_one, metrics, dump
from repeat_patient_cv import audit
from resolution_repeat_cv import check_plan
from gaussian_experiments import collect
from audit_gaussian_evidence import patient_moments
from patient_gaussian import fit_evidence, evidence_probability, mix_probability
from repeated_oof import paired_repeated

OUT = ROOT / 'runs/gaussian_repeat_cv'


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'results/research_gaussian_evidence/verification.json').read_text())['passed']
    selection_path = ROOT / 'results/research_gaussian_evidence/selection_before_repeat.json'
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == protocol['selection_record_sha256']
    chosen = json.loads(selection_path.read_text())['candidate']
    assert chosen['candidate_id'] == protocol['selected_candidate']
    assert chosen['granularity'] == 'coarse'
    assert chosen['weight'] == protocol['weight'] == .1
    assert chosen['shrinkage'] == protocol['covariance_shrinkage'] == .1
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    tf = frame.iloc[original]
    for ri, repeat in enumerate(protocol['outer_split_seeds']):
        splits = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        for fold, (fit_local, outer_local) in enumerate(splits.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[fit_local]]
            outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(groups.index, test_size=.2,
                random_state=20270124 + ri * 10 + fold, stratify=groups)
            fit = np.zeros(len(frame), dtype=bool)
            stop = fit.copy()
            fit[possible.index] = possible.split_group.isin(fit_ids)
            stop[possible.index] = possible.split_group.isin(stop_ids)
            check_plan(frame, fit, stop, outer, original)
            plan = {'manifest_sha256': digest, 'fit_indices': np.flatnonzero(fit).tolist(),
                    'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()}
            plan_path = OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json'
            if plan_path.exists():
                assert json.loads(plan_path.read_text()) == plan
            else:
                dump(plan_path, plan)
            fold_predictions = {arm: [] for arm in protocol['arms']}
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training_source_head', 'pid': os.getpid(),
                     'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                source = OUT / 'sources' / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                train_one(frame, features, digest, {**reference['config'], 'seed': seed}, source, fit, stop)
                c = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
                audit(c, frame, features, fit, stop, outer, digest)
                model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval().cuda()
                model.load_state_dict(c['state_dict'], strict=True)
                fitting = np.flatnonzero(fit)
                z, p = collect(model, features, np.r_[fitting, outer], c['stats'])
                fit_z, outer_z, base = z[:len(fitting)], z[len(fitting):], p[len(fitting):]
                np.savez_compressed(source / 'embeddings.npz', fit_indices=fitting,
                    outer_indices=outer, fit_embedding=fit_z, outer_embedding=outer_z, original_probability=base)
                fold_predictions[protocol['arms'][0]].append(base)
                for index, grain in enumerate(['coarse', 'subtype'], start=1):
                    evidence = fit_evidence(frame.iloc[fitting], fit_z, grain, protocol['covariance_shrinkage'])
                    np.savez_compressed(source / f'{grain}.npz', **evidence)
                    q = evidence_probability(evidence, outer_z)
                    fold_predictions[protocol['arms'][index]].append(mix_probability(base, q, protocol['weight']))
                del model
                torch.cuda.empty_cache()
            for arm, predictions in fold_predictions.items():
                target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                p = np.mean(predictions, 0)
                np.savez_compressed(target / 'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
                report = {'repeat': repeat, 'fold': fold, 'metrics': metrics(frame.iloc[outer].label, p)}
                dump(target / 'metrics.json', report)
                print(json.dumps({'event': 'gaussian_repeat_fold', 'arm': arm, **report}), flush=True)
    verify_and_report(frame, features, digest, protocol)


def verify_and_report(frame, features, digest, protocol):
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    checked = []
    for repeat in protocol['outer_split_seeds']:
        seen = []
        for fold in range(3):
            plan = json.loads((OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json').read_text())
            fit = np.zeros(len(frame), dtype=bool)
            stop = fit.copy()
            fit[plan['fit_indices']] = True
            stop[plan['stop_indices']] = True
            outer = np.asarray(plan['outer_indices'], dtype=int)
            check_plan(frame, fit, stop, outer, original)
            probabilities = {arm: [] for arm in protocol['arms']}
            for seed in range(3):
                source = OUT / 'sources' / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                c = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
                audit(c, frame, features, fit, stop, outer, digest)
                cache = np.load(source / 'embeddings.npz')
                assert np.array_equal(cache['fit_indices'], np.flatnonzero(fit))
                assert np.array_equal(cache['outer_indices'], outer)
                probabilities[protocol['arms'][0]].append(cache['original_probability'])
                for index, grain in enumerate(['coarse', 'subtype'], start=1):
                    fitted = dict(np.load(source / f'{grain}.npz'))
                    centers, covariance, parents, counts = patient_moments(frame.loc[fit], cache['fit_embedding'].astype('float64'), grain)
                    variance = max(float(np.trace(covariance) / len(covariance)), 1e-10)
                    expected = .9 * covariance + .10001 * variance * np.eye(len(covariance))
                    assert np.allclose(fitted['means'], centers, atol=1e-6, rtol=1e-6)
                    assert np.allclose(fitted['covariance'], expected, atol=1e-6, rtol=1e-6)
                    assert np.array_equal(fitted['parents'], parents) and np.array_equal(fitted['patient_counts'], counts)
                    q = evidence_probability(fitted, cache['outer_embedding'])
                    probabilities[protocol['arms'][index]].append(mix_probability(cache['original_probability'], q, protocol['weight']))
                checked.append({'repeat': repeat, 'fold': fold, 'seed': seed})
            for arm, values in probabilities.items():
                target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                saved = np.load(target / 'outer_predictions.npz')
                p = np.mean(values, 0)
                assert np.array_equal(saved['indices'], outer) and np.array_equal(saved['p'], p)
                actual = metrics(frame.iloc[outer].label, p)
                recorded = json.loads((target / 'metrics.json').read_text())['metrics']
                for key in ['accuracy', 'macro_f1', 'log_loss']:
                    assert abs(actual[key] - recorded[key]) < 1e-12
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
    assert len(checked) == protocol['planned_new_mlp_seed_fits'] == 18
    rows = []
    for arm in protocol['arms']:
        folds = [json.loads((OUT / arm / f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
                 for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        row = {'arm': arm, 'folds': folds,
            'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
            'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)),
            'mean_accuracy': float(np.mean([f['metrics']['accuracy'] for f in folds]))}
        if rows:
            row['paired_vs_reference'] = paired_repeated(OUT, protocol['arms'][0], arm, frame, protocol['outer_split_seeds'])
        rows.append(row)
    rows[1]['paired_vs_subtype_control'] = paired_repeated(OUT, protocol['arms'][2], protocol['arms'][1], frame, protocol['outer_split_seeds'])
    passed = rows[1]['mean_macro_f1'] > rows[0]['mean_macro_f1'] and rows[1]['paired_vs_reference']['macro_f1_ci95'][0] > 0
    destination = ROOT / 'results/research_gaussian_repeat'
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'mlp_seed_fits_checked': 18, 'generative_fits_checked': 36, 'actual_source_heads': checked,
        'checks': ['Only original training patients in every outer/stop/fit plan',
                   'Each source head normalization and subtype mapping fit only inner training patients',
                   'All36 prototype means and covariance matrices independently reconstructed from per-patient moments',
                   'All fixed mixtures and fold metrics reconstructed from saved source probabilities',
                   'Exactly one outer prediction per training image in each repeat']})
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'arms': rows, 'confirmation_gate_passed': passed, 'new_mlp_seed_fits': 18,
        'generative_fits': 36, 'external_test_evaluated': False,
        'decision': 'Eligible for frozen-inference review; no automatic external scoring.' if passed
                    else 'Keep milestone5; fixed Gaussian evidence did not pass predeclared repeated-patient confirmation.'})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_mlp_seed_fits': 18,
        'generative_fits': 36, 'confirmation_gate_passed': passed, 'external_test_evaluated': False})


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

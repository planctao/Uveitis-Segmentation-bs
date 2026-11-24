"""Four frozen Gaussian experts on existing normal-reference MLP embeddings."""
import fcntl
import json
import os
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data, normalize, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from normal_subspace import fit_normal_geometry
from verify_normal_subspace import independent_geometry
from patient_gaussian import fit_evidence, evidence_probability
from normal_reference_gaussian import collect_source, verify_evidence, independent_probability, composite_probability

OUT = ROOT / 'runs/normal_reference_gaussian'
DEST = ROOT / 'results/research_normal_reference_gaussian'


def verify_frozen(protocol):
    assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    for name, expected in protocol['code_sha256'].items():
        assert sha(ROOT / name) == expected
    for name, expected in {**protocol['source_checkpoints'], **protocol['frozen_artifacts']}.items():
        assert sha(name) == expected


def source_path(arm, phase, seed):
    folder = 'zero_control' if arm == 'raw' else 'normal_rank32_full'
    return ROOT / 'runs/normal_subspace' / folder / phase / f'seed_{seed}/model.pt'


def check_checkpoint(c, frame, features, fit, stop, digest, path, geometry):
    assert c['manifest_sha256'] == digest and c['diseases'] == sorted(frame.iloc[fit].disease.unique())
    disease_map = frame.iloc[fit].groupby('disease').label.first()
    assert c['dims']['disease_to_class'] == [int(disease_map[name]) for name in c['diseases']]
    history = json.loads((path.parent / 'history.json').read_text())
    assert max(history, key=lambda row: (row['val']['macro_f1'], -row['val']['log_loss']))['epoch'] == c['epoch']
    for key, values in features.items():
        assert np.isnan(values[2091:]).all()
        x = values[fit]; axes = tuple(range(x.ndim - 1))
        assert np.allclose(c['stats'][key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
        assert np.allclose(c['stats'][key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)
        assert np.array_equal(c['state_dict']['normal_center_' + key].numpy(), geometry[key + '_center'])
        assert np.array_equal(c['state_dict']['normal_basis_' + key].numpy(), geometry[key + '_basis'])


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text()); verify_frozen(protocol)
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256']
    for values in features.values():
        values[2091:] = np.nan
    records = [{**arm, 'candidate_id': arm['name'], 'folds': []} for arm in protocol['configuration_plan']]
    source_audits, generative_audits, control_metrics = [], [], []
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        evaluation = stop if phase == 'validation' else outer
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        mask = np.zeros(len(frame), dtype=bool); mask[fit] = True
        normalized, _ = normalize(features, mask)
        geometry = fit_normal_geometry(frame.iloc[fit].reset_index(drop=True), {key: v[fit] for key, v in normalized.items()})
        independent_geometry(frame.iloc[fit].reset_index(drop=True), {key: v[fit] for key, v in normalized.items()}, geometry)
        dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
        raw_predictions, normal_predictions = [], []
        gaussian_predictions = {arm['name']: [] for arm in protocol['configuration_plan']}
        for seed in range(3):
            dump(OUT / 'state.json', {'status': 'fitting_and_auditing', 'pid': os.getpid(), 'phase': phase, 'seed': seed,
                                     'new_seed_fits': 0, 'external_test_evaluated': False})
            folder = OUT / 'sources' / phase / f'seed_{seed}'; folder.mkdir(parents=True, exist_ok=True)
            probabilities = {}
            for arm in ['raw', 'normal']:
                path = source_path(arm, phase, seed); c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['cfg']['normal_strength'] == (0. if arm == 'raw' else 1.) and c['cfg']['normal_rank'] == 32
                check_checkpoint(c, frame, features, fit, stop, digest, path, geometry)
                indices = np.r_[fit, stop, outer] if arm == 'normal' else np.r_[stop, outer]
                z, p = collect_source(c, features, indices)
                other_z, other_p = collect_source(c, features, indices, independent=True)
                assert np.allclose(z, other_z, atol=5e-5, rtol=5e-5)
                assert np.allclose(p, other_p, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), other_p.argmax(1))
                offset = len(fit) if arm == 'normal' else 0
                stop_p = p[offset:offset + len(stop)]
                saved = np.load(path.parent / 'val_predictions.npz')['p']
                assert np.allclose(stop_p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(stop_p.argmax(1), saved.argmax(1))
                evaluation_p = stop_p if phase == 'validation' else p[offset + len(stop):]
                probabilities[arm] = evaluation_p
                if arm == 'normal':
                    fit_z = z[:len(fit)]; stop_z = z[len(fit):len(fit) + len(stop)]
                    evaluation_z = stop_z if phase == 'validation' else z[len(fit) + len(stop):]
                    np.savez_compressed(folder / 'normal_embeddings.npz', fit_indices=fit, stop_indices=stop, evaluation_indices=evaluation,
                        fit_embedding=fit_z, stop_embedding=stop_z, evaluation_embedding=evaluation_z)
                source_audits.append({'arm': arm, 'phase': phase, 'seed': seed, 'checkpoint': str(path), 'checkpoint_sha256': sha(path)})
            raw_predictions.append(probabilities['raw']); normal_predictions.append(probabilities['normal'])
            for row in records:
                evidence = fit_evidence(frame.iloc[fit], fit_z, row['granularity'], row['shrinkage'])
                np.savez_compressed(folder / (row['name'] + '.npz'), **evidence)
                restored = dict(np.load(folder / (row['name'] + '.npz')))
                moments = verify_evidence(frame.iloc[fit], fit_z, restored, row['granularity'], row['shrinkage'])
                q = evidence_probability(restored, np.r_[stop_z, evaluation_z])
                independent = independent_probability(moments, np.r_[stop_z, evaluation_z])
                assert np.allclose(q, independent, atol=3e-6, rtol=3e-6) and np.array_equal(q.argmax(1), independent.argmax(1))
                gaussian_predictions[row['name']].append(q[len(stop_z):])
                np.savez_compressed(folder / (row['name'] + '_probabilities.npz'), stop_p=q[:len(stop_z)], evaluation_p=q[len(stop_z):])
                generative_audits.append({'phase': phase, 'seed': seed, 'name': row['name'], 'model_sha256': sha(folder / (row['name'] + '.npz'))})
            np.savez_compressed(folder / 'source_probabilities.npz', raw=probabilities['raw'], normal=probabilities['normal'])
        raw = np.mean(raw_predictions, axis=0); normal = np.mean(normal_predictions, axis=0)
        for arm, actual in [('zero_control', raw), ('normal_rank32_full', normal)]:
            cached = np.load(ROOT / 'runs/normal_subspace' / arm / phase / 'predictions.npz')
            assert np.array_equal(cached['indices'], evaluation)
            assert np.allclose(actual, cached['p'], atol=5e-5, rtol=5e-5) and np.array_equal(actual.argmax(1), cached['pred'])
        base = (.25 * raw + .75 * normal).astype('float32')
        control_metrics.append({'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, base)})
        control_dir = OUT / 'm6_control' / phase; control_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(control_dir / 'predictions.npz', indices=evaluation, p=base, pred=base.argmax(1))
        if phase != 'validation':
            np.savez_compressed(control_dir / 'outer_predictions.npz', indices=evaluation, p=base, pred=base.argmax(1))
        for row in records:
            q = np.mean(gaussian_predictions[row['name']], axis=0)
            p = composite_probability(raw, normal, q)
            independent = (.225 * raw.astype('float64') + .675 * normal.astype('float64') + .1 * q.astype('float64')).astype('float32')
            assert np.allclose(p, independent, atol=2e-7) and np.array_equal(p.argmax(1), independent.argmax(1))
            target = OUT / row['name'] / phase; target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            score = metrics(frame.iloc[evaluation].label, p); dump(target / 'metrics.json', score)
            if phase == 'validation':
                row['val'] = score
            else:
                row['folds'].append({'phase': phase, 'metrics': score})
                np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
        print(json.dumps({'event': 'normal_reference_gaussian_phase_audited', 'phase': phase,
                          'reused_source_heads': 6, 'generative_models': 12}), flush=True)
    assert len(source_audits) == 24 and len(generative_audits) == 48
    assert abs(control_metrics[0]['metrics']['macro_f1'] - protocol['minimum_validation_macro_f1']) < 1e-12
    assert abs(np.mean([row['metrics']['macro_f1'] for row in control_metrics[1:]]) - protocol['minimum_cv_macro_f1']) < 1e-12
    for row in records:
        seen = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            ids = stop if phase == 'validation' else outer
            saved = np.load(OUT / row['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], ids)
            recomputed = metrics(frame.iloc[ids].label, saved['p'])
            stored = json.loads((OUT / row['name'] / phase / 'metrics.json').read_text())
            for name in ['accuracy', 'macro_f1', 'log_loss']:
                assert recomputed[name] == stored[name]
            if phase != 'validation':
                seen.extend(ids.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row['cv'] = {'mean_macro_f1': float(np.mean([fold['metrics']['macro_f1'] for fold in row['folds']])), 'folds': row.pop('folds')}
        row['development_gate_passed'] = row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and row['val']['macro_f1'] >= protocol['minimum_validation_macro_f1']
        row['paired_original_oof_vs_m6'] = paired_oof(OUT / 'm6_control', OUT / row['name'], frame)
    verify_frozen(protocol)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records, 'base_m6': control_metrics,
        'new_seed_fits': 0, 'generative_fits': 48, 'reused_source_heads': 24, 'external_test_evaluated': False})
    dump(DEST / 'verification.json', {'passed': True, 'actual_seed_heads_checked': 0, 'reused_source_heads_checked': 24,
        'generative_models_checked': 48, 'sources': source_audits, 'models': generative_audits,
        'checks': ['All actual source fit/stop/outer embeddings and probabilities independently restored with NumPy projector plus plain M5',
                   'All normal geometry buffers and original M6 development predictions restored',
                   'All Gaussian means, patient covariance and priors independently rebuilt; direct covariance quadratic scoring matches all stop/outer probabilities',
                   'All four fixed .225 raw/.675 normal/.1 normal-Gaussian mixtures independently reconstructed',
                   'Complete development patient coverage and all metrics checked; no external outcomes loaded']})
    eligible = [row for row in records if row['development_gate_passed']]
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']),
            'source_checkpoint_sha256': protocol['source_checkpoints'], 'gaussian_models': generative_audits,
            'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(DEST / 'freeze_integrity_after_training.json', {'passed': True, 'all_source_and_code_hashes_unchanged': True,
        'time_utc': datetime.now(timezone.utc).isoformat(), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 0, 'generative_fits': 48,
        'reused_source_heads': 24, 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

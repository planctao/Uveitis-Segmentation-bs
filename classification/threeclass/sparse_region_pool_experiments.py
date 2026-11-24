"""Predeclared softmax temperature and known sparse-pooling controls on M5."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from sparse_region_pool import SparseRegionModel
from verify_sparse_region_pool import independent_attention

OUT = ROOT / 'runs/sparse_region_pool'
DEST = ROOT / 'results/research_sparse_region_pool'


def configuration(reference, arm):
    return {**reference['selected']['config'], 'region_pool_kind': arm['kind'],
            'region_pool_temperature': arm['temperature']}


@torch.inference_mode()
def predict_cpu(c, features, indices, diagnostics=False):
    assert len(indices) and np.max(indices) < 2091
    model=SparseRegionModel(c['cfg'],c['dims'],len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'],strict=True)
    ps=[];entropies=[];supports=[];effective=[];maxima=[];errors=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256]
        data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        assert all(np.isfinite(v).all() for v in data.values())
        out=model({k:torch.from_numpy(v) for k,v in data.items()});ps.append(out['p'].numpy())
        if diagnostics:
            actual=out['pool_attention'].numpy()
            expected=independent_attention(out['pool_scores'].numpy(),c['cfg']['region_pool_kind'],c['cfg']['region_pool_temperature'])
            assert np.allclose(actual,expected,atol=2e-6,rtol=2e-6)
            errors.append(float(np.abs(actual-expected).max()))
            entropies.append(-(actual*np.log(np.maximum(actual,1e-12))).sum(1)/np.log(actual.shape[1]))
            supports.append((actual>0).sum(1));effective.append(1/np.square(actual).sum(1));maxima.append(actual.max(1))
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_normalized_attention_entropy':float(np.concatenate(entropies).mean()),
        'mean_nonzero_regions':float(np.concatenate(supports).mean()),
        'mean_effective_regions':float(np.concatenate(effective).mean()),
        'mean_maximum_weight':float(np.concatenate(maxima).mean()),
        'max_independent_attention_error':max(errors)}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    proof = json.loads((ROOT / 'verification_sparse_region_pool.json').read_text()); assert proof['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256'] == proof['manifest_sha256']
    # Every explicit prediction is also checked, but mark inaccessible rows here.
    for values in features.values(): values[2091:] = np.nan
    records = []
    for arm in protocol['configuration_plan']:
        cfg = configuration(reference, arm); target = OUT / arm['name']; folds = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            fit_mask = np.zeros(len(frame), dtype=bool); stop_mask = fit_mask.copy()
            fit_mask[fit] = True; stop_mask[stop] = True; ps = []
            dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                    'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target / phase / f'seed_{seed}'
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, fit_mask, stop_mask, model_class=SparseRegionModel)
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); folds.append(result)
        record = summarize_group(target / 'validation', frame)
        record.update(name=arm['name'], candidate_id=arm['name'], pool_configuration=arm)
        record['cv'] = {'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
            'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)), 'folds': folds}
        record['paired_original_oof_vs_reference'] = paired_oof(Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'sparse_region_pool_arm_complete', 'arm': arm['name'], 'val_f1': record['val']['macro_f1'],
            'cv_f1': record['cv']['mean_macro_f1'], 'development_gate_passed': record['development_gate_passed']}), flush=True)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
        'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'cpu_auditing', 'pid': os.getpid(), 'external_test_evaluated': False})
    audit(frame, features, digest, reference, records)
    eligible = [r for r in records if r['development_gate_passed']]
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
            'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def audit(frame, features, digest, reference, records):
    checked = []; diagnostics = []
    for record in records:
        seen = []; scores = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []
            for key in ['patient_group', 'split_group']:
                sets = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'; c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert c['cfg'] == {**configuration(reference, record['pool_configuration']), 'seed': seed}
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                for key, values in features.items():
                    assert np.isnan(values[2091:]).all()
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p, detail = predict_cpu(c, features, evaluation, True); ps.append(p)
                diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed, **detail})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed, 'checkpoint_sha256': sha(folder / 'model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1)) and np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p']); stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']: assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation': seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'sparse_region_pool_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'attention_diagnostics.json', {'models': diagnostics})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest, 'actual_seed_heads_checked': 48, 'fits': checked,
        'checks': ['All four predeclared softmax-temperature/entmax1.5/sparsemax configurations completed twelve actual heads each',
            'Fit/stop/outer patients and duplicate groups, fit-only normalization and disease marginalization reconstructed',
            'Every stopping/outer probability and class decision restored on CPU and all fold metrics reproduced',
            'Every actual trained pooling probability independently recovered with scalar SciPy root solutions',
            'Attention entropy and nonzero/effective region counts retained; exact outer coverage; external rows inaccessible']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
            ['sparse_region_pool.py', 'verify_sparse_region_pool.py', 'sparse_region_pool_experiments.py', 'experiment.py']}})
        try: run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()

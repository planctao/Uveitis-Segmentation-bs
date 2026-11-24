"""Predeclared uniform/content/position relation ablation on fixed M5."""
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
from regional_relation import RelationModel
from verify_regional_relation import independent_relation

OUT = ROOT / 'runs/regional_relation'
DEST = ROOT / 'results/research_regional_relation'


def configuration(reference, arm):
    return {**reference['selected']['config'], 'relation_rank': arm['rank'],
            'relation_kind': arm['kind'], 'relation_position': arm['position']}


@torch.inference_mode()
def predict_cpu(c, features, indices, diagnostics=False):
    assert len(indices) and np.max(indices) < 2091
    model = RelationModel(c['cfg'].copy(), c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    ps = []; off = []; relative = []; entropies = []; distances = []; maximum = []; errors = []
    xy = np.stack(np.meshgrid(np.arange(3), np.arange(3), indexing='ij'), -1).reshape(9, 2)
    spatial_distance = np.abs(xy[:, None] - xy[None, :]).sum(-1)
    for begin in range(0, len(indices), 256):
        ids = indices[begin:begin + 256]
        data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
        assert all(np.isfinite(v).all() for v in data.values())
        x = {k: torch.from_numpy(v) for k, v in data.items()}; out = model(x); ps.append(out['p'].numpy())
        if diagnostics:
            delta, attention = independent_relation(model.relation, out['relation_input'].numpy())
            assert np.allclose(out['relation_delta'].numpy(), delta, atol=2e-5, rtol=2e-5)
            assert np.allclose(out['relation_attention'].numpy(), attention, atol=2e-6, rtol=2e-6)
            errors.append(float(np.max(np.abs(out['relation_delta'].numpy() - delta))))
            relative.append((out['relation_delta'].norm(dim=-1) / out['original_region_projection'].norm(dim=-1).clamp(min=1e-7)).numpy())
            entropies.append(-(attention * np.log(np.maximum(attention, 1e-12))).sum(-1) / np.log(8))
            distances.append((attention * spatial_distance).sum(-1)); maximum.append(attention.max(-1))
            model.cfg['relation_disabled'] = True; off.append(model(x)['p'].numpy()); model.cfg.pop('relation_disabled')
    if not diagnostics: return np.concatenate(ps)
    parameters = {k: float(v.norm()) for k, v in model.relation.named_parameters()}
    return np.concatenate(ps), {'mean_relative_projection_delta': float(np.concatenate(relative).mean()),
        'max_relative_projection_delta': float(np.concatenate(relative).max()),
        'mean_normalized_relation_entropy': float(np.concatenate(entropies).mean()),
        'mean_relation_manhattan_distance': float(np.concatenate(distances).mean()),
        'mean_maximum_relation_weight': float(np.concatenate(maximum).mean()),
        'max_independent_delta_error': max(errors), 'parameter_norms': parameters, 'branch_off_p': np.concatenate(off)}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    proof = json.loads((ROOT / 'verification_regional_relation.json').read_text()); assert proof['passed']
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
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, fit_mask, stop_mask, model_class=RelationModel)
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); folds.append(result)
        record = summarize_group(target / 'validation', frame)
        record.update(name=arm['name'], candidate_id=arm['name'], relation_configuration=arm)
        record['cv'] = {'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
            'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)), 'folds': folds}
        record['paired_original_oof_vs_reference'] = paired_oof(Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'regional_relation_arm_complete', 'arm': arm['name'], 'val_f1': record['val']['macro_f1'],
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
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []; off = []
            for key in ['patient_group', 'split_group']:
                sets = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'; c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert c['cfg'] == {**configuration(reference, record['relation_configuration']), 'seed': seed}
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                for key, values in features.items():
                    assert np.isnan(values[2091:]).all()
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p, detail = predict_cpu(c, features, evaluation, True); ps.append(p); off.append(detail.pop('branch_off_p'))
                assert detail['mean_relative_projection_delta'] > 0
                assert detail['parameter_norms']['readout.weight'] > 0
                diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed, **detail})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed, 'checkpoint_sha256': sha(folder / 'model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1)) and np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p']); stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']: assert abs(actual[key] - stored[key]) < 1e-12
            dump(OUT / record['name'] / phase / 'branch_diagnostic.json', {'branch_on': actual,
                'branch_off': metrics(frame.iloc[evaluation].label, np.mean(off, 0)),
                'limitation': 'Disabling a jointly trained branch does not replace an independently trained M5 control.'})
            if phase != 'validation': seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'regional_relation_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'relation_diagnostics.json', {'models': diagnostics})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest, 'actual_seed_heads_checked': 48, 'fits': checked,
        'checks': ['All four predeclared uniform/content/rank/relative-position configurations completed twelve actual heads each',
            'Fit/stop/outer patients and duplicate groups, fit-only normalization and disease marginalization reconstructed',
            'Every stopping/outer probability and class decision restored on CPU and all fold metrics reproduced',
            'Every actual relation attention and projection residual independently recovered with double-precision NumPy/SciPy',
            'Relation entropy/distance/norms and branch-off diagnostics retained; exact outer coverage; external rows inaccessible']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
            ['regional_relation.py', 'verify_regional_relation.py', 'regional_relation_experiments.py', 'experiment.py']}})
        try: run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()

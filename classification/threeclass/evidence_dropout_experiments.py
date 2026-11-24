"""Fixed train-only evidence-group dropout and exact original M5 controls."""
import fcntl
import functools
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from evidence_dropout import EvidenceDropoutModel
from verify_evidence_dropout import independent_witness

OUT = ROOT / 'runs/evidence_dropout'
DEST = ROOT / 'results/research_evidence_dropout'


def configuration(reference, arm):
    return {**reference['selected']['config'], 'evidence_drop_probability': arm['probability']}


@torch.inference_mode()
def predict_cpu(c, features, indices):
    assert len(indices) and np.max(indices) < 2091
    # Restore the deployed ordinary M5 model directly.
    model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    ps = []
    for begin in range(0, len(indices), 256):
        ids = indices[begin:begin + 256]
        data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
        ps.append(model({k: torch.from_numpy(v) for k, v in data.items()})['p'].numpy())
    return np.concatenate(ps)

def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_evidence_dropout.json').read_text())['passed']
    for name, expected in protocol['code_sha256'].items(): assert sha(ROOT / name) == expected
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256']
    for value in features.values(): value[2091:] = np.nan
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
            dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(),
                 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                     'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target / phase / f'seed_{seed}'
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, fit_mask, stop_mask,
                          model_class=functools.partial(EvidenceDropoutModel, witness_path=folder / 'first_mask_witness.npz'))
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
                folds.append(result)
        record = summarize_group(target / 'validation', frame)
        record.update(name=arm['name'], candidate_id=arm['name'], dropout_configuration=arm)
        record['cv'] = {'mean_macro_f1': float(np.mean([x['metrics']['macro_f1'] for x in folds])),
                        'sd_macro_f1': float(np.std([x['metrics']['macro_f1'] for x in folds], ddof=1)), 'folds': folds}
        record['paired_original_oof_vs_reference'] = paired_oof(
            Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'evidence_dropout_arm_complete', 'arm': arm['name'],
            'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1'],
            'development_gate_passed': record['development_gate_passed']}), flush=True)
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
    checked = []; diagnostics = []; controls = []; reference_schemas = {}
    for record in records:
        seen = []; scores = []
        for phase, fit, stop, outer, sources in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert c['cfg'] == {**configuration(reference, record['dropout_configuration']), 'seed': seed}
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                if phase not in reference_schemas:
                    source = torch.load(sources[0], map_location='cpu', weights_only=False)
                    reference_schemas[phase] = {k: tuple(v.shape) for k, v in source['state_dict'].items()}
                assert {k: tuple(v.shape) for k, v in c['state_dict'].items()} == reference_schemas[phase]
                for key, values in features.items():
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                ps.append(p if phase == 'validation' else predict_cpu(c, features, outer))
                if c['cfg']['evidence_drop_probability']:
                    proof = folder / 'first_mask_witness.npz'
                    diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed,
                        'witness_sha256': sha(proof), **independent_witness(np.load(proof))})
                else:
                    original = torch.load(sources[seed], map_location='cpu', weights_only=False)
                    original_p = predict_cpu(original, features, evaluation)
                    assert np.allclose(ps[-1], original_p, atol=5e-5, rtol=5e-5)
                    assert np.array_equal(ps[-1].argmax(1), original_p.argmax(1))
                    controls.append({'phase': phase, 'seed': seed, 'all_source_decisions_reproduced': True})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed,
                                'checkpoint_sha256': sha(folder / 'model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':
                seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'evidence_dropout_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48 and len(diagnostics) == 36 and len(controls) == 12
    dump(DEST / 'mask_diagnostics.json', {'models': diagnostics, 'controls': controls})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest,
         'actual_seed_heads_checked': 48, 'fits': checked, 'checks': [
             'Allfourfixedtrain-onlyevidencegroupdropoutprobabilities completedtwelveactualheads each',
             'Fit/stop/outer patient and duplicate partitions checked; fit-only statistics and disease maps reconstructed',
             'Every stopping and outer probability, class decision and ensemble metric restored with plain M5 Model on CPU',
             'All checkpoints have exactly original M5 tensor schema and require one inference pass',
             'All36actualfirstmaskgroupgain/zeroing witnesses independentlyreconstructed; all12zero-probabilitycontrols reproduceoriginalM5heldoutdecisions',
             'Exact original training outer coverage reconstructed; external results never used']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
             ['evidence_dropout.py', 'evidence_dropout_experiments.py', 'verify_evidence_dropout.py', 'experiment.py']}})
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

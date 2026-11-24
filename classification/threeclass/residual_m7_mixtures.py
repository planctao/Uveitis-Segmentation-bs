"""Eight predeclared residual/M7 mixtures; no fitting or external evaluation."""
import fcntl
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT, dump, metrics
from kernel_residual_experiments import sha, phase_plans

OUT = ROOT / 'runs/residual_m7_mixtures'
DEST = ROOT / 'results/research_residual_m7_mixtures'


def verify_frozen(protocol):
    assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in protocol[key].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT / path) == expected, name


def sources(phase, arm):
    return [ROOT / 'runs/normal_subspace' / branch / phase / 'predictions.npz'
            for branch in ['zero_control', 'normal_rank32_full']] + [
        ROOT / 'runs/normal_large_reference/large_dual' / phase / 'predictions.npz',
        ROOT / 'runs/normal_large_residual' / arm / phase / 'predictions.npz']


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    verify_frozen(protocol)
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    assert sha(ROOT / 'metadata/manifest.csv') == protocol['manifest_sha256']
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    plans = list(phase_plans(frame, reference))
    records, checked = [], []
    for arm in protocol['configuration_plan']:
        folds = []
        for phase, fit, stop, outer, _ in plans:
            dump(OUT / 'state.json', {'status': 'evaluating_fixed_development_mixtures',
                 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'external_test_evaluated': False})
            ids = stop if phase == 'validation' else outer
            assert max(ids) < 2091
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[z][key]) for z in [fit, stop, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            arrays = [np.load(p) for p in sources(phase, arm['residual_arm'])]
            for z in arrays:
                assert np.array_equal(z['indices'], ids)
                assert np.isfinite(z['p']).all() and np.all(z['p'] >= 0)
                assert np.allclose(z['p'].sum(1), 1, atol=1e-5)
            raw, normal, large, residual = [z['p'] for z in arrays]
            # Match the original M7 arithmetic, and also independently expand all coefficients.
            m7 = (.75 * (.25 * raw + .75 * normal).astype('float32') + .25 * large).astype('float32')
            retained = np.load(ROOT / 'runs/normal_large_blend/large_dual_weight0p25' / phase / 'predictions.npz')
            assert np.array_equal(retained['indices'], ids) and np.array_equal(m7, retained['p'])
            weight = arm['weight']
            p = ((1 - weight) * m7 + weight * residual).astype('float32')
            independent = sum(coef * value.astype('float64') for coef, value in zip(
                [(1-weight)*.1875, (1-weight)*.5625, (1-weight)*.25, weight],
                [raw, normal, large, residual])).astype('float32')
            assert np.allclose(p, independent, atol=2e-7, rtol=2e-7)
            assert np.array_equal(p.argmax(1), independent.argmax(1))
            folder = OUT / arm['name'] / phase
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'predictions.npz', indices=ids, p=p, pred=p.argmax(1))
            row = {'phase': phase, 'metrics': metrics(frame.iloc[ids].label, p)}
            dump(folder / 'metrics.json', row)
            if phase == 'validation':
                validation = row['metrics']
            else:
                np.savez_compressed(folder / 'outer_predictions.npz', indices=ids, p=p, pred=p.argmax(1))
                folds.append(row)
            checked.append({'candidate': arm['name'], 'phase': phase, 'images': len(ids),
                'probability_sha256': sha(folder / 'predictions.npz'),
                'maximum_independent_probability_error': float(np.max(np.abs(p-independent)))})
        record = {'name': arm['name'], 'candidate_id': arm['name'], 'mixture_configuration': arm,
            'val': validation, 'cv': {'folds': folds,
            'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}, 'new_seed_fits': 0}
        record['development_gate_passed'] = bool(record['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1']
            and validation['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(record)
        dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'residual_m7_mixture', 'name': arm['name'],
            'val': validation['macro_f1'], 'cv': record['cv']['mean_macro_f1'],
            'passed': record['development_gate_passed']}), flush=True)
    assert len(checked) == 32
    # Read physical predictions afresh, reconstruct independently, and verify complete outer coverage.
    for row in records:
        seen, scores = [], []
        arm = row['mixture_configuration']
        for phase, fit, stop, outer, _ in plans:
            ids = stop if phase == 'validation' else outer
            saved = np.load(OUT / arm['name'] / phase / 'predictions.npz')
            arrays = [np.load(path)['p'].astype('float64') for path in sources(phase, arm['residual_arm'])]
            w = arm['weight']
            manual = ((1-w)*(.1875*arrays[0]+.5625*arrays[1]+.25*arrays[2])+w*arrays[3]).astype('float32')
            assert np.array_equal(saved['indices'], ids) and np.array_equal(saved['pred'], manual.argmax(1))
            assert np.allclose(saved['p'], manual, atol=2e-7, rtol=2e-7)
            actual = metrics(frame.iloc[ids].label, saved['p'])
            expected = row['val'] if phase == 'validation' else next(x['metrics'] for x in row['cv']['folds'] if x['phase'] == phase)
            assert actual == expected
            if phase != 'validation':
                seen.extend(ids.tolist())
                scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert np.mean(scores) == row['cv']['mean_macro_f1']
    verify_frozen(protocol)
    eligible = [r for r in records if r['development_gate_passed']]
    dump(DEST / 'comparison.json', {'manifest_sha256': protocol['manifest_sha256'], 'protocol': protocol,
        'candidates': records, 'new_seed_fits': 0, 'reused_source_heads': 84, 'external_test_evaluated': False})
    dump(DEST / 'verification.json', {'passed': True, 'actual_seed_heads_checked': 0, 'new_seed_fits': 0,
        'reused_source_heads_checked': 84, 'fixed_candidate_scores': 8, 'fixed_mixture_phases_checked': checked,
        'checks': ['Eight fixed configurations frozen before evaluation; no fitting or external predictions',
            '28 physical source predictions and84 previously audited source checkpoints hash-protected',
            'Original M7 recovered bit-for-bit in all four phases; all mixtures independently expanded in float64',
            'All32 physical predictions, scores, decisions, patient isolation and complete outer coverage checked']})
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
            'source_prediction_sha256': protocol['source_prediction_sha256'],
            'source_checkpoint_sha256': protocol['source_checkpoint_sha256'],
            'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(DEST / 'freeze_integrity_after_training.json', {'passed': True,
        'time_utc': datetime.now(timezone.utc).isoformat(), 'all_frozen_code_and_sources_unchanged': True,
        'protocol_sha256': sha(OUT / 'protocol.json'), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 0,
        'reused_source_heads': 84, 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    hypothesis = ROOT / 'results/next_residual_m7_mixtures_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    dump(hypothesis, {**planned, 'status': 'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed',
        'scores_evaluated': 8, 'comparison': str((DEST / 'comparison.json').relative_to(ROOT)),
        'verification': str((DEST / 'verification.json').relative_to(ROOT))})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

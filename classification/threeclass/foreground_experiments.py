"""Matched paired-scale regional density/foreground ablations, all original folds."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, Model, normalize, predict, train_one, summarize_group, metrics, dump
from foreground_data import OUT, load_foreground
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name in ['verification_foreground_pool.json', 'verification_foreground_features.json']:
        assert json.loads((ROOT / name).read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    records = []
    for arm in protocol['configuration_plan']:
        frame, features, digest = load_foreground(arm['name'])
        config = {**reference['selected']['config'], 'use_region_mask': arm['use_region_mask']}
        group = OUT / arm['name'] / 'validation'
        for seed in range(3):
            dump(OUT / 'state.json', {'status': 'formal_training', 'pid': os.getpid(), 'arm': arm['name'], 'seed': seed, 'external_test_evaluated': False})
            train_one(frame, features, digest, {**config, 'seed': seed}, group / f'seed_{seed}')
        record = summarize_group(group, frame)
        record.update(name=arm['name'], candidate_id='foreground_' + arm['name'], feature_configuration=arm)
        fold_reports = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference)[1:]:
            fit_mask = np.zeros(len(frame), dtype=bool);stop_mask = fit_mask.copy()
            fit_mask[fit] = True;stop_mask[stop] = True
            assert frame.iloc[np.r_[fit, stop, outer]].split.eq('train').all()
            for key in ['patient_group', 'split_group']:
                sets = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
            probabilities = []
            target = OUT / arm['name'] / phase
            dump(target / 'split_plan.json', {'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'patient_cv_training', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                train_one(frame, features, digest, {**config, 'seed': seed}, target / f'seed_{seed}', fit_mask, stop_mask)
                c = torch.load(target / f'seed_{seed}/model.pt', map_location='cpu', weights_only=False)
                model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval().cuda()
                model.load_state_dict(c['state_dict'], strict=True)
                normalized, _ = normalize({key: value[outer] for key, value in features.items()}, None, c['stats'])
                x = {key: torch.from_numpy(value).cuda() for key, value in normalized.items()}
                p, _ = predict(model, x);probabilities.append(p)
                del model, x;torch.cuda.empty_cache()
            p = np.mean(probabilities, 0)
            np.savez_compressed(target / 'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
            result = {'phase': phase, 'metrics': metrics(frame.iloc[outer].label, p)}
            dump(target / 'metrics.json', result);fold_reports.append(result)
        record['cv'] = {'mean_macro_f1': float(np.mean([row['metrics']['macro_f1'] for row in fold_reports])),
            'sd_macro_f1': float(np.std([row['metrics']['macro_f1'] for row in fold_reports], ddof=1)), 'folds': fold_reports}
        source_cv = Path(reference['selected']['source_cv_summary']).parent
        record['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / arm['name'], frame)
        record['development_gate_passed'] = record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'foreground_arm_complete', 'name': arm['name'], 'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1']}), flush=True)
        del features
    destination = ROOT / 'results/research_foreground_multiscale'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
        'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed_training', 'pid': os.getpid(), 'new_seed_fits': 48,
        'passing_development_gates': sum(row['development_gate_passed'] for row in records), 'external_test_evaluated': False})
    from audit_foreground_experiments import main as audit
    audit()
    eligible = [row for row in records if row['development_gate_passed']]
    if eligible:dump(destination / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in ['foreground_data.py', 'foreground_experiments.py', 'foreground_pool.py']}})
        try:run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()

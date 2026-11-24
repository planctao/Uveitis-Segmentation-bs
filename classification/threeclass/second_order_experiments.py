"""Five fixed regional second-order heads, each with full patient CV."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import torch
from experiment import ROOT, load_data, train_one, summarize_group, dump
from milestone2 import paired_oof

OUT = ROOT / 'runs/regional_second_order'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_second_order.json').read_text())['passed']
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    records = []
    for index, arm in enumerate(protocol['configuration_plan']):
        group = ROOT / 'runs/multiscale_vit/regional_second_order' / f'{index:02d}_{arm["name"]}'
        for seed in range(3):
            dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(),
                'arm': arm['name'], 'seed': seed, 'external_test_evaluated': False})
            train_one(frame, features, digest, {**arm['config'], 'seed': seed}, group / f'seed_{seed}')
        record = summarize_group(group, frame)
        identity = json.dumps({'feature': arm['feature'], 'config': arm['config']}, sort_keys=True)
        record.update(feature=arm['feature'], name=arm['name'], ablation_index=index,
            candidate_id='second_order_' + hashlib.sha256(identity.encode()).hexdigest()[:12])
        dump(group / 'candidate.json', record)
        records.append(record)
        dump(OUT / 'leaderboard.json', records)
    for record in records:
        candidate = ROOT / 'runs/multiscale_vit/regional_second_order' / record['group'] / 'candidate.json'
        dump(OUT / 'state.json', {'status': 'patient_cv', 'pid': os.getpid(),
            'arm': record['name'], 'external_test_evaluated': False})
        subprocess.run([sys.executable, '-u', str(ROOT / 'crossvalidate.py'), '--features',
            record['feature'], '--candidate-json', str(candidate)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'audit_candidate_cv.py'),
            '--candidate-json', str(candidate)], cwd=ROOT, check=True)
        cv_directory = ROOT / 'runs/multiscale_vit/candidate_crossvalidation' / ('continuous_' + record['candidate_id'])
        cv = json.loads((cv_directory / 'summary.json').read_text())
        record['cv'] = {key: cv[key] for key in ['mean_macro_f1', 'sd_macro_f1', 'mean_accuracy']}
        record['cv_directory'] = str(cv_directory)
        record['paired_original_oof_vs_reference'] = paired_oof(Path(reference['selected']['source_cv_summary']).parent, cv_directory, frame)
        record['development_gate_passed'] = cv['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
    eligible = [row for row in records if row['development_gate_passed']]
    destination = ROOT / 'results/research_second_order'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'candidates': records, 'new_seed_fits': 60,
        'reference': {'candidate_id': reference['selected']['candidate_id'],
            'val_macro_f1': reference['selected']['val']['macro_f1'], 'cv_macro_f1': reference['cv']['mean_macro_f1']},
        'external_test_evaluated': False, 'external_files_read': False,
        'decision': 'Any eligible second-order head requires newly fixed repeated-patient confirmation.' if eligible
                    else 'No second-order head passed both development gates; preserve frozen models and all negative results.',
        'code_sha256': {name: sha(ROOT / name) for name in ['second_order_head.py', 'experiment.py',
            'second_order_experiments.py', 'verify_second_order.py', 'crossvalidate.py', 'audit_candidate_cv.py']}})
    dump(OUT / 'state.json', {'status': 'auditing_actual_models', 'pid': os.getpid(),
        'new_seed_fits': 60, 'external_test_evaluated': False})
    subprocess.run([sys.executable, '-u', str(ROOT / 'analyze_second_order.py')], cwd=ROOT, check=True)
    if eligible:
        winner = max(eligible, key=lambda row: (row['cv']['mean_macro_f1'], row['val']['macro_f1']))
        dump(destination / 'selection_before_repeat.json', {'candidate': winner,
            'selection_policy': 'Highest original patientCV among candidates with nondecreasing validation macro-F1.',
            'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 60,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    subprocess.run([sys.executable, '-u', str(ROOT / 'report.py')], cwd=ROOT, check=True)


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

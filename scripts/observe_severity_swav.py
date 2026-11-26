"""Audit and evaluate each seed only after all five SwAV folds finish."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/severity_research_20261013/domain_ssl'


def main():
    lock = (OUT / 'swav_observer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = ['scripts/collect_severity_domain_ssl.py', 'scripts/evaluate_severity_baselines.py']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    complete_seeds = []
    state_path = OUT / 'swav_observer_state.json'
    def update(status, **kwargs):
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=status, owner_pid=os.getpid(), completed_seeds=complete_seeds, sources_sha256=hashes, **kwargs), indent=2) + '\n')
        temporary.replace(state_path)
    for seed in [42, 2026, 3407]:
        while not all((OUT / 'swav' / f'seed{seed}' / fold / 'complete.json').exists() for fold in ['f1', 'f2', 'f3', 'f4', 'f5']):
            queue = json.loads((OUT / 'swav_queue_state.json').read_text())
            if queue['status'] == 'failed':
                update('training_failed', seed=seed, queue=queue)
                raise RuntimeError('SwAV training queue failed')
            try:
                os.kill(queue['owner_pid'], 0)
            except ProcessLookupError:
                update('queue_owner_missing', seed=seed)
                raise RuntimeError('Training owner absent before required fold completion')
            update('waiting_for_five_folds', seed=seed)
            time.sleep(30)
        for name, expected in hashes.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        evaluation = OUT / 'swav_evaluation' / f'seed{seed}'
        evaluation.mkdir(parents=True, exist_ok=True)
        for label, command in [
            ('collect', [sys.executable, sources[0], '--family', 'swav', '--seed', str(seed)]),
            ('evaluate', [sys.executable, sources[1], '--run-root', 'domain_ssl/swav_evaluation', '--seed', str(seed)])]:
            update(label, seed=seed)
            with (evaluation / f'{label}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                update('failed', seed=seed, action=label, exit_code=result.returncode)
                raise RuntimeError(f'{label} failed for seed {seed}')
        complete_seeds.append(seed)
        print(f'SwAV seed {seed}: five-fold audit and evaluation complete', flush=True)
    update('complete')


if __name__ == '__main__':
    main()

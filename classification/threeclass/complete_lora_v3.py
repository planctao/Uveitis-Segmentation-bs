"""Audit and report v3 after the current live image experiment terminates."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment import ROOT, dump


def main():
    out = ROOT / 'runs/lora_v3_finalize'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_pid = int((ROOT / 'runs/lora_adaptation_v3/runner.pid').read_text())
        (out / 'runner.pid').write_text(str(os.getpid()))
        dump(out / 'state.json', {'status': 'waiting_for_v3', 'pid': os.getpid(),
            'source_pid': source_pid, 'next_action': 'Audit all completed formal seeds and update the report; no external scoring.'})
        while True:
            path = Path('/proc') / str(source_pid) / 'cmdline'
            if not path.exists() or b'lora_experiments_seeded.py' not in path.read_bytes():
                break
            time.sleep(5)
        terminal = json.loads((ROOT / 'runs/lora_adaptation_v3/state.json').read_text())
        if terminal['status'] != 'completed_development_experiments':
            dump(out / 'state.json', {'status': 'source_not_successful', 'pid': os.getpid(),
                 'source_terminal_state': terminal, 'automatic_restart_attempted': False})
            return
        try:
            dump(out / 'state.json', {'status': 'auditing', 'pid': os.getpid()})
            subprocess.run([sys.executable, '-u', str(ROOT / 'analyze_lora_versions.py'), '--version', '3'], cwd=ROOT, check=True)
            subprocess.run([sys.executable, '-u', str(ROOT / 'report.py')], cwd=ROOT, check=True)
            dump(out / 'state.json', {'status': 'completed', 'pid': os.getpid(),
                'checked_formal_seed_fits': 12, 'conditional_cv_requires_separate_promotion_audit': True,
                'external_test_evaluated': False})
        except BaseException:
            dump(out / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

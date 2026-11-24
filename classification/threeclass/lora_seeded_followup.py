"""Queue the reproducible LoRA study after the currently running v2 study."""
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
    out = ROOT / 'runs/lora_seeded_followup'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / 'runner.pid').write_text(str(os.getpid()))
        assert json.loads((ROOT / 'verification_lora_seed.json').read_text())['passed']
        source_pid = int((ROOT / 'runs/lora_adaptation_v2/runner.pid').read_text())
        dump(out / 'state.json', {'status': 'waiting_for_v2', 'pid': os.getpid(),
             'source_pid': source_pid, 'queued_script': 'lora_experiments_seeded.py',
             'reason': 'Adapter initialization must be reproducible independently of arm order and resume position.'})
        while True:
            command = Path('/proc') / str(source_pid) / 'cmdline'
            if not command.exists() or b'lora_experiments.py' not in command.read_bytes():
                break
            if (ROOT / 'STOP').exists():
                dump(out / 'state.json', {'status': 'stopped_before_launch', 'pid': os.getpid()})
                return
            time.sleep(5)
        if (ROOT / 'STOP').exists():
            dump(out / 'state.json', {'status': 'stopped_before_launch', 'pid': os.getpid()})
            return
        dump(out / 'state.json', {'status': 'running_v3', 'pid': os.getpid(),
             'previous_state': json.loads((ROOT / 'runs/lora_adaptation_v2/state.json').read_text())})
        try:
            with (ROOT / 'lora_experiments_v3.log').open('a') as log:
                subprocess.run([sys.executable, '-u', str(ROOT / 'lora_experiments_seeded.py')],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            dump(out / 'state.json', {'status': 'completed', 'pid': os.getpid()})
        except BaseException:
            dump(out / 'state.json', {'status': 'failed', 'pid': os.getpid(),
                                     'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()

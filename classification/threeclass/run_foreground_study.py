"""Wait for the live extraction, then verify before starting all matched fits."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment import ROOT, dump
from foreground_data import OUT


def main():
    with (OUT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'pipeline.pid').write_text(str(os.getpid()))
        pid = int((OUT / 'extraction.pid').read_text())
        dump(OUT / 'pipeline_state.json', {'status': 'waiting_for_extraction', 'pid': os.getpid(), 'extraction_pid': pid})
        while True:
            path = Path('/proc') / str(pid) / 'cmdline'
            if not path.exists() or b'extract_foreground_multiscale.py' not in path.read_bytes():break
            time.sleep(5)
        state = json.loads((OUT / 'extraction_state.json').read_text())
        if state['status'] != 'completed':
            dump(OUT / 'pipeline_state.json', {'status': 'extraction_not_successful', 'pid': os.getpid(), 'extraction_state': state})
            return
        try:
            for command in ['verify_foreground_features.py', 'foreground_experiments.py']:
                dump(OUT / 'pipeline_state.json', {'status': 'running', 'pid': os.getpid(), 'command': command})
                subprocess.run([sys.executable, '-u', str(ROOT / command)], cwd=ROOT, check=True)
            dump(OUT / 'pipeline_state.json', {'status': 'completed', 'pid': os.getpid(), 'external_test_evaluated': False})
        except BaseException:
            dump(OUT / 'pipeline_state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()

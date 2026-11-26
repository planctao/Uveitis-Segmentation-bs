"""Serial 15-task image SwAV queue; do not launch a second concurrent owner."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/severity_research_20261013/domain_ssl'


def write_state(data):
    path=OUT/'swav_queue_state.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'swav_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=ROOT/'scripts/train_severity_swav.py'
    digest=hashlib.sha256(script.read_bytes()).hexdigest();completed=[];start=time.monotonic()
    probe=json.loads((OUT/'probes/swav_seed42_f1/probe.json').read_text())
    assert probe['successful_updates']>0 and probe['encoder_stem_max_abs_update']>0
    assert probe['sources_sha256']['scripts/train_severity_swav.py']==digest
    for seed in [42,2026,3407]:
        for fold in ['f1','f2','f3','f4','f5']:
            run=OUT/'swav'/f'seed{seed}'/fold;run.mkdir(parents=True,exist_ok=True)
            if (run/'complete.json').exists():
                complete=json.loads((run/'complete.json').read_text())
                assert complete['source_sha256']['scripts/train_severity_swav.py']==digest
                completed.append(dict(seed=seed,fold=fold));continue
            assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
            with (run/'train.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(script),'--seed',str(seed),'--fold',fold],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_state(dict(status='running',owner_pid=os.getpid(),child_pid=child.pid,current=dict(seed=seed,fold=fold),
                                 completed=completed,script_sha256=digest,elapsed_seconds=time.monotonic()-start))
                print('started',seed,fold,'pid',child.pid,flush=True)
                code=child.wait()
            if code:
                write_state(dict(status='failed',owner_pid=os.getpid(),child_pid=child.pid,current=dict(seed=seed,fold=fold),exit_code=code,completed=completed))
                raise RuntimeError(f'SwAV {seed} {fold} exited {code}')
            assert (run/'complete.json').exists()
            completed.append(dict(seed=seed,fold=fold));print('completed',seed,fold,flush=True)
    write_state(dict(status='complete',owner_pid=os.getpid(),completed=completed,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()

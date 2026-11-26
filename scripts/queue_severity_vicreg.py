"""Serial fixed 15-task VICReg queue, with actual resource admission checks."""
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


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_state(data):
    path=OUT/'vicreg_queue_state.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def free_gpu_memory():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)
    return float(result.stdout.splitlines()[0])


def main():
    lock=(OUT/'vicreg_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=ROOT/'scripts/train_severity_vicreg.py'
    probe=json.loads((OUT/'probes/vicreg_seed42_f1/probe.json').read_text())
    sources=probe['sources_sha256']
    for name,expected in sources.items():assert digest(ROOT/name)==expected,name
    assert probe['successful_updates']>0 and probe['encoder_stem_max_abs_update']>0
    required=probe['peak_reserved_mb']+1024
    completed=[];evaluated=[];start=time.monotonic()
    for seed in [42,2026,3407]:
        for fold in ['f1','f2','f3','f4','f5']:
            run=OUT/'vicreg'/f'seed{seed}'/fold;run.mkdir(parents=True,exist_ok=True)
            if (run/'complete.json').exists():
                complete=json.loads((run/'complete.json').read_text());assert complete['source_sha256']==sources
                completed.append(dict(seed=seed,fold=fold));continue
            for name,expected in sources.items():assert digest(ROOT/name)==expected,name
            while free_gpu_memory()<required:
                write_state(dict(status='waiting_for_memory',owner_pid=os.getpid(),required_free_mb=required,
                    current=dict(seed=seed,fold=fold),completed=completed,evaluated_seeds=evaluated))
                time.sleep(30)
            with (run/'train.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(script),'--seed',str(seed),'--fold',fold],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_state(dict(status='training',owner_pid=os.getpid(),child_pid=child.pid,current=dict(seed=seed,fold=fold),
                    completed=completed,evaluated_seeds=evaluated,source_sha256=sources,elapsed_seconds=time.monotonic()-start))
                print('started VICReg',seed,fold,child.pid,flush=True);code=child.wait()
            if code:
                write_state(dict(status='failed',owner_pid=os.getpid(),current=dict(seed=seed,fold=fold),exit_code=code,completed=completed))
                raise RuntimeError(f'VICReg {seed} {fold} failed')
            assert (run/'complete.json').exists();completed.append(dict(seed=seed,fold=fold))
        evaluation=OUT/'vicreg_evaluation'/f'seed{seed}';evaluation.mkdir(parents=True,exist_ok=True)
        for action,command in [
            ('collect',[sys.executable,'scripts/collect_severity_vicreg.py','--seed',str(seed)]),
            ('evaluate',[sys.executable,'scripts/evaluate_severity_baselines.py','--run-root','domain_ssl/vicreg_evaluation','--seed',str(seed)])]:
            write_state(dict(status=action,owner_pid=os.getpid(),seed=seed,completed=completed,evaluated_seeds=evaluated))
            with (evaluation/f'{action}.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_state(dict(status='failed',owner_pid=os.getpid(),seed=seed,action=action,exit_code=result.returncode,completed=completed))
                raise RuntimeError(f'VICReg {action} {seed} failed')
        evaluated.append(seed);print('VICReg seed fully audited and evaluated',seed,flush=True)
    write_state(dict(status='complete',owner_pid=os.getpid(),completed=completed,evaluated_seeds=evaluated,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()

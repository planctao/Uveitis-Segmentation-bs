"""Observe the same live training process, then verify all predeclared frozen sources."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def verify_frozen_sources():
    source=ROOT/'runs/nonlinear_backbone_adapter'
    protocol=json.loads((source/'protocol.json').read_text())
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(ROOT/protocol['reference'])==protocol['reference_sha256']
    assert sha(ROOT/protocol['preflight_verification'])==protocol['preflight_verification_sha256']
    checked=[]
    for path in sorted((source/'split_plans').glob('*.json')):
        plan=json.loads(path.read_text())
        for checkpoint,expected in zip(plan['source_checkpoints'],plan['source_checkpoint_sha256']):
            assert sha(checkpoint)==expected
            checked.append({'checkpoint':checkpoint,'sha256':expected})
    assert len(checked)==12
    return checked


def main():
    out=ROOT/'runs/nonlinear_adapter_finalize';out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=int((ROOT/'runs/nonlinear_backbone_adapter/runner.pid').read_text())
        checked=verify_frozen_sources();(out/'runner.pid').write_text(str(os.getpid()))
        dump(out/'state.json',{'status':'waiting_for_training','pid':os.getpid(),'source_pid':pid,'initial_frozen_sources_verified':True})
        while True:
            cmd=Path('/proc')/str(pid)/'cmdline'
            if not cmd.exists() or b'nonlinear_adapter_experiments.py' not in cmd.read_bytes():break
            time.sleep(5)
        state=json.loads((ROOT/'runs/nonlinear_backbone_adapter/state.json').read_text())
        if state['status']!='completed':
            dump(out/'state.json',{'status':'source_not_successful','pid':os.getpid(),'source_terminal_state':state,'automatic_restart_attempted':False})
            return
        try:
            assert checked==verify_frozen_sources()
            target=ROOT/'results/research_nonlinear_backbone_adapter'
            verification=json.loads((target/'verification.json').read_text());assert verification['passed']
            comparison=json.loads((target/'comparison.json').read_text())
            assert verification['checked_seed_fits']==comparison['new_seed_fits']==state['new_seed_fits']
            for fit in verification['fits']:
                folder=(ROOT/'runs/nonlinear_backbone_adapter'/fit['candidate_id'] if fit['phase']=='validation' else ROOT/'runs/nonlinear_backbone_adapter/crossvalidation'/fit['candidate_id']/fit['phase'])/f'seed_{fit["seed"]}'
                assert sha(folder/'model.pt')==fit['checkpoint_sha256']
            dump(target/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
                'source_checkpoints':checked,'checked_actual_models':state['new_seed_fits'],
                'all_protocol_and_code_hashes_unchanged':True,'external_test_evaluated':False})
            subprocess.run([sys.executable,str(ROOT/'refresh_recent_progress.py')],cwd=ROOT,check=True)
            dump(out/'state.json',{'status':'completed','pid':os.getpid(),'checked_actual_models':state['new_seed_fits'],'external_test_evaluated':False})
        except BaseException:
            dump(out/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

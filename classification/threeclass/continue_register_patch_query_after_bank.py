"""Gated continuation: completed bank -> random256 model proof -> frozen study."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment import ROOT,dump
from kernel_residual_experiments import sha

OUT=ROOT/'runs/register_patch_query_continuation'


def verify(protocol):
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in protocol['frozen_artifacts'].items():assert sha(name)==expected,name


def main_run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    while True:
        state=json.loads((ROOT/'runs/register_patch_query_bank/state.json').read_text())
        if state['status']=='completed':break
        assert state['status']!='failed',state
        command_path=Path('/proc')/str(state.get('pid',protocol['bank_pid']))/'cmdline'
        command=command_path.read_bytes().replace(b'\0',b' ').decode() if command_path.exists() else ''
        assert 'extract_register_patch_query_bank.py' in command,'Bank extraction is not live and not completed; do not restart automatically.'
        dump(OUT/'state.json',{'status':'waiting_verified_live_bank','pid':os.getpid(),'bank_pid':state.get('pid'),'completed_images':state.get('completed_images',0),'classification_fits':0,'classification_scores':0,'external_test_evaluated':False})
        time.sleep(10)
    verify(protocol)
    dump(OUT/'state.json',{'status':'random256_model_preflight','pid':os.getpid(),'classification_fits':0,'classification_scores':0,'external_test_evaluated':False})
    subprocess.run([sys.executable,'-u',str(ROOT/'verify_register_patch_query_model.py')],cwd=ROOT,check=True)
    verify(protocol)
    proof=json.loads((ROOT/'verification_register_patch_query_model.json').read_text());assert proof['passed'] and len(proof['cases'])==4
    dump(OUT/'state.json',{'status':'freezing_formal_study','pid':os.getpid(),'classification_fits':0,'classification_scores':0,'external_test_evaluated':False})
    subprocess.run([sys.executable,'-u',str(ROOT/'start_register_patch_query.py')],cwd=ROOT,check=True)
    assert (ROOT/'runs/register_patch_query/protocol.json').exists()
    dump(OUT/'state.json',{'status':'completed_launched_formal_study','pid':os.getpid(),'study_protocol':'runs/register_patch_query/protocol.json','study_protocol_sha256':sha(ROOT/'runs/register_patch_query/protocol.json'),'external_test_evaluated':False})


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main_run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()

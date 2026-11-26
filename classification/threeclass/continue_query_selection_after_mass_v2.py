"""Versioned observer recovery for the already-launched worker; never relaunch that study."""
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

OUT=ROOT/'runs/query_selection_continuation_v2'


def status(value,**details):dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})


def verify(protocol):
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def live(pid,script):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and str(ROOT/script).encode() in p.read_bytes().split(b'\0')


def resolve_worker_pid(state,startup_pid,is_live):
    assert state['status']!='failed',state
    pid=state.get('pid')
    if pid is None:
        assert state['status']=='frozen_before_training',('Missing PID outside declared startup state',state)
        pid=startup_pid
    assert isinstance(pid,int) and pid>0
    assert pid==startup_pid,('Worker identity changed; never restart automatically',pid,startup_pid)
    assert is_live(pid),('Launched worker no longer live; never restart automatically',pid,state)
    return pid


def wait(family,script,terminal,startup_pid):
    while True:
        state=json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state['status']!='failed',state
        if state['status'] in terminal:return state
        pid=resolve_worker_pid(state,startup_pid,lambda value:live(value,script))
        status('waiting_'+family,worker_pid=pid,worker_status=state['status'],
            arm=state.get('arm'),phase=state.get('phase'),seed=state.get('seed'),startup_state_has_pid='pid' in state)
        time.sleep(10)


def step(script,*args):
    status('running_predeclared_step',script=script)
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    old=json.loads((ROOT/'runs/query_selection_continuation/state.json').read_text())
    assert old['status']=='failed' and "KeyError: 'pid'" in old['error']
    predecessor=json.loads((ROOT/'runs/register_patch_mass_calibration_continuation/state.json').read_text())
    assert predecessor['status']=='completed_mass_calibration_decision_requires_next_exploration'
    saved_review=json.loads((ROOT/'results/research_register_patch_mass_calibration/independent_saved_output_review.json').read_text())
    assert saved_review['review_passed'] and saved_review['fit_only_mass_buffers_and_actual_first_input_mass_checked']==48
    assert json.loads((ROOT/'verification_register_patch_query_selection_model.json').read_text())['passed']
    assert json.loads((ROOT/'verification_register_patch_query_selection_confirmation_plan.json').read_text())['passed']
    # Original launcher already created this worker. Never invoke it again.
    state=wait('register_patch_query_selection','register_patch_query_selection_experiments.py',{'completed'},protocol['existing_worker_pid'])
    result=ROOT/'results/research_register_patch_query_selection'
    audit=json.loads((result/'verification.json').read_text());comparison=json.loads((result/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==48
    assert audit['frozen_register_source_heads_checked']==12
    assert audit['reused_fusion_source_heads_checked']==len(audit['source_heads'])==24
    assert len(audit['fixed_fusion_phases_checked'])==16
    assert audit['query_specific_and_shared_frozen_cache_inputs_verified']
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    verify(protocol);step('refresh_recent_progress.py')
    if eligible:
        selected=json.loads((result/'selection_before_repeat.json').read_text())['candidate']
        assert selected==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        query=selected['register_patch_query_selection_configuration']['selection']=='query_top16'
        step('verify_register_patch_query_selection_confirmation.py')
        verify(protocol);step('start_register_patch_query_selection_confirmation.py')
        state=wait('register_patch_query_selection_confirmation','register_patch_query_selection_confirmation.py',{'completed'},json.loads((ROOT/'results/next_register_patch_query_selection_hypothesis.json').read_text())['confirmation_pid'])
        result=ROOT/'results/research_register_patch_query_selection_confirmation'
        audit=json.loads((result/'verification.json').read_text())
        assert audit['passed'] and audit['actual_seed_heads_checked']==(126 if query else 108)
        assert audit['actual_training_witnesses_checked']==(90 if query else 72) and audit['generative_fits_checked']==18
        assert audit['matched_controls_required']==['matched_shared64']+(['matched_shared16'] if query else [])
        verify(protocol);step('refresh_recent_progress.py')
        if state['confirmation_gate_passed']:
            status('confirmation_passed_requires_reference_review',reference_changed=False);return
    status('completed_query_selection_decision_requires_next_exploration',reference_changed=False)



def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()

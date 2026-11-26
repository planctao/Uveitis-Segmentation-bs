"""Frozen cache-proof/predecessor-decision/retrieval-study continuation; no automatic restarts."""
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

OUT=ROOT/'runs/query_selection_continuation'


def status(value,**details):dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})


def verify(protocol):
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def live(pid,script):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and str(ROOT/script).encode() in p.read_bytes().split(b'\0')


def wait(family,script,terminal):
    while True:
        state=json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state['status']!='failed',state
        if state['status'] in terminal:return state
        assert live(state['pid'],script),('Do not restart automatically',family,state)
        status('waiting_'+family,worker_pid=state['pid'],worker_status=state['status'],
            arm=state.get('arm'),phase=state.get('phase'),seed=state.get('seed'),completed_images=state.get('completed_images'))
        time.sleep(10)


def step(script,*args):
    status('running_predeclared_step',script=script)
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)


def proof_step(script,output,**expected):
    path=ROOT/output
    if not path.exists():step(script)
    proof=json.loads(path.read_text());assert proof.get('passed',proof.get('review_passed'))
    for key,value in expected.items():assert proof[key]==value,(output,key)
    return proof


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    wait('register_query_selection_bank','extract_register_query_selection_bank.py',{'completed'})
    proof_step('review_register_query_selection_bank.py','results/register_query_selection_bank_root_review.json',chunks_checked=262,development_images=2091)
    verify(protocol)
    model=proof_step('verify_register_patch_query_selection_model.py','verification_register_patch_query_selection_model.json',classification_fits=0,classification_scores=0,external_images_read=0)
    assert len(model['cases'])==4 and all(r['actual_images']==256 for r in model['cases'])
    planning=proof_step('verify_register_patch_query_selection_confirmation_plan.py','verification_register_patch_query_selection_confirmation_plan.json',classification_fits=0,classification_scores=0,external_images_read=0)
    assert len(planning['plans'])==6 and len(planning['synthetic_formula_cases'])==4
    assert len(planning['actual_matched_control_engine_cases'])==4
    assert planning['query_winner_supervised_fits']==126 and planning['query_winner_training_witnesses']==90
    assert planning['shared16_winner_supervised_fits']==108 and planning['shared16_winner_training_witnesses']==72
    verify(protocol)
    parent=wait('register_patch_mass_calibration_continuation','continue_register_patch_mass_calibration.py',
        {'completed_mass_calibration_decision_requires_next_exploration','confirmation_passed_requires_reference_review'})
    if parent['status']=='confirmation_passed_requires_reference_review':
        status('predecessor_confirmation_passed_requires_reference_review',reference_changed=False);return
    proof_step('review_register_patch_mass_calibration_outputs.py','results/research_register_patch_mass_calibration/independent_saved_output_review.json',
        classification_fits=48,actual_task_witnesses=48,fit_only_mass_buffers_and_actual_first_input_mass_checked=48)
    verify(protocol)
    step('start_register_patch_query_selection.py')
    state=wait('register_patch_query_selection','register_patch_query_selection_experiments.py',{'completed'})
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
        state=wait('register_patch_query_selection_confirmation','register_patch_query_selection_confirmation.py',{'completed'})
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

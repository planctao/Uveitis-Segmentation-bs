"""Observe one root-launched direct patch-evidence worker; never relaunch original training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from continue_query_selection_after_mass_v2 import resolve_worker_pid

OUT=ROOT/'runs/class_patch_evidence_continuation'


def status(value,**details):
    dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})


def verify(protocol):
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def live(pid,script):
    path=Path('/proc')/str(pid)/'cmdline'
    return path.exists() and str(ROOT/script).encode() in path.read_bytes().split(b'\0')


def wait(family,script,startup_pid):
    while True:
        state=json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state['status']!='failed',state
        if state['status']=='completed':return state
        pid=resolve_worker_pid(state,startup_pid,lambda value:live(value,script))
        status('waiting_'+family,worker_pid=pid,worker_status=state['status'],
            arm=state.get('arm'),phase=state.get('phase'),seed=state.get('seed'))
        time.sleep(10)


def step(script):
    status('running_predeclared_step',script=script)
    subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    assert json.loads((ROOT/'verification_class_query_patch_evidence_startup.json').read_text())['passed']
    assert json.loads((ROOT/'verification_class_query_patch_evidence_confirmation_plan.json').read_text())['passed']
    state=wait('class_query_patch_evidence','class_query_patch_evidence_experiments.py',protocol['existing_worker_pid'])
    result=ROOT/'results/research_class_query_patch_evidence'
    audit=json.loads((result/'verification.json').read_text());comparison=json.loads((result/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==48
    assert audit['frozen_register_source_heads_checked']==12
    assert audit['reused_fusion_source_heads_checked']==len(audit['source_heads'])==24
    assert len(audit['fixed_fusion_phases_checked'])==16
    assert audit['only_direct_patch_class_queries_readouts_and_bias_trained']
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    verify(protocol)
    saved=result/'independent_saved_output_review.json'
    if not saved.exists():step('review_class_query_patch_evidence_outputs.py')
    review=json.loads(saved.read_text())
    assert review['review_passed'] and review['original_and_both_matched_control_gates_independently_checked']==8
    assert review['gate_passers']==len(eligible)
    step('refresh_recent_progress.py')
    if eligible:
        selected=json.loads((result/'selection_before_repeat.json').read_text())['candidate']
        assert selected==max(eligible,key=lambda row:row['cv']['mean_macro_f1'])
        coupled=selected['class_query_patch_evidence_configuration']['learned_query']
        verify(protocol);step('verify_class_query_patch_evidence_confirmation.py')
        verify(protocol);step('start_class_query_patch_evidence_confirmation.py')
        pid=json.loads((ROOT/'results/next_class_query_patch_evidence_hypothesis.json').read_text())['confirmation_pid']
        state=wait('class_query_patch_evidence_confirmation','class_query_patch_evidence_confirmation.py',pid)
        result=ROOT/'results/research_class_query_patch_evidence_confirmation'
        audit=json.loads((result/'verification.json').read_text())
        assert audit['passed'] and audit['actual_seed_heads_checked']==(126 if coupled else 90)
        assert audit['actual_training_witnesses_checked']==(90 if coupled else 54) and audit['generative_fits_checked']==18
        assert audit['matched_controls_required']==(['matched_uniform','matched_source_query'] if coupled else [])
        verify(protocol);step('refresh_recent_progress.py')
        if state['confirmation_gate_passed']:
            step('report.py')
            status('confirmation_passed_requires_reference_review',reference_changed=False);return
    status('completed_class_patch_evidence_decision_requires_next_exploration',reference_changed=False)
    step('report.py')
    # Preserve the terminal state after report.py's transient step status.
    status('completed_class_patch_evidence_decision_requires_next_exploration',reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()

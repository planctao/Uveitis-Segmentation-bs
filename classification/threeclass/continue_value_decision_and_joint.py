"""Review the completed value study, conditionally confirm it, then fit joint Q/V.

Waits for the existing single-path continuation, never relaunches its work,
and stops for review after any positive confirmation or failed check.
"""
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

OUT = ROOT/'runs/value_decision_and_joint'


def verify(protocol):
    for section in ['code_sha256', 'frozen_artifacts']:
        for name, expected in protocol[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def live(pid, script):
    path = Path('/proc')/str(pid)/'cmdline'
    return path.exists() and str(ROOT/script).encode() in path.read_bytes().split(b'\0')


def status(value, **details):
    dump(OUT/'state.json', {'status': value, 'pid': os.getpid(), **details, 'external_test_evaluated': False})


def wait_existing_loop():
    while True:
        state = json.loads((ROOT/'runs/patch_architecture_loop/state.json').read_text())
        assert state['status'] != 'failed', state
        if state['status'] in ['query_confirmation_passed_requires_reference_review', 'completed_value_study_requires_result_review']:
            return state
        assert live(state['pid'], 'continue_patch_architecture_loop.py'), 'Existing single-path continuation is not live or terminal; never restart automatically.'
        status('waiting_existing_single_path_loop', predecessor_pid=state['pid'], predecessor_status=state['status'])
        time.sleep(10)


def wait_worker(family, script, starting_pid):
    while True:
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state.get('status') != 'failed', state
        if state['status'] == 'completed': return state
        pid = state.get('pid', starting_pid)
        assert live(pid, script), f'{family} is not live or terminal; never restart automatically.'
        status('waiting_'+family, worker_pid=pid, worker_status=state['status'], arm=state.get('arm'),
            phase=state.get('phase'), repeat=state.get('repeat'), fold=state.get('fold'), seed=state.get('seed'))
        time.sleep(10)


def run_step(script):
    status('running_predeclared_step', script=script)
    subprocess.run([sys.executable, '-u', str(ROOT/script)], cwd=ROOT, check=True)


def main_run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    predecessor = wait_existing_loop()
    verify(protocol)
    if predecessor['status'] == 'query_confirmation_passed_requires_reference_review':
        status('query_confirmation_passed_requires_reference_review', reference_changed=False)
        return
    out = ROOT/'runs/register_patch_value'
    state = json.loads((out/'state.json').read_text())
    assert state['status'] == 'completed'
    verification = json.loads((ROOT/'results/research_register_patch_value/verification.json').read_text())
    comparison = json.loads((ROOT/'results/research_register_patch_value/comparison.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['frozen_register_source_heads_checked'] == 12
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert len(eligible) == state['passing_development_gates']
    if eligible:
        assert not (ROOT/'runs/register_patch_value_confirmation/protocol.json').exists(), 'An existing confirmation requires review; do not launch a duplicate.'
        run_step('verify_register_patch_value_confirmation_v2.py')
        verify(protocol)
        run_step('start_register_patch_value_confirmation_v2.py')
        hypothesis = json.loads((ROOT/'results/next_register_patch_value_hypothesis.json').read_text())
        confirmed = wait_worker('register_patch_value_confirmation', 'register_patch_value_confirmation.py', hypothesis['confirmation_pid'])
        proof = json.loads((ROOT/'results/research_register_patch_value_confirmation/verification.json').read_text())
        assert proof['passed'] and proof['actual_seed_heads_checked'] == 90 and proof['actual_training_witnesses_checked'] == 54
        if confirmed['confirmation_gate_passed']:
            verify(protocol)
            status('value_confirmation_passed_requires_reference_review', reference_changed=False)
            return
    verify(protocol)
    assert not (ROOT/'runs/register_patch_joint/protocol.json').exists(), 'An existing joint study requires review; never launch a duplicate.'
    run_step('start_register_patch_joint.py')
    hypothesis = json.loads((ROOT/'results/next_register_patch_joint_hypothesis.json').read_text())
    joint = wait_worker('register_patch_joint', 'register_patch_joint_experiments.py', hypothesis['pid'])
    proof = json.loads((ROOT/'results/research_register_patch_joint/verification.json').read_text())
    assert proof['passed'] and proof['actual_seed_heads_checked'] == proof['actual_training_witnesses_checked'] == 48
    verify(protocol)
    run_step('refresh_recent_progress.py')
    status('completed_joint_study_requires_result_review', joint_passing_development_gates=joint['passing_development_gates'], reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main_run()
        except BaseException:
            status('failed', error=traceback.format_exc())
            raise


if __name__ == '__main__': main()

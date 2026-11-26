"""Finish query research and, when appropriate, launch the preregistered value study.

Never restart an existing fit, score an external cohort, or promote a reference.
Every failed process/check stops this continuation for review.
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

OUT = ROOT/'runs/patch_architecture_loop'


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


def wait_query():
    while True:
        path = ROOT/'runs/register_patch_query/state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('status') == 'completed': return state
        assert state.get('status') != 'failed', state
        pid = state.get('pid')
        if pid is None:
            hypothesis = json.loads((ROOT/'results/next_register_patch_query_hypothesis.json').read_text())
            pid = hypothesis.get('pid')
        runner_live = live(pid, 'register_patch_query_experiments.py') if pid is not None else False
        if not runner_live:
            parent = json.loads((ROOT/'runs/register_patch_query_continuation/state.json').read_text())
            assert parent['status'] != 'failed', parent
            assert live(parent['pid'], 'continue_register_patch_query_after_bank.py'), 'No actual query runner or authorized bank continuation; no automatic restart.'
        status('waiting_existing_query_study', query_status=state.get('status', 'not_yet_started'),
            query_pid=pid, arm=state.get('arm'), phase=state.get('phase'), seed=state.get('seed'))
        time.sleep(10)


def wait_runner(family, script):
    while True:
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        if state.get('status') == 'completed': return state
        assert state.get('status') != 'failed', state
        pid = state.get('pid')
        if pid is None:
            hypothesis_name = 'register_patch_query' if family.endswith('_confirmation') else family
            hypothesis = json.loads((ROOT/'results'/('next_'+hypothesis_name+'_hypothesis.json')).read_text())
            pid = hypothesis.get('confirmation_pid' if family.endswith('_confirmation') else 'pid')
        assert pid is not None and live(pid, script), f'{family} is not actually live and not completed; no automatic restart.'
        status('waiting_'+family, worker_pid=pid, worker_status=state['status'], arm=state.get('arm'),
            phase=state.get('phase'), repeat=state.get('repeat'), fold=state.get('fold'), seed=state.get('seed'))
        time.sleep(10)


def run_step(script):
    status('running_predeclared_step', script=script)
    subprocess.run([sys.executable, '-u', str(ROOT/script)], cwd=ROOT, check=True)


def main_run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    query_state = wait_query()
    verification = json.loads((ROOT/'results/research_register_patch_query/verification.json').read_text())
    comparison = json.loads((ROOT/'results/research_register_patch_query/comparison.json').read_text())
    assert verification['passed']
    assert verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24 and len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert len(eligible) == query_state['passing_development_gates']
    verify(protocol)
    if eligible:
        assert not (ROOT/'runs/register_patch_query_confirmation/protocol.json').exists(), 'An existing confirmation requires review; never launch a duplicate.'
        run_step('verify_register_patch_query_confirmation.py')
        verify(protocol)
        run_step('start_register_patch_query_confirmation.py')
        confirmation = wait_runner('register_patch_query_confirmation', 'register_patch_query_confirmation.py')
        verification = json.loads((ROOT/'results/research_register_patch_query_confirmation/verification.json').read_text())
        assert verification['passed'] and verification['actual_seed_heads_checked'] == 90 and verification['actual_training_witnesses_checked'] == 54
        if confirmation['confirmation_gate_passed']:
            verify(protocol)
            status('query_confirmation_passed_requires_reference_review', reference_changed=False)
            return
    verify(protocol)
    assert not (ROOT/'runs/register_patch_value/protocol.json').exists(), 'An existing value study requires review; never launch a duplicate.'
    run_step('verify_register_patch_value_model.py')
    verify(protocol)
    run_step('start_register_patch_value.py')
    value_state = wait_runner('register_patch_value', 'register_patch_value_experiments.py')
    value_verification = json.loads((ROOT/'results/research_register_patch_value/verification.json').read_text())
    assert value_verification['passed'] and value_verification['actual_seed_heads_checked'] == value_verification['actual_training_witnesses_checked'] == 48
    verify(protocol)
    run_step('refresh_recent_progress.py')
    status('completed_value_study_requires_result_review', value_passing_development_gates=value_state['passing_development_gates'], reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main_run()
        except BaseException:
            status('failed', error=traceback.format_exc())
            raise


if __name__ == '__main__': main()

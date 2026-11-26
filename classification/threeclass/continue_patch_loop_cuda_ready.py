"""Versioned continuation after the preserved CUDA restoration audit failure."""
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

OUT = ROOT/'runs/patch_loop_cuda_ready'


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


def run_step(script):
    status('running_predeclared_step', script=script)
    subprocess.run([sys.executable, '-u', str(ROOT/script)], cwd=ROOT, check=True)


def wait_worker(family, script, initial_pid, state_path=None):
    path = state_path or ROOT/'runs'/family/'state.json'
    while True:
        state = json.loads(path.read_text())
        assert state.get('status') != 'failed', state
        if state['status'] == 'completed': return state
        pid = state.get('pid', initial_pid)
        assert live(pid, script), f'{family} is not actually live or terminal; never restart automatically.'
        status('waiting_'+family, worker_pid=pid, worker_status=state['status'], arm=state.get('arm'),
            phase=state.get('phase'), repeat=state.get('repeat'), fold=state.get('fold'), seed=state.get('seed'))
        time.sleep(10)


def audit_record(family):
    state = json.loads((ROOT/'runs'/family/'state.json').read_text())
    assert state['status'] == 'completed'
    verification = json.loads((ROOT/'results'/('research_'+family)/'verification.json').read_text())
    comparison = json.loads((ROOT/'results'/('research_'+family)/'comparison.json').read_text())
    assert verification['passed']
    assert verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == len(verification['fits']) == 48
    assert verification['frozen_register_source_heads_checked'] == 12
    assert verification['reused_fusion_source_heads_checked'] == len(verification['source_heads']) == 24
    assert len(verification['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert len(eligible) == state['passing_development_gates']
    return eligible


def main_run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    recovery_path = ROOT/'runs/register_patch_query/cuda_ready_audit_recovery/state.json'
    wait_worker('register_patch_query_audit_recovery', 'recover_register_patch_query_cuda_ready.py', protocol['query_recovery_pid'], recovery_path)
    stages = [
        ('register_patch_query', None, None, None,
            'verify_register_patch_query_confirmation.py', 'start_patch_query_confirmation_cuda_ready.py', 'run_patch_query_confirmation_cuda_ready.py'),
        ('register_patch_value', 'verify_register_patch_value_model.py', 'start_patch_value_cuda_ready.py', 'run_patch_value_cuda_ready.py',
            'verify_register_patch_value_confirmation_v2.py', 'start_patch_value_confirmation_cuda_ready.py', 'run_patch_value_confirmation_cuda_ready.py'),
        ('register_patch_joint', None, 'start_patch_joint_cuda_ready.py', 'run_patch_joint_cuda_ready.py',
            'verify_register_patch_joint_confirmation.py', 'start_patch_joint_confirmation_cuda_ready.py', 'run_patch_joint_confirmation_cuda_ready.py')]
    for family, preflight, starter, script, formula_check, confirm_starter, confirm_script in stages:
        verify(protocol)
        if starter is not None:
            assert not (ROOT/'runs'/family/'protocol.json').exists(), f'{family} already exists; do not launch a duplicate.'
            if preflight is not None:
                run_step(preflight)
                verify(protocol)
            run_step(starter)
            hypothesis = json.loads((ROOT/'results'/('next_'+family+'_hypothesis.json')).read_text())
            wait_worker(family, script, hypothesis['pid'])
        eligible = audit_record(family)
        verify(protocol)
        run_step('refresh_recent_progress.py')
        if eligible:
            confirmation = family+'_confirmation'
            assert not (ROOT/'runs'/confirmation/'protocol.json').exists(), f'{confirmation} already exists; do not launch a duplicate.'
            run_step(formula_check)
            verify(protocol)
            run_step(confirm_starter)
            hypothesis = json.loads((ROOT/'results'/('next_'+family+'_hypothesis.json')).read_text())
            confirmed = wait_worker(confirmation, confirm_script, hypothesis['confirmation_pid'])
            proof = json.loads((ROOT/'results'/('research_'+confirmation)/'verification.json').read_text())
            assert proof['passed'] and proof['actual_seed_heads_checked'] == 90 and proof['actual_training_witnesses_checked'] == 54
            assert proof['generative_fits_checked'] == 18
            verify(protocol)
            if confirmed['confirmation_gate_passed']:
                run_step('refresh_recent_progress.py')
                status('confirmation_passed_requires_reference_review', family=family, reference_changed=False)
                return
    verify(protocol)
    run_step('refresh_recent_progress.py')
    status('completed_three_architecture_decisions_requires_next_exploration', reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main_run()
        except BaseException:
            status('failed', error=traceback.format_exc())
            raise


if __name__ == '__main__': main()

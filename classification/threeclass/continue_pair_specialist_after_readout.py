"""Continue the controlled readout/local-content study after the gated-value decision."""
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

OUT = ROOT/'runs/pair_specialist_continuation'


def status(value, **details):
    dump(OUT/'state.json', {'status': value, 'pid': os.getpid(), **details, 'external_test_evaluated': False})


def live(pid, script):
    path = Path('/proc')/str(pid)/'cmdline'
    return path.exists() and str(ROOT/script).encode() in path.read_bytes().split(b'\0')


def verify(protocol):
    for section in ['code_sha256', 'frozen_artifacts']:
        for name, expected in protocol[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def run_step(script):
    status('running_predeclared_step', script=script)
    subprocess.run([sys.executable, '-u', str(ROOT/script)], cwd=ROOT, check=True)


def wait_family(family, script, initial_pid):
    while True:
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state['status'] != 'failed', state
        if state['status'] == 'completed': return state
        pid = state.get('pid', initial_pid)
        assert live(pid, script), (family, state, 'Do not restart automatically.')
        status('waiting_'+family, worker_pid=pid, worker_status=state['status'],
            arm=state.get('arm'), phase=state.get('phase'), seed=state.get('seed'),
            repeat=state.get('repeat'), fold=state.get('fold'))
        time.sleep(10)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    while True:
        parent = json.loads((ROOT/'runs/readout_adaptation_continuation/state.json').read_text())
        assert parent['status'] != 'failed', parent
        if parent['status'] in ['confirmation_passed_requires_reference_review','predecessor_confirmation_passed_requires_reference_review']:
            status('predecessor_confirmation_passed_requires_reference_review', family=parent.get('family','predecessor'))
            return
        if parent['status'] == 'completed_readout_adaptation_decision_requires_next_exploration': break
        assert live(parent.get('pid', protocol['predecessor_pid']), 'continue_readout_adaptation_after_gated.py'), parent
        status('waiting_readout_adaptation_continuation', worker_pid=parent.get('pid'), worker_status=parent['status'])
        time.sleep(10)
    verify(protocol)
    family = 'inflammatory_pair_specialist'
    assert not (ROOT/'runs'/family/'protocol.json').exists()
    run_step('start_inflammatory_pair_specialist.py')
    draft = json.loads((ROOT/'results/next_inflammatory_pair_specialist_hypothesis.json').read_text())
    state = wait_family(family, 'inflammatory_pair_specialist_experiments.py', draft['pid'])
    source = ROOT/'results/research_inflammatory_pair_specialist'
    audit = json.loads((source/'verification.json').read_text())
    comparison = json.loads((source/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked'] == audit['actual_training_witnesses_checked'] == len(audit['fits']) == 48
    assert audit['frozen_register_source_heads_checked'] == 12
    assert audit['reused_fusion_source_heads_checked'] == len(audit['source_heads']) == 24
    assert len(audit['fixed_fusion_phases_checked']) == 16
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert len(eligible) == state['passing_development_gates']
    verify(protocol)
    run_step('refresh_recent_progress.py')
    if eligible:
        run_step('verify_inflammatory_pair_specialist_confirmation.py')
        verify(protocol)
        run_step('start_inflammatory_pair_specialist_confirmation.py')
        draft = json.loads((ROOT/'results/next_inflammatory_pair_specialist_hypothesis.json').read_text())
        state = wait_family(family+'_confirmation', 'inflammatory_pair_specialist_confirmation.py', draft['confirmation_pid'])
        proof = json.loads((ROOT/'results/research_inflammatory_pair_specialist_confirmation/verification.json').read_text())
        assert proof['passed'] and proof['actual_seed_heads_checked']==90 and proof['actual_training_witnesses_checked']==54
        assert proof['generative_fits_checked'] == 18
        verify(protocol)
        run_step('refresh_recent_progress.py')
        if state['confirmation_gate_passed']:
            status('confirmation_passed_requires_reference_review', family=family, reference_changed=False)
            return
    status('completed_pair_specialist_decision_requires_next_exploration', reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            status('failed', error=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()

"""Observe the single mass-calibration worker; confirm only its predeclared CV winner."""
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

OUT = ROOT/'runs/register_patch_mass_calibration_continuation'
FAMILY = 'register_patch_mass_calibration'


def status(value, **details):
    dump(OUT/'state.json', {'status': value, 'pid': os.getpid(), **details, 'external_test_evaluated': False})


def verify(protocol):
    for group in ['code_sha256','frozen_artifacts']:
        for name, expected in protocol[group].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path) == expected, name


def wait_for(family, script):
    while True:
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        assert state['status'] != 'failed', state
        if state['status'] == 'completed': return state
        pid = state['pid']
        path = Path('/proc')/str(pid)/'cmdline'
        assert path.exists() and str(ROOT/script).encode() in path.read_bytes().split(b'\0'), ('Do not restart automatically',state)
        status('waiting_'+family, worker_pid=pid, worker_status=state['status'],
            arm=state.get('arm'),phase=state.get('phase'),seed=state.get('seed'),repeat=state.get('repeat'),fold=state.get('fold'))
        time.sleep(10)


def step(script):
    status('running_predeclared_step', script=script)
    subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify(protocol)
    state = wait_for(FAMILY,FAMILY+'_experiments.py')
    result = ROOT/'results'/('research_'+FAMILY)
    audit = json.loads((result/'verification.json').read_text())
    comparison = json.loads((result/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked'] == audit['actual_training_witnesses_checked'] == len(audit['fits']) == 48
    assert audit['frozen_register_source_heads_checked'] == 12
    assert audit['reused_fusion_source_heads_checked'] == len(audit['source_heads']) == 24
    assert len(audit['fixed_fusion_phases_checked']) == 16 and audit['fit_only_mass_statistics_verified_for_all_models']
    eligible = [r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible) == state['passing_development_gates']
    verify(protocol); step('refresh_recent_progress.py')
    if eligible:
        selected = json.loads((result/'selection_before_repeat.json').read_text())['candidate']
        assert selected == max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        source_mass = selected[FAMILY+'_configuration']['mass_mode'] == 'source_mass'
        step('verify_register_patch_mass_calibration_confirmation.py')
        verify(protocol); step('start_register_patch_mass_calibration_confirmation.py')
        state = wait_for(FAMILY+'_confirmation',FAMILY+'_confirmation.py')
        result = ROOT/'results'/('research_'+FAMILY+'_confirmation')
        audit = json.loads((result/'verification.json').read_text())
        assert audit['passed'] and audit['actual_seed_heads_checked'] == (108 if source_mass else 90)
        assert audit['actual_training_witnesses_checked'] == (72 if source_mass else 54)
        assert audit['generative_fits_checked'] == 18
        assert audit['matched_constant_control_required'] == source_mass
        verify(protocol); step('refresh_recent_progress.py')
        if state['confirmation_gate_passed']:
            status('confirmation_passed_requires_reference_review',family=FAMILY,reference_changed=False)
            return
    status('completed_mass_calibration_decision_requires_next_exploration',reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except BaseException:
            status('failed',error=traceback.format_exc()); raise


if __name__ == '__main__': main()

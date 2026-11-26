"""Observe one bound worker and execute only its preregistered review/confirmation."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from experiment import ROOT, dump
from start_large_lastblock_readout_coupling import FAMILY, sha, live

OUT = ROOT/'runs'/(FAMILY+'_observer')


def wait_worker(family, pid, entrypoint):
    while True:
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        if state['status'] == 'completed':
            return state
        assert state['status'] != 'failed', (family, state)
        assert live(pid, entrypoint), ('Bound worker exited before completion', family, pid, state)
        dump(OUT/'state.json', {'status': 'waiting_'+family, 'pid': os.getpid(),
            'worker_pid': pid, 'worker_status': state['status'], 'external_test_evaluated': False})
        time.sleep(10)


def step(entrypoint, *arguments):
    dump(OUT/'state.json', {'status': 'running_predeclared_step', 'pid': os.getpid(),
        'entrypoint': entrypoint, 'arguments': arguments, 'external_test_evaluated': False})
    result = subprocess.run([sys.executable, '-u', str(ROOT/entrypoint), *arguments], cwd=ROOT)
    assert result.returncode == 0, (entrypoint, arguments, result.returncode)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    def frozen():
        for key in ['code_sha256', 'frozen_artifacts']:
            for name, expected in protocol[key].items():
                assert sha(name) == expected, name
    frozen()
    state = wait_worker(FAMILY, protocol['worker_pid'], FAMILY+'_experiments.py')
    result = ROOT/'results'/('research_'+FAMILY)
    audit = json.loads((result/'verification.json').read_text())
    comparison = json.loads((result/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked'] == audit['actual_training_witnesses_checked'] == state['new_seed_fits'] == 36
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert len(eligible) == state['passing_development_gates']
    step('review_'+FAMILY+'_outputs.py')
    saved = json.loads((result/'independent_saved_output_review.json').read_text())
    assert saved['review_passed'] and saved['gate_passers'] == len(eligible)
    step('review_'+FAMILY+'_completed_root.py')
    final = json.loads((ROOT/'results'/(FAMILY+'_completed_root_review.json')).read_text())
    assert final['passed'] and final['gate_passers'] == len(eligible)
    if eligible:
        selected = json.loads((result/'selection_before_repeat.json').read_text())['candidate']
        assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
        step('verify_'+FAMILY+'_confirmation.py')
        step('start_'+FAMILY+'_confirmation.py')
        hypothesis = json.loads((ROOT/'results'/('next_'+FAMILY+'_hypothesis.json')).read_text())
        pid = hypothesis['confirmation_pid']
        assert live(pid, FAMILY+'_confirmation.py')
        fresh = wait_worker(FAMILY+'_confirmation', pid, FAMILY+'_confirmation.py')
        folder = ROOT/'results'/('research_'+FAMILY+'_confirmation')
        checked = json.loads((folder/'verification.json').read_text())
        scored = json.loads((folder/'comparison.json').read_text())
        assert checked['passed'] and checked['actual_seed_heads_checked'] == fresh['new_seed_fits'] == 126
        assert checked['actual_training_witnesses_checked'] == 90
        assert checked['matched_controls_required'] == ['token_joint', 'readout_only']
        step('review_'+FAMILY+'_fresh_outputs.py', '--prepare')
        step('review_'+FAMILY+'_fresh_outputs.py')
        step('review_'+FAMILY+'_completed_root.py', '--fresh')
        final = json.loads((ROOT/'results'/(FAMILY+'_fresh_root_review.json')).read_text())
        assert final['passed'] and final['confirmation_gate_passed'] == scored['confirmation_gate_passed']
        status = ('completed_fresh_positive_requires_inference_and_reference_review' if scored['confirmation_gate_passed']
                  else 'completed_fresh_negative_requires_next_exploration')
        detail = {'confirmation_gate_passed': scored['confirmation_gate_passed'], 'fresh_saved_and_final_reviews_passed': True}
    else:
        status = 'completed_negative_requires_next_exploration'
        detail = {'confirmation_launched': False}
    frozen()
    dump(OUT/'state.json', {'status': status, 'pid': os.getpid(),
        'original_saved_output_review_passed': True, 'original_final_root_review_passed': True,
        'original_gate_passers': len(eligible), 'reference_changed': False,
        'external_test_evaluated': False, **detail})


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(),
                'traceback': traceback.format_exc(), 'external_test_evaluated': False})
            raise


if __name__ == '__main__':
    main()

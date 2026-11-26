"""Root-launched bridge from the existing query decision to the preregistered next study."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha

OUT=ROOT/'runs/attention_coupling_after_query'


def status(value,**details):
    dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})


def verify(protocol):
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def step(script):
    status('running_predeclared_step',script=script)
    subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)


def predecessor_status(state,expected_pid,live):
    assert state['status']!='failed',state
    if state['status']=='confirmation_passed_requires_reference_review':return 'positive'
    if state['status']=='completed_query_selection_decision_requires_next_exploration':return 'negative'
    assert state['pid']==expected_pid and live,('Existing observer identity lost; never relaunch',state)
    return 'wait'


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    while True:
        state=json.loads((ROOT/'runs/query_selection_continuation_v2/state.json').read_text())
        path=Path('/proc')/str(protocol['existing_predecessor_observer_pid'])/'cmdline'
        live=path.exists() and str(ROOT/'continue_query_selection_after_mass_v2.py').encode() in path.read_bytes().split(b'\0')
        decision=predecessor_status(state,protocol['existing_predecessor_observer_pid'],live)
        if decision!='wait':break
        status('waiting_existing_query_decision',worker_pid=protocol['existing_predecessor_observer_pid'],worker_status=state['status'])
        time.sleep(10)
    if decision=='positive':
        status('positive_predecessor_requires_reference_review',reference_changed=False);return
    verify(protocol)
    saved=ROOT/'results/research_register_patch_query_selection/independent_saved_output_review.json'
    if not saved.exists():step('review_register_patch_query_selection_outputs.py')
    assert json.loads(saved.read_text())['review_passed']
    verify(protocol);step('start_register_patch_attention_coupling.py')
    hypothesis=json.loads((ROOT/'results/next_register_patch_attention_coupling_hypothesis.json').read_text())
    worker_pid=hypothesis['pid']
    process_path=Path('/proc')/str(worker_pid)/'cmdline'
    assert process_path.exists() and str(ROOT/'register_patch_attention_coupling_experiments.py').encode() in process_path.read_bytes().split(b'\0')
    # Only the original starter above launches training. This observer has no relaunch path.
    observer=ROOT/'runs/attention_coupling_continuation';observer.mkdir(exist_ok=True)
    assert not (observer/'protocol.json').exists(), 'Never duplicate an observer.'
    formal_path=ROOT/'runs/register_patch_attention_coupling/protocol.json'
    formal=json.loads(formal_path.read_text())
    artifacts={**protocol['frozen_artifacts'],str(formal_path):sha(formal_path),str(saved):sha(saved)}
    artifacts.update(formal['frozen_artifacts'])
    code={**protocol['code_sha256'],**formal['code_sha256']}
    observer_protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'execution_entrypoint':'continue_register_patch_attention_coupling.py','existing_worker_pid':worker_pid,
        'code_sha256':code,'frozen_artifacts':artifacts,
        'policy':'Observe the same48fit worker, run frozen saved-output review once, confirm only highest eligible originalCV candidate under90/126fit rules. Positive requires root reference review. Never automatically score external sets, promote references or restart workers.'}
    dump(observer/'protocol.json',observer_protocol)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'attention_coupling_continuation.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'continue_register_patch_attention_coupling.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(observer/'launch_record.json',{'pid':process.pid,'existing_worker_pid':worker_pid,'protocol_sha256':sha(observer/'protocol.json')})
    status('handed_off_to_attention_coupling_observer',worker_pid=worker_pid,observer_pid=process.pid,reference_changed=False)
    subprocess.run([sys.executable,str(ROOT/'refresh_recent_progress.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT,check=True)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()

"""Recover only the frozen selected fresh confirmation; never relaunch original48fits."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from continue_query_selection_after_mass_v2 import resolve_worker_pid
OUT=ROOT/'runs/class_patch_evidence_confirmation_recovery_v2'


def status(value,**details):dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})
def verify(protocol):
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():
            p=Path(name);assert sha(p if p.is_absolute() else ROOT/p)==expected,name

def live(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and str(ROOT/'class_query_patch_evidence_confirmation.py').encode() in p.read_bytes().split(b'\0')

def wait(pid):
    while True:
        state=json.loads((ROOT/'runs/class_query_patch_evidence_confirmation/state.json').read_text())
        assert state['status']!='failed',state
        if state['status']=='completed':return state
        actual=resolve_worker_pid(state,pid,live)
        status('waiting_same_fresh_confirmation',worker_pid=actual,worker_status=state['status'],arm=state.get('arm'),repeat=state.get('repeat'),fold=state.get('fold'),seed=state.get('seed'))
        time.sleep(10)

def step(name):
    status('running_frozen_recovery_step',script=name)
    subprocess.run([sys.executable,'-u',str(ROOT/name)],cwd=ROOT,check=True)

def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    assert not (ROOT/'runs/class_query_patch_evidence_confirmation/protocol.json').exists(),'Never duplicate confirmation.'
    result=ROOT/'results/research_class_query_patch_evidence'
    state=json.loads((ROOT/'runs/class_query_patch_evidence/state.json').read_text())
    audit=json.loads((result/'verification.json').read_text());review=json.loads((result/'independent_saved_output_review.json').read_text())
    comparison=json.loads((result/'comparison.json').read_text());selected=json.loads((result/'selection_before_repeat.json').read_text())['candidate']
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert state['status']=='completed' and audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==48
    assert review['review_passed'] and review['gate_passers']==len(eligible)==2
    assert selected==max(eligible,key=lambda r:r['cv']['mean_macro_f1']) and selected['name']=='class_query_pool_m8replace'
    assert selected['class_query_patch_evidence_configuration']['learned_query']
    step('start_class_query_patch_evidence_confirmation.py')
    hypothesis=json.loads((ROOT/'results/next_class_query_patch_evidence_hypothesis.json').read_text());pid=hypothesis['confirmation_pid'];assert live(pid)
    status('fresh_confirmation_started',worker_pid=pid,original48fit_worker_restarted=False)
    terminal=wait(pid);verify(protocol)
    audit=json.loads((ROOT/'results/research_class_query_patch_evidence_confirmation/verification.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==126 and audit['actual_training_witnesses_checked']==90 and audit['generative_fits_checked']==18
    assert audit['matched_controls_required']==['matched_uniform','matched_source_query']
    step('refresh_recent_progress.py');step('report.py')
    status('confirmation_passed_requires_reference_review' if terminal['confirmation_gate_passed'] else 'completed_class_patch_evidence_decision_requires_next_exploration',reference_changed=False)


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()

"""Observe one bound formal worker, review saved outputs, and execute only predeclared confirmation."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha
OUT=ROOT/'runs/class_patch_spatial_semivariance_observer'

def live(pid,entrypoint):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and str(ROOT/entrypoint).encode() in p.read_bytes().split(b'\0')

def wait_worker(family,pid,entrypoint):
    while True:
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text())
        if state['status']=='completed':return state
        assert state['status']!='failed',(family,state)
        assert live(pid,entrypoint),('Bound worker exited before completed state',family,pid,state)
        dump(OUT/'state.json',{'status':'waiting_'+family,'pid':os.getpid(),'worker_pid':pid,'worker_status':state['status'],'external_test_evaluated':False})
        time.sleep(10)

def step(entrypoint):
    dump(OUT/'state.json',{'status':'running_predeclared_step','pid':os.getpid(),'entrypoint':entrypoint,'external_test_evaluated':False})
    result=subprocess.run([sys.executable,'-u',str(ROOT/entrypoint)],cwd=ROOT,check=False)
    assert result.returncode==0,(entrypoint,result.returncode)

def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():assert sha(name)==expected,name
    state=wait_worker('class_patch_spatial_semivariance',protocol['worker_pid'],'class_patch_spatial_semivariance_experiments.py')
    result=ROOT/'results/research_class_patch_spatial_semivariance'
    audit=json.loads((result/'verification.json').read_text());comparison=json.loads((result/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==state['new_seed_fits']==24
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    review_path=result/'independent_saved_output_review.json'
    if not review_path.exists():step('review_class_patch_spatial_semivariance_outputs.py')
    saved=json.loads(review_path.read_text());assert saved['review_passed'] and saved['gate_passers']==len(eligible)
    for name,expected in saved['input_sha256'].items():assert sha(name)==expected,name
    if eligible:
        selection=json.loads((result/'selection_before_repeat.json').read_text())['candidate']
        assert selection==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        assert not (ROOT/'runs/class_patch_spatial_semivariance_confirmation/protocol.json').exists()
        step('verify_class_patch_spatial_semivariance_confirmation.py')
        step('start_class_patch_spatial_semivariance_confirmation.py')
        hypothesis=json.loads((ROOT/'results/next_class_patch_spatial_semivariance_hypothesis.json').read_text())
        pid=hypothesis['confirmation_pid'];assert live(pid,'class_patch_spatial_semivariance_confirmation.py')
        fresh=wait_worker('class_patch_spatial_semivariance_confirmation',pid,'class_patch_spatial_semivariance_confirmation.py')
        folder=ROOT/'results/research_class_patch_spatial_semivariance_confirmation'
        checked=json.loads((folder/'verification.json').read_text());scored=json.loads((folder/'comparison.json').read_text())
        assert checked['passed'] and checked['actual_seed_heads_checked']==fresh['new_seed_fits']==126
        assert checked['actual_training_witnesses_checked']==90 and checked['matched_controls_required']==['matched_mean','matched_variance']
        status='completed_fresh_confirmation_requires_root_saved_output_and_reference_review'
        detail={'confirmation_gate_passed':scored['confirmation_gate_passed']}
    else:
        status='completed_negative_requires_next_exploration';detail={'confirmation_launched':False}
    for section in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[section].items():assert sha(name)==expected,name
    dump(OUT/'state.json',{'status':status,'pid':os.getpid(),'original_saved_output_review_passed':True,'original_gate_passers':len(eligible),'reference_changed':False,'external_test_evaluated':False,**detail})

def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'external_test_evaluated':False});raise
if __name__=='__main__':main()

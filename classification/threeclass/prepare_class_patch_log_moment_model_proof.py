"""Freeze the log-moment model proof before actual cached-image engineering checks."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def main():
    out=ROOT/'runs/class_patch_log_moment_preflight';assert not (out/'protocol.json').exists()
    parent=json.loads((ROOT/'runs/class_patch_dispersion_preflight/protocol.json').read_text())
    code=set(parent['code_sha256'])-EXCLUDED
    for n,expected in parent['code_sha256'].items():
        if n not in EXCLUDED:assert sha(ROOT/n)==expected,n
    code.update(['class_patch_log_moment.py','audit_class_patch_log_moment.py','class_patch_log_moment_witness.py','verify_class_patch_log_moment_model.py','prepare_class_patch_log_moment_model_proof.py'])
    code=scientific_closure(code-EXCLUDED);assert not (set(code)&EXCLUDED)
    artifacts=dict(parent['input_artifact_sha256'])
    names=['runs/class_patch_dispersion_preflight/protocol.json','results/dispersion_scientific_code_inventory_recovery_v2.json','results/class_patch_log_moment_preregistration.json','results/research_class_patch_hadamard_variance/comparison.json','results/research_class_patch_hadamard_variance/verification.json','results/research_class_patch_hadamard_variance/independent_saved_output_review.json','results/class_patch_hadamard_variance_completed_root_review.json']
    for name,expected in artifacts.items():assert sha(name)==expected,name
    for name in names:artifacts[str(ROOT/name)]=sha(ROOT/name)
    assert sha(ROOT/names[2])=='f685cfd3fe411924afd23abb8f85d3bf2d4569cb7f1801349049c1d99136d326'
    audit=json.loads((ROOT/names[4]).read_text());saved=json.loads((ROOT/names[5]).read_text());root=json.loads((ROOT/names[6]).read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['gate_passers']==0 and root['passed']
    assert json.loads((ROOT/'runs/class_patch_hadamard_variance_observer/state.json').read_text())['status']=='completed_negative_requires_next_exploration'
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_class_patch_log_moment_model.py','code_sha256':{n:sha(ROOT/n) for n in code},'input_artifact_sha256':artifacts,'excluded_mutable_reporting_modules_before_closure':sorted(EXCLUDED),'classification_fits':0,'classification_scores':0,'actual_model_cases_planned':6,'engineering_optimizer_steps_planned':2,'external_test_evaluated':False,'purpose':'Actual256trainingimages per uniform/source xmean/unrotatedvariance/learneddirectionLogMoment model/task/fullgradient/witness/ordinaryCUDArestore; exact parent reductions, independent max-centered exponentials and direction/derivative/covariance properties. No classifier fit/score/backbone extraction.'}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_model_engineering_proof','classification_fits':0,'classification_scores':0})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'classification_fits':0,'classification_scores':0})
    print(json.dumps({'pid':process.pid,'scientific_modules':len(code),'frozen_inputs':len(artifacts),'protocol_sha256':sha(out/'protocol.json')}),flush=True)
if __name__=='__main__':main()

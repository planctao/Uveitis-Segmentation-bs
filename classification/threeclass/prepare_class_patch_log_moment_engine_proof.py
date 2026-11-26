"""Freeze original/fresh learned-direction log-moment engines and patient/control decisions before proof execution."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def main():
    out=ROOT/'runs/class_patch_log_moment_engine_preflight'
    assert not (out/'protocol.json').exists()
    model_path=ROOT/'verification_class_patch_log_moment_model.json';model=json.loads(model_path.read_text())
    assert model['passed'] and len(model['cases'])==6 and model['engineering_optimizer_steps']==2
    assert model['classification_fits']==model['classification_scores']==model['external_images_read']==0
    code=set(model['code_sha256'])-EXCLUDED
    for n,expected in model['code_sha256'].items():assert sha(ROOT/n)==expected,n
    code.update(['class_patch_log_moment_experiments.py','class_patch_log_moment_confirmation.py','class_patch_log_moment_plans.py','verify_class_patch_log_moment_confirmation_plan.py','prepare_class_patch_log_moment_engine_proof.py','review_class_patch_log_moment_outputs.py'])
    code=scientific_closure(code-EXCLUDED);assert not (set(code)&EXCLUDED)
    artifacts=dict(model['input_artifact_sha256']);artifacts[str(model_path)]=sha(model_path)
    for row in model['cases']:artifacts[row['first_task_witness_path']]=row['first_task_witness_sha256']
    saved_path=ROOT/'results/research_class_patch_dispersion/independent_saved_output_review.json';saved=json.loads(saved_path.read_text())
    assert saved['review_passed'] and saved['gate_passers']==0
    artifacts.update(saved['input_sha256'])
    from class_patch_log_moment_plans import assert_unused
    inventory=assert_unused(ROOT)
    for p in inventory:artifacts[str(p)]=sha(p)
    for name,expected in artifacts.items():assert sha(name)==expected,name
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_class_patch_log_moment_confirmation_plan.py','code_sha256':{n:sha(ROOT/n) for n in code},'input_artifact_sha256':artifacts,'excluded_mutable_reporting_modules_before_closure':sorted(EXCLUDED),'classification_fits':0,'classification_scores':0,'actual_engine_cases_planned':6,'fresh_patient_plan_count':6,'supervised_fits_if_eligible':126,'actual_task_witnesses_if_eligible':90,'external_test_evaluated':False,'purpose':'Original/fresh warmstart, both equal-capacity matched controls, actual256 complete task/allgradients/predict/ordinaryCUDArestore; six patient+duplicate-isolated plans,4fusionformulas,60original/29freshgatecases. No classifier fitting or score.'}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_engine_proof','classification_fits':0,'classification_scores':0})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'classification_fits':0,'classification_scores':0})
    print(json.dumps({'pid':process.pid,'code_modules':len(code),'frozen_inputs':len(artifacts),'protocol_sha256':sha(out/'protocol.json')}),flush=True)
if __name__=='__main__':main()

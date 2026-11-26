"""Freeze actual original/fresh108 shared engines before executing patient/task/control proof."""
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from start_class_query_patch_evidence import scientific_closure
from large_lastblock_delta_lora_plans import assert_unused
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    out=ROOT/'runs/large_lastblock_delta_lora_engine_preflight';assert not (out/'protocol.json').exists()
    previous=ROOT/'verification_large_lastblock_delta_lora_model.json';model=json.loads(previous.read_text())
    assert model['passed'] and len(model['cases'])==2 and model['engineering_optimizer_steps']==2
    code=set(model['code_sha256'])-EXCLUDED
    for name,expected in model['code_sha256'].items():assert sha(ROOT/name)==expected,name
    code.update(['large_lastblock_delta_lora_engine.py','large_lastblock_delta_lora_data.py','large_lastblock_delta_lora_experiments.py',
        'large_lastblock_delta_lora_confirmation.py','large_lastblock_delta_lora_plans.py','verify_large_lastblock_delta_lora_engine.py','prepare_large_lastblock_delta_lora_engine_proof.py'])
    code=scientific_closure(code-EXCLUDED);assert not set(code).intersection(EXCLUDED)
    inputs=dict(model['input_artifact_sha256']);inputs[str(previous)]=sha(previous)
    inputs[str(ROOT/'runs/large_lastblock_delta_lora_model_preflight/protocol.json')]=sha(ROOT/'runs/large_lastblock_delta_lora_model_preflight/protocol.json')
    for case in model['cases']:inputs[case['actual_first_task_witness_path']]=case['actual_first_task_witness_sha256']
    for path in assert_unused(ROOT):inputs[str(path)]=sha(path)
    for name,expected in inputs.items():assert sha(name)==expected,name
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_large_lastblock_delta_lora_engine.py',
        'code_sha256':{name:sha(ROOT/name) for name in code},'input_artifact_sha256':inputs,'package_implementation_sha256':model['package_implementation_sha256'],
        'excluded_reporting_modules_before_closure':sorted(EXCLUDED),'actual_engine_cases':2,'actual_images_per_case':256,'patient_plans':6,
        'original_gate_cases':24,'fresh_gate_cases':25,'fusion_formula_cases':2,'fresh_supervised_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,
        'classification_fits':0,'classification_scores':0,'scope':'Actual shared engine/warmstart/task input/allgradient audit/ordinary CUDArestore, unused6patientplans and fixed original/fresh decisions; no optimizer trajectory or accuracy score.'}
    dump(out/'protocol.json',protocol)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':p.pid,'protocol_sha256':sha(out/'protocol.json')})
    hypothesis=ROOT/'results/next_large_lastblock_delta_lora_hypothesis.json';data=json.loads(hypothesis.read_text())
    dump(hypothesis,{**data,'status':'shared_engine_patient_gate_engineering_running','engine_proof_pid':p.pid,'engine_proof_protocol':str((out/'protocol.json').relative_to(ROOT))})
    print(json.dumps({'pid':p.pid,'scientific_modules':len(code),'inputs':len(inputs),'protocol_sha256':sha(out/'protocol.json'),'classifier_training_launched':False}),flush=True)
if __name__=='__main__':main()

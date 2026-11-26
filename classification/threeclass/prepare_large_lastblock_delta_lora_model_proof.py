"""Freeze actual task/gradient/anchor/control model engineering before executing it."""
import hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call
from experiment import ROOT,dump
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    out=ROOT/'runs/large_lastblock_delta_lora_model_preflight';assert not (out/'protocol.json').exists()
    previous=ROOT/'verification_large_lastblock_delta_lora_cache_v2.json';proof=json.loads(previous.read_text())
    assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==0
    code=set(proof['code_sha256'])-EXCLUDED
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected,name
    code.update(['large_lastblock_delta_lora.py','audit_large_lastblock_delta_lora.py','verify_large_lastblock_delta_lora_model.py','prepare_large_lastblock_delta_lora_model_proof.py'])
    code=scientific_closure(code-EXCLUDED);assert not set(code).intersection(EXCLUDED)
    inputs=dict(proof['input_artifact_sha256']);inputs[str(previous)]=sha(previous)
    parent=ROOT/'runs/large_lastblock_delta_lora_cache_preflight_v2/protocol.json';inputs[str(parent)]=sha(parent)
    for name in ['runs/large_token_evidence/registers/validation/seed_0/model.pt','verification_large_token_feature_bank.json','features/multiscale_vit.json']:
        inputs[str(ROOT/name)]=sha(ROOT/name)
    feature_proof=json.loads((ROOT/'verification_large_token_feature_bank.json').read_text())
    inputs[feature_proof['feature_storage']]=feature_proof['feature_bank_sha256']
    large_parent=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    for path in [Path(large_parent['feature_storage']),Path(large_parent['feature_storage']).with_suffix('.json')]:
        inputs[str(path)]=sha(path)
    for path,expected in inputs.items():assert sha(path)==expected,path
    packages=dict(json.loads(parent.read_text())['package_implementation_sha256'])
    for obj in [checkpoint,functional_call]:
        path=str(Path(inspect.getfile(obj)).resolve());packages[path]=sha(path)
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_large_lastblock_delta_lora_model.py',
        'indices':proof['metadata']['indices'],'code_sha256':{name:sha(ROOT/name) for name in code},'input_artifact_sha256':inputs,
        'package_implementation_sha256':packages,'excluded_reporting_modules_before_closure':sorted(EXCLUDED),
        'actual_model_cases':2,'actual_images_per_case':256,'engineering_optimizer_steps':2,'classification_fits':0,'classification_scores':0,
        'scope':'Both actual256complete tasks/allgradients atzero/nonzero, frozen source/classifier, float32RAM producer reconstruction, zero anchors, actual8checkpoint/direct gradient check andCUDArestore. No classifier score or fit.'}
    dump(out/'protocol.json',protocol)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':p.pid,'protocol_sha256':sha(out/'protocol.json')})
    hypothesis=ROOT/'results/next_large_lastblock_delta_lora_hypothesis.json';data=json.loads(hypothesis.read_text())
    dump(hypothesis,{**data,'status':'actual256_model_engineering_running','model_proof_pid':p.pid,'model_proof_protocol':str((out/'protocol.json').relative_to(ROOT))})
    print(json.dumps({'pid':p.pid,'scientific_modules':len(code),'inputs':len(inputs),'protocol_sha256':sha(out/'protocol.json'),'classification_fits':0}),flush=True)
if __name__=='__main__':main()

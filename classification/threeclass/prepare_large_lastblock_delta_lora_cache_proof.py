"""Freeze before any new activation extraction, then start one actual engineering worker."""
import hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import timm,torch
from timm.models.eva import Eva,apply_rot_embed_cat
from experiment import ROOT,dump
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    out=ROOT/'runs/large_lastblock_delta_lora_cache_preflight';assert not (out/'protocol.json').exists()
    prereg=ROOT/'results/large_lastblock_delta_lora_preregistration.json'
    assert sha(prereg)=='fb5448b0e829ac1f606c73266fe546bc7cbfc512cc884bfe9d71b0b553bedca1'
    saved=ROOT/'results/research_class_patch_log_moment/independent_saved_output_review.json'
    reviewed=ROOT/'results/class_patch_log_moment_completed_root_review.json'
    assert json.loads(saved.read_text())['review_passed'] and json.loads(saved.read_text())['gate_passers']==0
    assert json.loads(reviewed.read_text())['passed']
    assert json.loads((ROOT/'runs/class_patch_log_moment_observer/state.json').read_text())['status']=='completed_negative_requires_next_exploration'
    code=scientific_closure({'large_lastblock_delta_lora_cache.py','verify_large_lastblock_delta_lora_cache.py','prepare_large_lastblock_delta_lora_cache_proof.py'}-EXCLUDED)
    assert not set(code).intersection(EXCLUDED)
    parent=ROOT/'runs/dinov3_large/protocol.json';weights=json.loads(parent.read_text())['pretrained_weights']
    artifacts=[prereg,saved,reviewed,parent,ROOT/'metadata/manifest.csv',ROOT/'results/research_reference_registry.json',
        ROOT/'results/research_reference_m8.json',ROOT/'results/large_lastblock_delta_lora_static_size_clarification_v2.json',Path(weights['storage'])]
    assert sha(weights['storage'])==weights['sha256']
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20294400).permutation(len(fit))[:256]]
    artifacts.extend(Path(frame.iloc[index].cache_path) for index in ids)
    package={str(Path(inspect.getfile(obj)).resolve()):None for obj in [Eva,apply_rot_embed_cat]}
    package={path:sha(path) for path in package}
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'verify_large_lastblock_delta_lora_cache.py',
        'indices':ids.tolist(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),'preregistration_sha256':sha(prereg),
        'code_sha256':{name:sha(ROOT/name) for name in code},
        'input_artifact_sha256':{str(path):sha(path) for path in sorted(set(artifacts))},
        'package_implementation_sha256':package,'timm_version':timm.__version__,'torch_version':torch.__version__,
        'excluded_reporting_modules_before_initial_closure':sorted(EXCLUDED),
        'classification_fits':0,'classification_scores':0,'actual_training_images_planned':256,
        'actual_full_backbone_views_planned':512,'independent_prefix_views_planned':16,'actual_float32_lastblock_views_planned':512,
        'scope':'Freeze finite-precision float16 RAM-cache bridge and producer/source image/backbone/package bytes before extraction. No classifier task/gradient/anchor/control proof or scientific score from this step.'}
    dump(out/'protocol.json',protocol)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (out/'proof.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(out/'protocol.json')})
    hypothesis=ROOT/'results/next_large_lastblock_delta_lora_hypothesis.json';document=json.loads(hypothesis.read_text())
    dump(hypothesis,{**document,'status':'cache_bridge_engineering_running','new_activation_extraction_launched':True,'cache_proof_pid':process.pid,'cache_proof_protocol':str((out/'protocol.json').relative_to(ROOT))})
    print(json.dumps({'pid':process.pid,'scientific_modules':len(code),'input_artifacts':len(protocol['input_artifact_sha256']),'protocol_sha256':sha(out/'protocol.json'),'classifier_training_launched':False}),flush=True)
if __name__=='__main__':main()

"""Review completed cache/model/shared-engine contracts and immutable bytes without replay."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 target=ROOT/'results/large_lastblock_delta_lora_engineering_root_review.json';assert not target.exists()
 inputs={};modules=set()
 def check(name,expected=None):
  path=Path(name);path=path if path.is_absolute() else ROOT/path
  digest=sha(path)
  if expected is not None:assert digest==expected,str(path)
  assert str(path) not in inputs or inputs[str(path)]==digest
  inputs[str(path)]=digest;return path
 def read(name):return json.loads(check(name).read_text())
 cache=read('verification_large_lastblock_delta_lora_cache_v2.json')
 model=read('verification_large_lastblock_delta_lora_model.json')
 engine=read('verification_large_lastblock_delta_lora_engine.json')
 pairs=[(cache,'large_lastblock_delta_lora_cache_preflight_v2'),(model,'large_lastblock_delta_lora_model_preflight'),(engine,'large_lastblock_delta_lora_engine_preflight')]
 for proof,family in pairs:
  assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==proof['external_raw_images_read']==0
  protocol=read('runs/'+family+'/protocol.json');state=read('runs/'+family+'/state.json')
  assert state['status']=='completed'
  check('runs/'+family+'/protocol.json',proof.get('engineering_protocol_sha256',proof.get('protocol_sha256')))
  assert not set(proof['code_sha256']).intersection(EXCLUDED)
  for section in ['code_sha256','input_artifact_sha256','package_implementation_sha256']:
   assert proof[section]==protocol[section]
   for name,digest in proof[section].items():check(name,digest)
  modules.update(proof['code_sha256'])
 assert cache['metadata']['full_backbone_view_evaluations']==512
 assert cache['metadata']['float32_lastblock_view_evaluations']==512
 assert cache['metadata']['independent_prefix_view_checks']==16
 assert cache['metadata']['cache_storage_dtype']=='float32'
 assert len(model['cases'])==2 and model['engineering_optimizer_steps']==2
 for case in model['cases']:
  assert case['actual_images']==256 and case['trainable_parameters']==16384 and case['engineering_optimizer_steps']==1
  for key in ['zero_source_raw_features_probability_subtype_embeddings_and_complete_task_bitwise','zero_a_gradient_exactly_zero','zero_b_gradient_nonzero','nonzero_a_and_b_gradients_finite_and_nonzero','ordinary_cuda_restore_bitwise','all_other_parameters_and_buffers_unchanged']:assert case[key],key
  assert case['cross_microbatch_positive_pairs_in_single256_task']>0
  check(case['actual_first_task_witness_path'],case['actual_first_task_witness_sha256'])
 assert len(engine['plans'])==6 and len(engine['synthetic_formula_cases'])==2
 assert len(engine['development_gate_cases'])==24 and len(engine['confirmation_gate_cases'])==25
 assert engine['supervised_fits_if_eligible']==108 and engine['actual_task_witnesses_if_eligible']==72
 assert len(engine['actual_engine_cases'])==2
 for case in engine['actual_engine_cases']:
  assert case['actual_images']==256 and case['ordinary_cuda_restore_bitwise'] and case['exact_source_warmstart']
  assert case['original_fresh_same_trainer_predict_restore_witness_entrypoints']
  witness=case['actual_task_witness'];assert witness['actual_images']==256 and witness['trainable_parameters']==16384
  for key in ['zero_a_gradient_exact','zero_b_gradient_nonzero','actual_source_and_ram_input_hashes_verified','original_cuda_outputs_sha256_restored','frozen_head_and_block_verified']:assert witness[key],key
 prereg=read('results/large_lastblock_delta_lora_preregistration_v2.json')
 check('results/large_lastblock_delta_lora_preregistration_v2.json','ee784341a64d46619fa28990e7452ac76e5410faeea6d39790df3d43d21d72c3')
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_input_bytes_checked':len(inputs),'scientific_modules_checked':len(modules),
  'completed_engineering_cases':2,'actual_images_per_case':256,'patient_plans':6,'original_gate_cases':24,'fresh_gate_cases':25,
  'original_fits_planned':24,'fresh_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,'engineering_optimizer_steps':2,
  'proof_sha256':{name:sha(ROOT/name) for name in ['verification_large_lastblock_delta_lora_cache_v2.json','verification_large_lastblock_delta_lora_model.json','verification_large_lastblock_delta_lora_engine.json']},
  'code_sha256':sha(__file__),'classification_fits':0,'classification_scores':0,'model_task_or_image_replays':0,
  'scope':'Saved completed contracts and actual immutable code/package/input bytes checked. No proof replay. Engine temporary checkpoint/warmstart files were verified inside its completed actual proof; they were not persisted or rechecked here. Float32 storage recovery changes no architecture/hyperparameters/gates. Full formal startup still required.'}
 with target.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()

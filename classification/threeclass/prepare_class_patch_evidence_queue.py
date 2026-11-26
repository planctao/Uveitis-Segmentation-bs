"""Root byte/contract review and frozen queue preparation; does not launch models."""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/uveitis_threeclass_20261124')
cache={}
def sha(path):
 path=Path(path);path=path if path.is_absolute() else ROOT/path
 actual=path.resolve(strict=True);st=actual.stat();key=(str(actual),st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 if key not in cache:
  with actual.open('rb') as f:cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[key]
def read(name):return json.loads((ROOT/name).read_text())
def write(name,data):
 p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
code={};artifacts={}
def check_section(record,section):
 for name,expected in record[section].items():
  assert sha(name)==expected,name
  if section=='code_sha256':
   if name in code:assert code[name]==expected,name
   code[name]=expected
  else:
   path=Path(name);path=path if path.is_absolute() else ROOT/path
   if str(path) in artifacts:assert artifacts[str(path)]==expected,name
   artifacts[str(path)]=expected
records=['runs/register_patch_attention_coupling/protocol.json',
 'runs/class_query_patch_evidence_preflight/protocol.json','verification_class_query_patch_evidence_model.json',
 'runs/class_query_patch_evidence_engine_preflight/protocol.json','verification_class_query_patch_evidence_confirmation_plan.json',
 'verification_class_query_patch_evidence_startup.json','verification_class_patch_evidence_continuation_v2.json',
 'verification_class_patch_evidence_after_coupling.json','results/class_patch_evidence_observer_proof_filename_recovery.json',
 'results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json',
 'runs/class_query_patch_evidence_saved_review/protocol.json','runs/attention_coupling_continuation/protocol.json',
 'runs/register_patch_attention_coupling_saved_review/protocol.json',
 'results/research_register_patch_attention_coupling/independent_saved_output_review.json',
 'results/research_register_patch_attention_coupling/comparison.json','results/research_register_patch_attention_coupling/verification.json',
 'results/research_register_patch_attention_coupling/freeze_integrity_after_training.json']
for name in records:
 record=read(name);artifacts[str(ROOT/name)]=sha(name)
 for section in ['code_sha256','frozen_artifacts','input_artifact_sha256','input_sha256','source_checkpoint_sha256','source_prediction_sha256']:
  if isinstance(record.get(section),dict):check_section(record,section)
  elif section=='code_sha256' and isinstance(record.get(section),str):
   entry={'runs/class_query_patch_evidence_saved_review/protocol.json':'review_class_query_patch_evidence_outputs.py',
    'runs/register_patch_attention_coupling_saved_review/protocol.json':'review_register_patch_attention_coupling_outputs.py',
    'results/research_register_patch_attention_coupling/independent_saved_output_review.json':'review_register_patch_attention_coupling_outputs.py'}[name]
   assert sha(entry)==record[section];code[entry]=record[section]
model=read('verification_class_query_patch_evidence_model.json');plans=read('verification_class_query_patch_evidence_confirmation_plan.json')
prereg=read('results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json')
assert model['passed'] and len(model['cases'])==4 and model['engineering_optimizer_steps']==4
for row in model['cases']:
 assert row['actual_images']==256 and row['ordinary_cuda_checkpoint_restore_bitwise']
 assert row['source_p_subtype_embedding_task_bitwise_at_zero'] and row['within_class_conditional_disease_odds_preserved']
 assert row['normal_shift_exact_zero'] and row['all_frozen_buffers_unchanged_after_one_optimizer_step']
 assert sha(row['first_task_witness_path'])==row['first_task_witness_sha256']
 artifacts[row['first_task_witness_path']]=row['first_task_witness_sha256']
assert plans['passed'] and len(plans['plans'])==6 and len(plans['synthetic_formula_cases'])==4
assert len(plans['development_gate_cases'])==40 and len(plans['confirmation_gate_cases'])==24
assert len(plans['actual_matched_control_engine_cases'])==2
assert len(plans['actual_prediction_restore_engine_cases'])==len(plans['original_source_engine_preflight_cases'])==4
assert plans['control_winner_supervised_fits']==90 and plans['learned_winner_supervised_fits']==126
for record in [model,plans]:assert record['classification_fits']==record['classification_scores']==record['external_images_read']==0
for name,count,key in [('verification_class_query_patch_evidence_startup.json',7,'guard_cases'),('verification_class_patch_evidence_continuation_v2.json',7,'cases'),('verification_class_patch_evidence_after_coupling.json',6,'cases')]:
 r=read(name);assert r['passed'] and len(r[key])==count and r['child_processes_started']==0
assert model['preregistration_sha256']==sha('results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json')
hypothesis=read('results/next_class_query_patch_evidence_hypothesis.json')
for key in ['configuration_plan','architecture_plan','fixed_training','original_selection','confirmation_plan']:assert hypothesis[key]==prereg[key]
assert hypothesis['classification_fits']==hypothesis['scores_evaluated']==0 and not hypothesis['training_launched']
worker=read('runs/register_patch_attention_coupling/state.json');observer=read('runs/attention_coupling_continuation/state.json')
assert worker['status']=='completed' and worker['new_seed_fits']==48 and worker['passing_development_gates']==0
assert observer['status']=='completed_attention_coupling_decision_requires_next_exploration' and observer['pid']==77585
comparison=read('results/research_register_patch_attention_coupling/comparison.json');audit=read('results/research_register_patch_attention_coupling/verification.json')
assert not any(r['development_gate_passed'] for r in comparison['candidates'])
assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==48
saved=read('results/research_register_patch_attention_coupling/independent_saved_output_review.json');assert saved['review_passed'] and saved['gate_passers']==0
registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
assert read(registry['active_reference'])['retained_development_model']=='M8';artifacts[str(ROOT/registry['active_reference'])]=registry['active_reference_sha256']
# Include scripts invoked by name; import closure alone misses subprocess targets.
explicit=['continue_class_patch_evidence_after_coupling.py','continue_class_query_patch_evidence.py',
 'verify_class_patch_evidence_continuation.py','verify_class_patch_evidence_continuation_v2.py','verify_class_patch_evidence_after_coupling.py',
 'start_class_query_patch_evidence.py','start_class_query_patch_evidence_confirmation.py','verify_class_query_patch_evidence_confirmation.py',
 'review_class_query_patch_evidence_outputs.py','verify_class_query_patch_evidence_startup.py','prepare_class_patch_evidence_queue.py']
excluded={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
pending=list(set(code)|set(explicit));names=set(pending)
while pending:
 name=pending.pop()
 for node in ast.walk(ast.parse((ROOT/name).read_text())):
  modules=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
  for module in modules:
   dependency=module+'.py'
   if dependency not in names and dependency not in excluded and (ROOT/dependency).exists():names.add(dependency);pending.append(dependency)
for name in names:
 if name in code:assert sha(name)==code[name],name
 code[name]=sha(name)
review={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'model_cases_checked':4,'model_engineering_optimizer_steps':4,'original_prediction_restore_cases_checked':4,'actual_matched_control_task_cases_checked':2,
 'patient_plans_checked':6,'original_gate_boundary_cases_checked':40,'confirmation_gate_boundary_cases_checked':24,
 'startup_guard_cases_checked':7,'observer_wait_loop_cases_checked':7,'predecessor_decision_cases_checked':6,
 'predecessor_completed_negative_saved_review_passed':True,'existing_predecessor_observer_pid':77585,
 'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
 'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,
 'scope':'Root byte/contract review of existing actual engineering proofs and completed negative predecessor. Does not replay model/task proofs, claim architecture performance or prove unexecuted formal/fresh/full saved-output runtime.'}
write('results/class_query_patch_evidence_queue_root_review.json',review)
artifacts[str(ROOT/'results/class_query_patch_evidence_queue_root_review.json')]=sha('results/class_query_patch_evidence_queue_root_review.json')
protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'continue_class_patch_evidence_after_coupling.py',
 'existing_predecessor_observer_pid':77585,'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
 'policy':'Predecessor terminal negative and independent saved review are proven. Launch the preregistered four-arm48fit study once, bind its actual worker PID to the observer, review outputs and only highest eligible candidate receives90/126fit confirmation. No automatic external scoring, reference promotion or original-worker restart.'}
write('runs/class_patch_evidence_after_coupling/protocol.json',protocol)
print(json.dumps({'review_passed':True,'code_modules':len(code),'frozen_artifacts':len(artifacts),'physical_hashes_checked':len(cache),'queue_protocol_sha256':sha('runs/class_patch_evidence_after_coupling/protocol.json')}),flush=True)

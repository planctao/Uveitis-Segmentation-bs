"""Freeze reviewed environment-only recovery and launch selected fresh confirmation observer once."""
import ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/class_patch_evidence_confirmation_recovery_v2'
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
CACHE={}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in CACHE:
  with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return CACHE[key]
def read(name):return json.loads((ROOT/name).read_text())
def write(name,data):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def main():
 assert not (OUT/'protocol.json').exists()
 assert not (ROOT/'runs/class_query_patch_evidence_confirmation/protocol.json').exists()
 for pid in [12654,12660,14083]:assert not Path(f'/proc/{pid}/cmdline').exists(),pid
 parent=read('runs/class_query_patch_evidence/protocol.json');observer=read('runs/class_patch_evidence_continuation/protocol.json')
 assert sha('runs/class_query_patch_evidence/protocol.json')=='8006a8ec78f58dc292c8931a17f922d21c20c9589e67dbe1a548e8af4173af3e'
 assert sha('runs/class_patch_evidence_continuation/protocol.json')=='1c0ea17a72a517b67e0bd100a1edc92be29616b8bc259a51753efcef25837ce3'
 code={};artifacts={}
 for record in [parent,observer]:
  assert not (set(record['code_sha256']) & EXCLUDED)
  for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
   for name,expected in record.get(section,{}).items():
    assert sha(name)==expected,name
    if section=='code_sha256':
     if name in code:assert code[name]==expected
     code[name]=expected
    else:
     p=Path(name);artifacts[str(p if p.is_absolute() else ROOT/p)]=expected
 print('Unchanged original scientific source/input inventories checked.',flush=True)
 names=['runs/class_query_patch_evidence/protocol.json','runs/class_patch_evidence_continuation/protocol.json',
  'results/class_query_patch_evidence_formal_root_review.json','runs/class_query_patch_evidence/state.json',
  'runs/class_patch_evidence_continuation/state.json','class_patch_evidence_continuation.log',
  'runs/dispersion_after_direct_patch_v2/state.json','dispersion_after_direct_patch_v2.log',
  'class_query_patch_evidence_confirmation_original_start_failure.log',
  'results/research_class_query_patch_evidence/comparison.json','results/research_class_query_patch_evidence/verification.json',
  'results/research_class_query_patch_evidence/freeze_integrity_after_training.json',
  'results/research_class_query_patch_evidence/independent_saved_output_review.json',
  'results/research_class_query_patch_evidence/selection_before_repeat.json',
  'verification_class_query_patch_evidence_confirmation.json','verification_class_query_patch_evidence_confirmation_plan.json',
  'verification_class_patch_evidence_confirmation_recovery_v2.json',
  'results/dispersion_scientific_code_inventory_recovery_v2.json',
  'results/historical_confirmation_storage_recovery_20261125.json',
  'results/continuous_search_storage_recovery_20261125.json',
  'results/continuous_search_failure_and_relocation_review_20261125.json',
  'results/continuous_log_storage_recovery_20261125.json','results/validation_source_storage_recovery_20261125.json']
 for name in names:artifacts[str(ROOT/name)]=sha(name)
 completed=read('results/research_class_query_patch_evidence/verification.json');review=read('results/research_class_query_patch_evidence/independent_saved_output_review.json')
 selection=read('results/research_class_query_patch_evidence/selection_before_repeat.json');comparison=read('results/research_class_query_patch_evidence/comparison.json')
 eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
 assert read('runs/class_query_patch_evidence/state.json')['status']=='completed'
 assert completed['passed'] and completed['actual_seed_heads_checked']==completed['actual_training_witnesses_checked']==48
 assert review['review_passed'] and review['gate_passers']==len(eligible)==2
 assert selection['candidate']==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
 assert selection['candidate']['name']=='class_query_pool_m8replace' and len(selection['checkpoint_sha256'])==12
 for p,expected in selection['checkpoint_sha256'].items():assert sha(p)==expected;artifacts[p]=expected
 failed=read('runs/class_patch_evidence_continuation/state.json');assert failed['status']=='failed' and 'refresh_recent_progress.py' in failed['error']
 assert read('runs/dispersion_after_direct_patch_v2/state.json')['status']=='failed'
 original_failure=(ROOT/'class_query_patch_evidence_confirmation_original_start_failure.log').read_text()
 assert 'shutil.disk_usage' in original_failure and 'AssertionError' in original_failure
 for name in ['verification_class_query_patch_evidence_confirmation.json','verification_class_query_patch_evidence_confirmation_plan.json','verification_class_patch_evidence_confirmation_recovery_v2.json']:
  record=read(name);assert record['passed']
  for p,expected in record.get('code_sha256',{}).items():assert sha(p)==expected;code[p]=expected
  for p,expected in record.get('input_artifact_sha256',{}).items():assert sha(p)==expected;artifacts[p]=expected
 assert len(read('verification_class_patch_evidence_confirmation_recovery_v2.json')['cases'])==7
 storage=read('results/historical_confirmation_storage_recovery_20261125.json')
 assert storage['passed'] and Path(storage['source']).is_symlink() and Path(storage['source']).resolve()==Path(storage['destination']).resolve()
 free=shutil.disk_usage('/autodl-fs/data').free;assert free>2*1024**3
 registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
 assert read(registry['active_reference'])['retained_development_model']=='M8';artifacts[str(ROOT/registry['active_reference'])]=registry['active_reference_sha256']
 explicit={'prepare_class_patch_evidence_confirmation_recovery_v2.py','continue_class_patch_evidence_confirmation_recovery_v2.py',
  'start_class_query_patch_evidence_confirmation.py','class_query_patch_evidence_confirmation.py','class_query_patch_evidence_plans.py'}
 scientific=set(code)|explicit;pending=list(scientific)
 while pending:
  name=pending.pop()
  for node in ast.walk(ast.parse((ROOT/name).read_text())):
   mods=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
   for mod in mods:
    dep=mod+'.py'
    if dep not in scientific and dep not in EXCLUDED and (ROOT/dep).exists():scientific.add(dep);pending.append(dep)
 assert not (scientific & EXCLUDED)
 for name in scientific:
  if name in code:assert sha(name)==code[name],name
  code[name]=sha(name)
 recovery={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'finding':'Original48fits and saved-output review completed. Original observer failed in mutable reporting refresh before fresh confirmation. Original frozen fresh starter then correctly refused insufficient shared space before any protocol or fit. Historical completed archive relocated byte-for-byte, preserving absolute logical paths. Retry the unchanged fresh starter through a new observer only.',
  'original_completed_fits':48,'original_observer_pid':12660,'old_worker_and_observer_and_bridge_absent':[12654,12660,14083],
  'original_failed_state_and_logs_sha256':{n:sha(n) for n in names if n.endswith('.log') or n.endswith('/state.json')},
  'selected_candidate':'class_query_pool_m8replace','eligible_candidates':2,'selected_checkpoints_checked':12,
  'original_fresh_protocol_absent_before_recovery':True,'shared_free_bytes_before_launch':free,
  'original_frozen_startup_guard_unchanged':True,'original_worker_restarted':False,'mutable_reporting_excluded_from_scientific_code':sorted(EXCLUDED),
  'code_modules_checked':len(code),'frozen_artifacts_checked':len(artifacts),'classification_fits':0,'classification_scores':0,'model_predictions':0,'external_images_read':0,
  'scope':'Source/input/result byte and saved decision checks; no model/task replay or promotion. Fresh126fits/90witnesses still required.'}
 write('results/class_patch_evidence_confirmation_recovery_v2_root_review.json',recovery)
 artifacts[str(ROOT/'results/class_patch_evidence_confirmation_recovery_v2_root_review.json')]=sha('results/class_patch_evidence_confirmation_recovery_v2_root_review.json')
 protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'continue_class_patch_evidence_confirmation_recovery_v2.py',
  'selected_candidate':'class_query_pool_m8replace','planned_new_seed_fits':126,'planned_training_witnesses':90,
  'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
  'policy':'Run unchanged frozen selected fresh starter once; do not restart original48fit worker/observer/bridge. Observe same fresh126fit worker; require90actual witnesses,18Gaussianfits and both matched controls. No external scoring or automatic reference promotion.'}
 write('runs/class_patch_evidence_confirmation_recovery_v2/protocol.json',protocol)
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 with (ROOT/'class_patch_evidence_confirmation_recovery_v2.log').open('a') as log:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('runs/class_patch_evidence_confirmation_recovery_v2/launch_record.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'pid':p.pid,'protocol_sha256':sha(OUT/'protocol.json'),'selected_candidate':'class_query_pool_m8replace'})
 print(json.dumps({'review_passed':True,'recovery_observer_pid':p.pid,'code_modules':len(code),'frozen_artifacts':len(artifacts),'protocol_sha256':sha(OUT/'protocol.json'),'shared_free_bytes':free}),flush=True)
if __name__=='__main__':main()

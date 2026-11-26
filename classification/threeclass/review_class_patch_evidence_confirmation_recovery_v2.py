"""Review the actual frozen126-fit confirmation launch and same-worker observer binding."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from class_query_patch_evidence_plans import make_plans,OUTER_SEEDS
ROOT=Path(__file__).resolve().parent
cache={}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in cache:
  with p.open('rb') as f:cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[key]
def read(name):return json.loads((ROOT/name).read_text())
def main():
 destination=ROOT/'results/class_patch_evidence_confirmation_formal_recovery_v2_root_review.json';assert not destination.exists()
 formal_path=ROOT/'runs/class_query_patch_evidence_confirmation/protocol.json';formal=read(formal_path)
 recovery_path=ROOT/'runs/class_patch_evidence_confirmation_recovery_v2/protocol.json';recovery=read(recovery_path)
 assert sha(formal_path)=='aa3b38617b7b312620324aa039c01ceae8fde8d0cc5d2021d3a25f27931ccfe0'
 assert sha(recovery_path)=='212de6330d1fb1d75906cb77b7b0f775c0b9a27515b48402e70356deb0930566'
 for record in [formal,recovery]:
  assert not set(record['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
  for section in ['code_sha256','frozen_artifacts','split_plan_sha256','fresh_seed_audit']:
   for name,expected in record.get(section,{}).items():assert sha(name)==expected,name
 selected=read('results/research_class_query_patch_evidence/selection_before_repeat.json')['candidate']
 assert selected['name']==formal['selected_candidate']=='class_query_pool_m8replace'
 assert formal['selected_configuration']=={'architecture':selected['class_query_patch_evidence_configuration'],'replacement_weight':.25}
 assert formal['planned_new_seed_fits']==126 and formal['planned_training_witnesses']==90 and formal['planned_generative_fits']==18
 assert formal['outer_split_seeds']==OUTER_SEEDS==[20293601,20293611]
 assert {'matched_uniform','matched_uniform_control','matched_source_query','matched_source_query_control'}.issubset(formal['arms']) and len(formal['arms'])==14
 frame=pd.read_csv(ROOT/'metadata/manifest.csv');plans=make_plans(frame,sha('metadata/manifest.csv'))
 assert len(plans)==len(formal['split_plan_sha256'])==6
 for plan in plans:
  path=ROOT/'runs/class_query_patch_evidence_confirmation/split_plans'/f"repeat_{plan['repeat']}_fold_{plan['fold']}.json"
  assert json.loads(path.read_text())==plan
 hypothesis=read('results/next_class_query_patch_evidence_hypothesis.json');worker=hypothesis['confirmation_pid']
 launch=read('runs/class_patch_evidence_confirmation_recovery_v2/launch_record.json');observer=launch['pid']
 for pid,name in [(worker,formal['execution_entrypoint']),(observer,recovery['execution_entrypoint'])]:
  assert str(ROOT/name).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 state=read('runs/class_patch_evidence_confirmation_recovery_v2/state.json')
 assert state['pid']==observer and state['worker_pid']==worker and state['status']=='waiting_same_fresh_confirmation'
 registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
 assert read(registry['active_reference'])['retained_development_model']=='M8'
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'formal_protocol_sha256':sha(formal_path),
  'recovery_observer_protocol_sha256':sha(recovery_path),'selected_candidate':selected['name'],'replacement_weight':.25,
  'formal_code_modules_checked':len(formal['code_sha256']),'recovery_code_modules_checked':len(recovery['code_sha256']),
  'formal_frozen_artifacts_checked':len(formal['frozen_artifacts']),'recovery_frozen_artifacts_checked':len(recovery['frozen_artifacts']),
  'patient_and_duplicate_isolated_six_plans_recomputed':True,'unused_seed_inventory_bytes_checked':len(formal['fresh_seed_audit']),
  'planned_new_seed_fits':126,'planned_training_witnesses':90,'planned_generative_fits':18,
  'matched_controls_required':['matched_uniform','matched_source_query'],'worker_pid':worker,'recovery_observer_pid':observer,
  'actual_worker_and_observer_cmdlines_verified':True,'observer_bound_to_same_fresh_worker':True,'reference_changed':False,
  'classification_fits':0,'classification_scores':0,'model_predictions':0,'external_images_read':0,
  'code_sha256':{'review_class_patch_evidence_confirmation_recovery_v2.py':sha('review_class_patch_evidence_confirmation_recovery_v2.py')},
  'scope':'Actual formal-protocol/source/input-byte, fresh-plan and worker-observer binding review. Does not claim completed126fits or actual model/task runtime audit. No reference promotion or external score.'}
 with destination.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()

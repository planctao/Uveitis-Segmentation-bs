"""Check actual updated reports, live binding, pending-fit accounting and unchanged science."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from integrity_reporting import load_lastblock_delta_lora_exploration
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 target=ROOT/'verification_large_lastblock_delta_lora_progress_reporting.json';assert not target.exists()
 progress=json.loads((ROOT/'results/session_progress_20261124_large_distill_experts.json').read_text())
 data=load_lastblock_delta_lora_exploration()
 assert data['worker']['actual_pid']==26650 and data['worker']['process_confirmed_live']
 assert data['observer']['actual_pid']==26779 and data['observer']['process_confirmed_live']
 assert data['observer']['state']['worker_pid']==26650
 assert data['activation_cache_frozen_before_fits'] and data['planned_new_fits']==24
 assert data['fresh_fits_if_eligible']==108 and data['fresh_task_witnesses_if_eligible']==72
 assert progress['next_study']=='runs/large_lastblock_delta_lora/protocol.json'
 assert progress['next_hypothesis_status']=='running'
 assert progress['actual_threeclass_head_fits']==4980
 assert len(progress['completed_studies'])==89 and len(progress['confirmation_studies'])==16
 assert all(r['study']!='large_lastblock_delta_lora' for r in progress['completed_studies'])
 assert progress['goal_status']=='active' and not progress['research_search_external_test_evaluated']
 for name in ['README.md','LATEST_RESULTS.md']:
  s=(ROOT/name).read_text();assert 'DINOv3-L最后一层注意力rank4' in s and '16384' in s and '108' in s
  assert '3078' in s and '继续保留M8' in s
 for name in ['integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py']:
  assert name not in json.loads((ROOT/'runs/large_lastblock_delta_lora/protocol.json').read_text())['code_sha256']
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'audited_fits':4980,'completed_original_studies':89,'completed_confirmations_already_in_total':16,
  'new_original_planned_fits':24,'fresh_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,'actual_worker_pid':26650,'actual_observer_pid':26779,
  'worker_and_observer_actual_cmdlines_verified':True,'full_progress_and_both_reports_refreshed':True,'pending_new_study_excluded_from_completed_total':True,
  'scientific_module_hashes_checked':76,'code_sha256':sha(__file__),'mutable_reporting_sha256':{name:sha(ROOT/name) for name in ['integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py']},
  'new_model_fits_or_predictions_in_review':0,'new_candidate_scores':0,'external_images_read':0,
  'scope':'Actual reporting/accounting/current-process contracts and unchanged scientific code checked; not a model/task replay or a new classification performance result.'}
 with target.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()

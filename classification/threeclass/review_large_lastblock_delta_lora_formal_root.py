"""Verify actual frozen formal launch and bound observer; do not replay engineering."""
import json
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT
from start_large_lastblock_delta_lora import sha

def main():
 target=ROOT/'results/large_lastblock_delta_lora_formal_root_review.json';assert not target.exists()
 run=ROOT/'runs/large_lastblock_delta_lora';observer=ROOT/'runs/large_lastblock_delta_lora_observer'
 inputs={}
 def check(name,expected=None):
  path=Path(name);path=path if path.is_absolute() else ROOT/path;digest=sha(path)
  if expected is not None:assert digest==expected,str(path)
  assert str(path) not in inputs or inputs[str(path)]==digest;inputs[str(path)]=digest
  return path
 def read(name):return json.loads(check(name).read_text())
 protocol=read(run/'protocol.json');preview=read('runs/large_lastblock_delta_lora_startup/protocol_preview.json')
 assert {k:v for k,v in protocol.items() if k!='created_at_utc'}=={k:v for k,v in preview.items() if k!='created_at_utc'}
 startup=read('verification_large_lastblock_delta_lora_startup.json');assert startup['passed'] and startup['success_path_protocol_constructed_without_launch']
 check(startup['protocol_preview'],startup['protocol_preview_sha256'])
 for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
  for name,expected in protocol[section].items():check(name,expected)
 observe=read(observer/'protocol.json');launch=read(run/'launch_record.json');observe_launch=read(observer/'launch_record.json')
 assert launch['actual_success_preview_matched'] and observe['worker_pid']==observe_launch['bound_worker_pid']==launch['pid']
 check(run/'protocol.json',launch['protocol_sha256']);check(observer/'protocol.json',observe_launch['protocol_sha256'])
 for section in ['code_sha256','frozen_artifacts']:
  for name,expected in observe[section].items():check(name,expected)
 process=[]
 for pid,entrypoint in [(launch['pid'],'large_lastblock_delta_lora_experiments.py'),(observe_launch['pid'],'observe_large_lastblock_delta_lora.py')]:
  command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
  assert str(ROOT/entrypoint).encode() in command
  process.append({'pid':pid,'entrypoint':entrypoint,'actual_cmdline_verified':True})
 assert len(protocol['actual_development_image_sha256'])==2091 and protocol['planned_new_seed_fits']==24
 for name,expected in protocol['actual_development_image_sha256'].items():check(name,expected)
 assert not set(protocol['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
 rule=read('runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json')
 assert rule['expected_completed_models']==108 and rule['expected_actual_task_witnesses']==72
 assert not rule['outcome_known_before_review_rule'] and rule['expected_training_products']==576
 assert rule['expected_saved_fold_metric_sets']==72 and rule['expected_fixed_fusion_checks']==30 and rule['expected_paired_comparisons']==6
 assert read('results/research_reference_registry.json')['active_reference']=='results/research_reference_m8.json'
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_input_bytes_checked':len(inputs),'scientific_modules_checked':len(protocol['code_sha256']),
  'actual_development_image_bytes_checked':2091,'formal_protocol_sha256':sha(run/'protocol.json'),'observer_protocol_sha256':sha(observer/'protocol.json'),
  'actual_process_binding':process,'actual_success_preview_matched':True,'fresh_saved_review_rule_before_any_outcome':True,'fresh_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,
  'code_sha256':sha(__file__),'new_classifier_fits_or_scores':0,'model_or_image_forward_replays':0,'reference_changed':False,'external_test_evaluated':False,
  'scope':'Actual worker+observer cmdline binding and complete frozen scientific/input/package/image-byte inventory checked. No task/engineering replay and no completed accuracy claim.'}
 with target.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()

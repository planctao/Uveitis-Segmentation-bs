"""Review completed fresh confirmation and its predeclared saved-output review on either outcome."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from experiment import ROOT
from start_large_lastblock_delta_lora import sha
OUT=ROOT/'runs/large_lastblock_delta_lora_fresh_root_review'
TARGET=ROOT/'results/large_lastblock_delta_lora_fresh_root_review.json'

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
def prepare_rule():
 assert not (OUT/'rule_before_outcome.json').exists()
 assert not (ROOT/'results/research_large_lastblock_delta_lora_confirmation/comparison.json').exists()
 saved_rule=ROOT/'runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json'
 saved=json.loads(saved_rule.read_text())
 assert not saved['outcome_known_before_review_rule'] and saved['expected_completed_models']==108
 write(OUT/'rule_before_outcome.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'outcome_known_before_rule':False,
  'code_sha256':sha(__file__),'fresh_saved_rule_sha256':sha(saved_rule),'original_protocol_sha256':sha(ROOT/'runs/large_lastblock_delta_lora/protocol.json'),
  'fits':108,'actual_tasks':72,'products':576,'metric_phases':72,'fixed_fusions':30,'patient_plans':6,'paired_comparisons':6,
  'scope':'Both fresh outcomes require completed original root, actual108/72audit, predeclared saved-output reconstruction and all immutable inputs. No model/task/statistic replay, automatic promotion or external scoring.'})
 print('Fresh root review rule frozen before any fresh outcome.',flush=True)
def main():
 assert not TARGET.exists(),'Preserve completed root review; version recovery.'
 rule=json.loads((OUT/'rule_before_outcome.json').read_text());assert sha(__file__)==rule['code_sha256']
 inputs={}
 def check(name,expected=None):
  path=Path(name);path=path if path.is_absolute() else ROOT/path;digest=sha(path)
  if expected is not None:assert digest==expected,str(path)
  assert str(path) not in inputs or inputs[str(path)]==digest;inputs[str(path)]=digest
  return path
 def read(name):return json.loads(check(name).read_text())
 check(OUT/'rule_before_outcome.json')
 check('runs/large_lastblock_delta_lora/protocol.json',rule['original_protocol_sha256'])
 check('runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json',rule['fresh_saved_rule_sha256'])
 original=read('results/large_lastblock_delta_lora_completed_root_review.json')
 assert original['passed'] and original['control_never_nominated'] and original['gate_passers']>0
 original_saved=read('results/research_large_lastblock_delta_lora/independent_saved_output_review.json')
 assert original['saved_review_sha256']==sha(ROOT/'results/research_large_lastblock_delta_lora/independent_saved_output_review.json')
 run=ROOT/'runs/large_lastblock_delta_lora_confirmation';dest=ROOT/'results/research_large_lastblock_delta_lora_confirmation'
 state=read(run/'state.json');assert state['status']=='completed' and state['new_seed_fits']==108
 formal=read(run/'protocol.json');comparison=read(dest/'comparison.json');audit=read(dest/'verification.json');saved=read(dest/'independent_saved_output_review.json')
 assert formal['planned_new_seed_fits']==108 and formal['planned_training_witnesses']==72
 assert formal['selected_configuration']['architecture']['role']=='sole_upgrade_candidate'
 assert comparison['protocol']==formal and comparison['new_seed_fits']==108
 assert audit['passed'] and audit['actual_seed_heads_checked']==len(audit['fits'])==108
 assert audit['actual_training_witnesses_checked']==sum('witness' in row for row in audit['fits'])==72
 assert audit['matched_controls_required']==['matched_token'] and audit['generative_fits_checked']==18
 assert saved['review_passed'] and saved['completed_classification_fits']==saved['checkpoint_records_checked']==108
 assert saved['completed_actual_task_witnesses']==72 and saved['training_products_checked']==576
 assert saved['saved_fold_metric_sets_recomputed']==len(saved['rows'])==72
 assert saved['fixed_fusion_phases_recomputed']==len(saved['fusion_checks'])==30
 assert saved['paired_patient_comparisons_reconstructed']==len(saved['paired_comparisons'])==6
 assert saved['patient_and_duplicate_group_plans_checked']==6
 assert saved['confirmation_gate_independently_checked'] and not saved['outcome_known_before_review_rule']
 assert saved['training_images_per_repeat']==1717 and saved['case_rows']==3434
 assert set(saved['adapter_selected_epoch_summaries'])=={'value','matched_token'}
 assert all(row['heads']==18 for row in saved['adapter_selected_epoch_summaries'].values())
 assert saved['repeats']==[20294501,20294511]
 assert all(saved[key]==0 for key in ['new_model_fits','new_model_predictions','new_candidate_scores','external_images_read'])
 assert not saved['external_test_evaluated'] and not comparison['external_test_evaluated']
 assert read(dest/'freeze_integrity_after_confirmation.json')['passed']
 assert read(ROOT/'runs/large_lastblock_delta_lora_fresh_saved_review/state.json')['status']=='completed'
 for section in ['code_sha256','frozen_artifacts','split_plan_sha256']:
  for name,digest in formal[section].items():check(name,digest)
 for name,digest in saved['input_sha256'].items():check(name,digest)
 check(saved['case_table'],saved['case_table_sha256'])
 products=read(run/'training_products_freeze_before_audit.json')['input_artifact_sha256'];assert len(products)==576
 for name,digest in products.items():check(name,digest)
 assert not set(formal['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
 controls={'m8','m7','m6','reference','fixed_gaussian','matched_token_control'}
 assert set(saved['paired_comparisons'])==set(comparison['paired_candidate_vs_controls'])==controls
 rows={row['arm']:row for row in comparison['arms']};assert len(rows)==12
 for arm in controls:
  for key,value in saved['paired_comparisons'][arm].items():
   assert np.allclose(value,comparison['paired_candidate_vs_controls'][arm][key],atol=1e-12,rtol=0),(arm,key)
 passed=all(rows['candidate']['mean_macro_f1']>rows[arm]['mean_macro_f1'] and saved['paired_comparisons'][arm]['macro_f1_ci95'][0]>0 for arm in controls)
 assert passed==saved['confirmation_gate_passed']==comparison['confirmation_gate_passed']==state['confirmation_gate_passed']
 registry=read('results/research_reference_registry.json');assert registry['active_reference']=='results/research_reference_m8.json'
 check(registry['active_reference'],registry['active_reference_sha256'])
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'saved_review_sha256':sha(dest/'independent_saved_output_review.json'),
  'frozen_inputs_checked':len(inputs),'scientific_modules_checked':len(formal['code_sha256']),
  'completed_classification_fits':108,'completed_actual_task_witnesses':72,'products':576,'metric_phases':72,'fixed_fusions':30,
  'patient_plans':6,'paired_comparisons':6,'confirmation_gate_passed':passed,'retained_model':'M8','code_sha256':sha(__file__),
  'rule_before_outcome_sha256':sha(OUT/'rule_before_outcome.json'),'new_model_fits_or_predictions':0,'new_candidate_scores':0,'external_images_read':0,
  'scope':'Completed actual108model/72task and predeclared saved72metrics/30fusions/6pairedCI contracts plus immutable bytes checked. No model/task/statistic replay. A positive gate still requires frozen inference/package/reference review; no automatic promotion or external scoring.'}
 write(TARGET,record);print(json.dumps(record),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-rule',action='store_true');args=parser.parse_args()
 if args.prepare_rule:prepare_rule()
 else:main()

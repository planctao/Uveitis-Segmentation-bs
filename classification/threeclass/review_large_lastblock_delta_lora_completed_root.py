"""Review the completed original study's immutable audit and saved-output contracts."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT
from start_large_lastblock_delta_lora import sha
OUT=ROOT/'runs/large_lastblock_delta_lora_completed_root_review'
TARGET=ROOT/'results/large_lastblock_delta_lora_completed_root_review.json'
FAMILY='large_lastblock_delta_lora'

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
def prepare_rule():
 assert not (OUT/'rule_before_outcome.json').exists()
 assert not (ROOT/'results/research_large_lastblock_delta_lora/comparison.json').exists()
 write(OUT/'rule_before_outcome.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'outcome_known_before_rule':False,'code_sha256':sha(__file__),
  'original_protocol_sha256':sha(ROOT/'runs/large_lastblock_delta_lora/protocol.json'),
  'fits':24,'actual_tasks':24,'training_products':120,'metric_phases':16,'matched_token_phases_included':8,
  'fixed_fusions':8,'patient_plans':4,'gate_cases':4,'sole_upgrade_candidate_role':'sole_upgrade_candidate',
  'scope':'Review completed saved contracts and all immutable bytes on either outcome. No image/model/task/statistic replay, no promotion or external scoring.'})
 print('Completed-original root review rule frozen before original outcome.',flush=True)
def main():
 assert not TARGET.exists(),'Preserve completed root review; version any recovery.'
 rule=json.loads((OUT/'rule_before_outcome.json').read_text());assert sha(__file__)==rule['code_sha256']
 inputs={}
 def check(name,expected=None):
  path=Path(name);path=path if path.is_absolute() else ROOT/path;digest=sha(path)
  if expected is not None:assert digest==expected,str(path)
  assert str(path) not in inputs or inputs[str(path)]==digest;inputs[str(path)]=digest
  return path
 def read(name):return json.loads(check(name).read_text())
 check(OUT/'rule_before_outcome.json')
 run=ROOT/'runs'/FAMILY;dest=ROOT/'results'/('research_'+FAMILY)
 protocol=read(run/'protocol.json');check(run/'protocol.json',rule['original_protocol_sha256'])
 state=read(run/'state.json');assert state['status']=='completed' and state['new_seed_fits']==rule['fits']
 saved=read(dest/'independent_saved_output_review.json');audit=read(dest/'verification.json');comparison=read(dest/'comparison.json')
 assert saved['review_passed'] and saved['classification_fits']==saved['actual_task_witnesses']==24
 assert saved['training_products_rechecked']==120 and saved['patient_and_duplicate_group_plans_checked']==4
 assert saved['saved_candidate_phases_metrics_recomputed']==16 and saved['matched_token_control_metric_phases_recomputed']==8
 assert saved['fixed_fusion_phases_recomputed']==8 and saved['original_and_matched_token_control_gates_independently_checked']==4
 assert saved['actual_first_raw_feature_witnesses_checked']==24 and saved['rope_hashes_per_training_witness_checked']==2
 assert all(saved[k]==0 for k in ['new_model_fits','new_model_predictions','new_candidate_scores'])
 assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==24
 assert audit['frozen_register_source_heads_checked']==len(audit['register_source_heads'])==12
 assert audit['reused_fusion_source_heads_checked']==len(audit['source_heads'])==24
 assert audit['only_lowrank_a_b_trained'] and audit['attention_trainable_parameters']==audit['matched_token_control_trainable_parameters']==16384
 assert audit['original_ram_cache_and_frozen_source_verified'] and len(audit['fixed_fusion_phases_checked'])==8
 assert comparison['protocol']==protocol and comparison['new_seed_fits']==24
 assert read(dest/'freeze_integrity_after_training.json')['passed']
 assert not audit['external_test_evaluated'] and not saved['external_test_evaluated']
 products=read(run/'training_products_freeze_before_audit.json')['input_artifact_sha256'];assert len(products)==120
 for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
  for name,digest in protocol[section].items():check(name,digest)
 for name,digest in {**products,**saved['input_sha256']}.items():check(name,digest)
 observer=read(ROOT/'runs/large_lastblock_delta_lora_observer/protocol.json')
 for section in ['code_sha256','frozen_artifacts']:
  for name,digest in observer[section].items():check(name,digest)
 assert not set(protocol['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
 assert len(comparison['candidates'])==4
 controls=[r for r in comparison['candidates'] if r['large_lastblock_delta_lora_configuration']['role']!='sole_upgrade_candidate']
 assert len(controls)==2 and not any(r['development_gate_passed'] for r in controls)
 eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
 assert len(eligible)==state['passing_development_gates']==saved['gate_passers']
 if eligible:
  selected=read(dest/'selection_before_repeat.json')
  assert selected['candidate']==max(eligible,key=lambda row:row['cv']['mean_macro_f1'])
  for name,digest in selected['checkpoint_sha256'].items():check(name,digest)
 else:
  assert not (dest/'selection_before_repeat.json').exists()
  assert not (ROOT/'runs/large_lastblock_delta_lora_confirmation/protocol.json').exists()
  assert read(ROOT/'runs/large_lastblock_delta_lora_observer/state.json')['status']=='completed_negative_requires_next_exploration'
 registry=read(ROOT/'results/research_reference_registry.json')
 assert registry['active_reference']=='results/research_reference_m8.json';check(registry['active_reference'],registry['active_reference_sha256'])
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'saved_review_sha256':sha(dest/'independent_saved_output_review.json'),
  'frozen_inputs_checked':len(inputs),'scientific_modules_checked':len(protocol['code_sha256']),
  'completed_classification_fits':24,'completed_actual_task_witnesses':24,'training_products':120,
  'metric_phases':16,'matched_token_phases_included':8,'fixed_fusions':8,'patient_plans':4,'gate_passers':len(eligible),
  'control_never_nominated':True,'retained_model':'M8','code_sha256':sha(__file__),'rule_before_outcome_sha256':sha(OUT/'rule_before_outcome.json'),
  'new_model_fits_or_predictions':0,'new_candidate_scores':0,'external_images_read':0,
  'scope':'Completed actual model/task audit and independent saved-output contracts plus all immutable bytes reviewed; no model/task/statistic replay. Original eligibility does not establish fresh confirmation or promotion.'}
 if not eligible:record['confirmation_launched']=False
 write(TARGET,record);print(json.dumps(record),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-rule',action='store_true');args=parser.parse_args()
 if args.prepare_rule:prepare_rule()
 else:main()

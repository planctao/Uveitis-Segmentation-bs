"""Build the actual complete frozen protocol without launching a child or training."""
import json
from experiment import ROOT,dump
from start_large_lastblock_delta_lora import build_protocol,OUT,PREREG,sha

def main():
 target=ROOT/'verification_large_lastblock_delta_lora_startup.json';assert not target.exists()
 assert not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
 path=ROOT/'results/next_large_lastblock_delta_lora_hypothesis.json';before=sha(path)
 protocol=build_protocol();prereg=json.loads((ROOT/PREREG).read_text())
 for target_key,source_key in [('architecture','architecture_plan'),('preregistered_configuration_plan','configuration_plan'),('training','fixed_training'),('selection','original_selection'),('confirmation_plan','confirmation_plan'),('fixed_fusions','fixed_fusions'),('ram_activation_cache','ram_activation_cache')]:assert protocol[target_key]==prereg[source_key]
 assert protocol['planned_new_seed_fits']==24 and protocol['predeclared_scores']==4 and protocol['upgrade_candidate_scores']==2
 assert len(protocol['actual_development_image_sha256'])==2091
 assert len(protocol['source_checkpoint_sha256'])==36 and len(protocol['source_prediction_sha256'])==8
 assert {a['role'] for a in protocol['configuration_plan']}=={'sole_upgrade_candidate','matched_capacity_control_not_upgrade_candidate'}
 assert sha(path)==before and not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
 preview=ROOT/'runs/large_lastblock_delta_lora_startup/protocol_preview.json';dump(preview,protocol)
 dump(target,{'passed':True,'success_path_protocol_constructed_without_launch':True,'protocol_preview':str(preview.relative_to(ROOT)),'protocol_preview_sha256':sha(preview),
  'actual_development_image_bytes_frozen':2091,'preregistered_architecture_training_gates_unchanged':True,'matched_token_control_never_upgrade_candidate':True,
  'code_sha256':{n:sha(ROOT/n) for n in ['start_large_lastblock_delta_lora.py','verify_large_lastblock_delta_lora_startup.py']},
  'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,
  'scope':'Actual complete frozen protocol build, including2091rawimage byte identities; no formal run or child launch.24fits/actualtasks and eventual eligible108fresh/72tasks remain separate.'})
 print('Actual last-block delta-LoRA full startup passed; no child or classifier fitting.',flush=True)
if __name__=='__main__':main()

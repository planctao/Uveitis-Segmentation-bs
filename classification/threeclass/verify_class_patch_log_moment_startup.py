"""Exercise actual completed-predecessor/model/engine/control protocol construction without launch."""
import json
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_patch_log_moment import build_protocol,OUT,PREREG

def main():
    destination=ROOT/'verification_class_patch_log_moment_startup.json';assert not destination.exists()
    assert not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
    hypothesis=ROOT/'results/next_class_patch_log_moment_hypothesis.json';before=sha(hypothesis)
    protocol=build_protocol();prereg=json.loads((ROOT/PREREG).read_text())
    for target,source in [('architecture','architecture'),('preregistered_configuration_plan','configuration_plan'),('training','fixed_training'),('selection','original_selection'),('confirmation_plan','confirmation_plan'),('fixed_fusions','fixed_fusions')]:assert protocol[target]==prereg[source]
    assert protocol['planned_new_seed_fits']==protocol['reused_mean_control_trajectories']==protocol['reused_global_variance_control_trajectories']==24 and protocol['predeclared_scores']==4
    assert {r['name'] for r in protocol['frozen_mean_controls']}=={'uniform_pool','source_query_pool','uniform_pool_m8replace','source_query_pool_m8replace'}
    assert {r['name'] for r in protocol['frozen_variance_controls']}=={'uniform_variance','source_query_variance','uniform_variance_m8replace','source_query_variance_m8replace'}
    assert sha(hypothesis)==before and not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
    preview=ROOT/'runs/class_patch_log_moment_startup/protocol_preview.json';dump(preview,protocol)
    dump(destination,{'passed':True,'success_path_protocol_constructed_without_launch':True,'protocol_preview':str(preview.relative_to(ROOT)),'protocol_preview_sha256':sha(preview),'unchanged_preregistered_architecture_training_gates':True,'both_matched_controls_verified':True,'hypothesis_bytes_unchanged':True,'code_sha256':{n:sha(ROOT/n) for n in ['start_class_patch_log_moment.py','verify_class_patch_log_moment_startup.py']},'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,'scope':'Actual complete learned-direction log-moment success path without formal run/state or launch; future actual24training/audit/savedreview and any eligible126freshconfirmation must complete separately.'})
    print('Actual learned-direction log-moment success-path protocol constructed; no classifier fit or child launch.',flush=True)
if __name__=='__main__':main()

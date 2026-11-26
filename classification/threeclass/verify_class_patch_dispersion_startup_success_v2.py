"""Run the real protocol-construction success path with process launch forbidden."""
import json,os
from unittest.mock import patch
from experiment import ROOT,dump
from kernel_residual_experiments import sha
import start_class_patch_dispersion_v2 as starter


def main():
    destination=ROOT/'verification_class_patch_dispersion_startup_success_v2.json';assert not destination.exists()
    out=ROOT/'runs/class_patch_dispersion';assert not (out/'protocol.json').exists()
    hypothesis=ROOT/'results/next_class_patch_dispersion_hypothesis.json';before=sha(hypothesis)
    with patch.dict(os.environ,{'CLASS_PATCH_DISPERSION_PREFLIGHT_ONLY':'1'}),patch('subprocess.Popen',side_effect=RuntimeError('Forbidden preview launch')):
        protocol=starter.main()
    prereg=json.loads((ROOT/'results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json').read_text())
    for target,source in [('architecture','architecture_plan'),('configuration_plan','configuration_plan'),('training','fixed_training'),('selection','original_selection'),('confirmation_plan','confirmation_plan'),('fixed_fusions','fixed_fusions')]:assert protocol[target]==prereg[source]
    assert protocol['planned_new_seed_fits']==24 and protocol['reused_mean_control_trajectories']==24 and protocol['predeclared_scores']==4
    assert {r['name'] for r in protocol['frozen_mean_controls']}=={'uniform_pool','source_query_pool','uniform_pool_m8replace','source_query_pool_m8replace'}
    assert sha(hypothesis)==before and not (out/'protocol.json').exists() and not (out/'state.json').exists()
    preview=ROOT/'runs/class_patch_dispersion_startup_success_v2/protocol_preview.json';dump(preview,protocol)
    dump(destination,{'passed':True,'success_path_protocol_constructed_without_launch':True,
        'protocol_preview':str(preview.relative_to(ROOT)),'protocol_preview_sha256':sha(preview),
        'preregistered_architecture_arms_training_gate_and_confirmation_unchanged':True,'hypothesis_bytes_unchanged':True,
        'code_sha256':{name:sha(ROOT/name) for name in ['start_class_patch_dispersion_v2.py','verify_class_patch_dispersion_startup_success_v2.py','class_patch_dispersion_inventory_recovery_v2.py']},
        'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,
        'scope':'Real completed-predecessor/model/engine/control/protocol path, no formal protocol/state or classifier launched. Full formal and confirmation workflows still require their actual run.'})
    print('Actual dispersion success-path protocol constructed; no classifier fit or child launch.',flush=True)


if __name__=='__main__':main()

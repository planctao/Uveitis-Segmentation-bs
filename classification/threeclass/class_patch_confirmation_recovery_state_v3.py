"""Verified scientific terminal state, distinct from preserved reporting-execution failures."""
import json
from experiment import ROOT
from kernel_residual_experiments import sha
RECOVERY='results/class_patch_confirmation_post_report_recovery_v3.json'
RECOVERY_SHA='4fd35859b88917466202600c9e4a2f14050681f79d7c61e9eb0308e72d52399a'
def recovery():
    assert sha(ROOT/RECOVERY)==RECOVERY_SHA
    value=json.loads((ROOT/RECOVERY).read_text())
    assert value['passed'] and value['actual_seed_heads_checked']==126 and value['actual_training_witnesses_checked']==90 and value['generative_fits_checked']==18
    assert not value['confirmation_gate_passed'] and not value['reference_changed']
    return value
def effective_confirmation_state():return dict(recovery()['recovered_completed_confirmation_state'])
def effective_continuation_state():return dict(recovery()['recovered_continuation_state'])

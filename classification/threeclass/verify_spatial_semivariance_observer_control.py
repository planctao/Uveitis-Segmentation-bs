"""Bound-observer orchestration tests with isolated synthetic states, no training or scores."""
import json,tempfile
from pathlib import Path
from datetime import datetime,timezone
import observe_class_patch_spatial_semivariance as observer
from experiment import ROOT,dump
from kernel_residual_experiments import sha
REAL_ROOT=ROOT

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def run_case(name,eligible=False,confirmation=False,bad_review=False,bad_controls=False):
    with tempfile.TemporaryDirectory(prefix='spatial-observer-control-') as directory:
        root=Path(directory);out=root/'runs/class_patch_spatial_semivariance_observer';out.mkdir(parents=True)
        observer.ROOT=root;observer.OUT=out
        put(out/'protocol.json',{'worker_pid':111,'code_sha256':{},'frozen_artifacts':{}})
        run=root/'runs/class_patch_spatial_semivariance';run.mkdir(parents=True)
        put(run/'state.json',{'status':'completed','new_seed_fits':24,'passing_development_gates':int(eligible)})
        result=root/'results/research_class_patch_spatial_semivariance';result.mkdir(parents=True)
        candidate={'name':'preselected','candidate_id':'preselected','cv':{'mean_macro_f1':.85},'development_gate_passed':True}
        put(result/'verification.json',{'passed':True,'actual_seed_heads_checked':24,'actual_training_witnesses_checked':24})
        put(result/'comparison.json',{'candidates':[candidate] if eligible else []})
        if eligible:put(result/'selection_before_repeat.json',{'candidate':candidate})
        steps=[]
        def step(entrypoint):
            steps.append(entrypoint)
            if entrypoint=='review_class_patch_spatial_semivariance_outputs.py':
                put(result/'independent_saved_output_review.json',{'review_passed':not bad_review,'gate_passers':int(eligible),'input_sha256':{}})
            elif entrypoint=='start_class_patch_spatial_semivariance_confirmation.py':
                fresh=root/'runs/class_patch_spatial_semivariance_confirmation';fresh.mkdir(parents=True)
                put(fresh/'protocol.json',{'test_fixture':True})
                put(fresh/'state.json',{'status':'completed','new_seed_fits':126})
                put(root/'results/next_class_patch_spatial_semivariance_hypothesis.json',{'confirmation_pid':222})
                target=root/'results/research_class_patch_spatial_semivariance_confirmation'
                put(target/'verification.json',{'passed':True,'actual_seed_heads_checked':126,'actual_training_witnesses_checked':90,'matched_controls_required':['matched_mean'] if bad_controls else ['matched_mean','matched_variance']})
                put(target/'comparison.json',{'confirmation_gate_passed':confirmation})
        observer.step=step;observer.live=lambda *args:True
        rejected=False
        try:observer.run()
        except AssertionError:rejected=True
        assert rejected==(bad_review or bad_controls),(name,rejected)
        if not rejected:
            state=json.loads((out/'state.json').read_text())
            assert not state['reference_changed'] and not state['external_test_evaluated']
            if eligible:
                assert state['status']=='completed_fresh_confirmation_requires_root_saved_output_and_reference_review'
                assert state['confirmation_gate_passed']==confirmation and len(steps)==3
            else:assert state['status']=='completed_negative_requires_next_exploration' and len(steps)==1
        return {'case':name,'rejected':rejected,'steps':steps,'passed':True}

def main():
    target=REAL_ROOT/'verification_spatial_semivariance_observer_control.json';assert not target.exists()
    inputs={str(REAL_ROOT/'observe_class_patch_spatial_semivariance.py'):sha(REAL_ROOT/'observe_class_patch_spatial_semivariance.py'),str(REAL_ROOT/'verify_spatial_semivariance_observer_control.py'):sha(REAL_ROOT/'verify_spatial_semivariance_observer_control.py')}
    actual_live=observer.live
    launch=json.loads((REAL_ROOT/'runs/class_patch_spatial_semivariance/launch_record.json').read_text())
    assert actual_live(launch['pid'],'class_patch_spatial_semivariance_experiments.py')
    assert not actual_live(launch['pid'],'wrong_entrypoint.py')
    rows=[run_case('negative_no_confirmation'),run_case('eligible_negative_confirmation',True,False),run_case('eligible_positive_requires_root_review',True,True),run_case('failed_saved_review_rejected',bad_review=True),run_case('missing_variance_control_rejected',True,False,bad_controls=True)]
    with tempfile.TemporaryDirectory(prefix='spatial-observer-wait-') as directory:
        root=Path(directory);out=root/'observer';out.mkdir();observer.ROOT=root;observer.OUT=out
        path=root/'runs/family/state.json'
        observer.live=lambda *args:False
        put(path,{'status':'failed'})
        try:observer.wait_worker('family',111,'test.py')
        except AssertionError:rows.append({'case':'failed_worker_rejected','passed':True})
        else:raise AssertionError('failed worker accepted')
        put(path,{'status':'training'})
        try:observer.wait_worker('family',111,'test.py')
        except AssertionError:rows.append({'case':'exited_worker_rejected','passed':True})
        else:raise AssertionError('exited worker accepted')
        observer.live=lambda *args:True
        observer.time.sleep=lambda seconds:put(path,{'status':'completed'})
        put(path,{'status':'training'})
        assert observer.wait_worker('family',111,'test.py')['status']=='completed'
        rows.append({'case':'active_wait_then_completed','passed':True})
    observer.ROOT=REAL_ROOT
    for name,expected in inputs.items():assert sha(name)==expected
    dump(target,{'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'cases':rows,'actual_live_worker_entrypoint_binding_checked':True,'wrong_entrypoint_rejected':True,'input_sha256':inputs,'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,'scope':'Actual observer control flow with isolated synthetic completed/failed/live states; no scientific/model/metric proof or actual confirmation completion claim.'})
    print('Eight observer control cases passed; actual live-worker binding checked; no classifier fits/scores.',flush=True)
if __name__=='__main__':main()

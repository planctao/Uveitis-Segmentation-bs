"""Check completed Hadamard study contracts/bytes without replaying models or statistics."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    target=ROOT/'results/class_patch_hadamard_variance_completed_root_review.json'
    assert not target.exists(),'Preserve completed root reviews.'
    family='class_patch_hadamard_variance';run=ROOT/'runs'/family;dest=ROOT/'results'/('research_'+family)
    read=lambda p:json.loads(Path(p).read_text())
    state=read(run/'state.json');assert state['status']=='completed' and state['new_seed_fits']==24
    saved=read(dest/'independent_saved_output_review.json');comparison=read(dest/'comparison.json')
    audit=read(dest/'verification.json');protocol=read(run/'protocol.json')
    assert saved['review_passed'] and saved['classification_fits']==saved['actual_task_witnesses']==24
    assert saved['training_products_rechecked']==120 and saved['patient_and_duplicate_group_plans_checked']==4
    assert saved['saved_candidate_phases_metrics_recomputed']==saved['matched_mean_control_saved_phases_recomputed']==saved['matched_variance_control_saved_phases_recomputed']==16
    assert saved['fixed_fusion_phases_recomputed']==8 and saved['original_and_both_matched_control_gates_independently_checked']==4
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==24
    assert audit['frozen_register_source_heads_checked']==len(audit['register_source_heads'])==12
    assert audit['reused_fusion_source_heads_checked']==len(audit['source_heads'])==24
    assert comparison['new_seed_fits']==24 and comparison['protocol']==protocol
    assert read(dest/'freeze_integrity_after_training.json')['passed']
    passing=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(passing)==saved['gate_passers']==state['passing_development_gates']
    assert all(saved[k]==0 for k in ['new_model_fits','new_model_predictions','new_candidate_scores'])
    assert not saved['external_test_evaluated'] and not audit['external_test_evaluated']
    assert not set(protocol['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    inputs={}
    def check(path,digest):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==digest,str(path);key=str(path)
        assert key not in inputs or inputs[key]==digest
        inputs[key]=digest
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for path,digest in protocol[section].items():check(path,digest)
    for path,digest in saved['input_sha256'].items():check(path,digest)
    observer=read(ROOT/'runs/class_patch_hadamard_variance_observer/protocol.json')
    for section in ['code_sha256','frozen_artifacts']:
        for path,digest in observer[section].items():check(path,digest)
    selection=dest/'selection_before_repeat.json'
    if passing:assert read(selection)['candidate']==max(passing,key=lambda r:r['cv']['mean_macro_f1'])
    else:
        assert not selection.exists() and not (ROOT/'runs/class_patch_hadamard_variance_confirmation/protocol.json').exists()
        assert read(ROOT/'runs/class_patch_hadamard_variance_observer/state.json')['status']=='completed_negative_requires_next_exploration'
    reference=read(ROOT/'results/research_reference_registry.json')
    assert reference['active_reference']=='results/research_reference_m8.json'
    check(reference['active_reference'],reference['active_reference_sha256'])
    record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'saved_review_sha256':sha(dest/'independent_saved_output_review.json'),
        'frozen_inputs_checked':len(inputs),'scientific_modules_checked':len(protocol['code_sha256']),'completed_classification_fits':24,'completed_actual_task_witnesses':24,
        'training_products':120,'saved_candidate_metric_phases':16,'matched_mean_metric_phases':16,'matched_variance_metric_phases':16,'fixed_fusions':8,'patient_plans':4,
        'gate_passers':len(passing),'retained_model':'M8','root_review_code_sha256':sha(__file__),'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_images_read':0,
        'scope':'Root completed scientific/saved-review contracts and all frozen input/science bytes checked; no model/task/statistic replay.24actual task semantics rely on completed original independent audit. Original eligibility is not fresh confirmation or promotion.'}
    if not passing:record['confirmation_launched']=False
    with target.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()

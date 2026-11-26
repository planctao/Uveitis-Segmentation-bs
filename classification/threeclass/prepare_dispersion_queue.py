"""Root verification of dispersion engineering records and one predecessor-bound queue."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
cache={}
def sha(name):
    path=Path(name);path=path if path.is_absolute() else ROOT/path
    real=path.resolve(strict=True);st=real.stat();key=(str(real),st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if key not in cache:
        with real.open('rb') as f:cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return cache[key]
def read(name):return json.loads((ROOT/name).read_text())
def write(name,record):
    p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
def main():
    assert not (ROOT/'runs/dispersion_after_direct_patch_v2/protocol.json').exists()
    names=['results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json',
        'runs/class_patch_dispersion_preflight/protocol.json','verification_class_patch_dispersion_model.json',
        'runs/class_patch_dispersion_engine_preflight/protocol.json','verification_class_patch_dispersion_confirmation_plan.json',
        'verification_class_patch_dispersion_startup.json','verification_class_patch_dispersion_continuation.json',
        'verification_dispersion_after_direct_patch.json','verification_dispersion_after_direct_patch_v2.json',
        'results/dispersion_bridge_identity_review_recovery_v2.json','runs/class_patch_dispersion_saved_review/protocol.json',
        'runs/class_query_patch_evidence/protocol.json','runs/class_patch_evidence_continuation/protocol.json']
    code={};artifacts={}
    for name in names:
        record=read(name);artifacts[str(ROOT/name)]=sha(name)
        for section in ['code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','seed_inventory_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,expected in mapping.items():
                    assert sha(path)==expected,path
                    if section=='code_sha256':
                        if path in code:assert code[path]==expected,path
                        code[path]=expected
                    else:
                        target=Path(path);target=target if target.is_absolute() else ROOT/target
                        artifacts[str(target)]=expected
            elif section=='code_sha256':
                assert name=='runs/class_patch_dispersion_saved_review/protocol.json'
                assert sha('review_class_patch_dispersion_outputs.py')==mapping
                code['review_class_patch_dispersion_outputs.py']=mapping
    model=read('verification_class_patch_dispersion_model.json');planning=read('verification_class_patch_dispersion_confirmation_plan.json')
    assert model['passed'] and planning['passed'] and len(model['cases'])==4 and model['engineering_optimizer_steps']==2
    assert model['equal_mean_different_variance_toy_passed'] and model['constant_value_cases']==4
    for row in model['cases']:
        assert row['actual_images']==256 and row['ordinary_cuda_restore_bitwise']
        assert row['trainable_parameters']==row['mean_control_trainable_parameters']==4098
        assert row['zero_source_probability_subtype_and_complete_task_bitwise']
        assert sha(row['first_task_witness_path'])==row['first_task_witness_sha256']
        artifacts[row['first_task_witness_path']]=row['first_task_witness_sha256']
    assert len(planning['plans'])==6 and len(planning['synthetic_formula_cases'])==4
    assert len(planning['development_gate_cases'])==40 and len(planning['confirmation_gate_cases'])==25
    assert len(planning['actual_original_fresh_control_engine_cases'])==4
    assert planning['supervised_fits']==108 and planning['actual_task_witnesses']==72
    for record in [model,planning]:assert record['classification_fits']==record['classification_scores']==record['external_images_read']==0
    for name,key,count in [('verification_class_patch_dispersion_startup.json','guard_cases',7),('verification_class_patch_dispersion_continuation.json','cases',7),('verification_dispersion_after_direct_patch_v2.json','cases',6)]:
        proof=read(name);assert proof['passed'] and len(proof[key])==count and proof['child_processes_started']==0
    bridge_proof=read('verification_dispersion_after_direct_patch_v2.json')
    assert bridge_proof['actual_predecessor_observer_pid_checked']==12660 and bridge_proof['actual_predecessor_cmdline_checked']
    prereg=read('results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json');draft=read('results/next_class_patch_dispersion_hypothesis.json')
    for key in ['architecture_plan','configuration_plan','fixed_training','original_selection','confirmation_plan']:assert prereg[key]==draft[key]
    assert model['preregistration_sha256']==sha('results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json')
    registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
    assert read(registry['active_reference'])['retained_development_model']=='M8'
    artifacts[str(ROOT/registry['active_reference'])]=registry['active_reference_sha256']
    actual=Path('/proc/12660/cmdline');assert actual.exists() and str(ROOT/'continue_class_query_patch_evidence.py').encode() in actual.read_bytes().split(b'\0')
    explicit=['prepare_dispersion_queue.py','continue_dispersion_after_direct_patch.py','continue_dispersion_after_direct_patch_v2.py',
        'continue_class_patch_dispersion.py','verify_class_patch_dispersion_continuation.py','verify_dispersion_after_direct_patch.py','verify_dispersion_after_direct_patch_v2.py',
        'start_class_patch_dispersion.py','start_class_patch_dispersion_confirmation.py','verify_class_patch_dispersion_confirmation.py',
        'review_class_patch_dispersion_outputs.py','verify_class_patch_dispersion_startup.py','verify_class_patch_dispersion_startup_success.py']
    excluded={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
    scientific=set(code)|set(explicit);pending=list(scientific)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            mods=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
            for module in mods:
                dependency=module+'.py'
                if dependency not in scientific and dependency not in excluded and (ROOT/dependency).exists():scientific.add(dependency);pending.append(dependency)
    for name in scientific:
        if name in code:assert sha(name)==code[name],name
        code[name]=sha(name)
    review={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':dict(sorted(code.items())),
        'frozen_artifacts':dict(sorted(artifacts.items())),'model_actual256_cases':4,'engineering_optimizer_steps':2,
        'patient_plans':6,'synthetic_formulas':4,'development_gate_cases':40,'confirmation_gate_cases':25,
        'actual_original_fresh_control_engine_cases':4,'startup_rejection_cases':7,'observer_wait_loop_cases':7,'predecessor_decision_cases':6,
        'actual_predecessor_observer_pid_checked':12660,'actual_predecessor_cmdline_checked':True,
        'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,
        'scope':'Root record/code/input byte and declared-engineering-contract review, not model/task replay. Actual original success path awaits completed negative predecessor and is a compulsory prelaunch queue step. Full original/fresh/output review require actual formal execution.'}
    write('results/class_patch_dispersion_queue_root_review.json',review)
    artifacts[str(ROOT/'results/class_patch_dispersion_queue_root_review.json')]=sha('results/class_patch_dispersion_queue_root_review.json')
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'continue_dispersion_after_direct_patch_v2.py',
        'existing_predecessor_observer_pid':12660,'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
        'policy':'Wait same direct-patch observer. Positive requires root reference review; negative requires saved-output review and actual no-launch protocol-success proof, then start24newvariancefits once, reusing24completed originalmeancontrols. Freeze highest eligible originalCV once;108freshfits/72witnesses with fresh corresponding meancontrol. No automatic external scoring, promotion or original worker restart.'}
    write('runs/dispersion_after_direct_patch_v2/protocol.json',protocol)
    draft.update(status='model_engine_guard_and_queue_preflights_passed_waiting_direct_patch',engineering_optimizer_steps=2,
        model_preflight='verification_class_patch_dispersion_model.json',engine_preflight='verification_class_patch_dispersion_confirmation_plan.json',
        queue_root_review='results/class_patch_dispersion_queue_root_review.json')
    (ROOT/'results/next_class_patch_dispersion_hypothesis.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2))
    env=dict(os.environ);env.pop('CLASS_PATCH_DISPERSION_PREFLIGHT_ONLY',None);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'dispersion_after_direct_patch_v2.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write('runs/dispersion_after_direct_patch_v2/launch_record.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'existing_predecessor_observer_pid':12660,'protocol_sha256':sha('runs/dispersion_after_direct_patch_v2/protocol.json')})
    print(json.dumps({'review_passed':True,'bridge_pid':process.pid,'code_modules':len(code),'frozen_artifacts':len(artifacts),'physical_hashes':len(cache),'protocol_sha256':sha('runs/dispersion_after_direct_patch_v2/protocol.json')}),flush=True)

if __name__=='__main__':main()

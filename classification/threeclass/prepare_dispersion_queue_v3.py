"""Root-reviewed versioned recovery queue bound to the actual selected fresh-confirmation observer."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CACHE={}
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in CACHE:
  with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return CACHE[key]
def read(name):return json.loads((ROOT/name).read_text())
def write(name,data):
 p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def main():
 assert not (ROOT/'runs/dispersion_after_direct_patch_v3/protocol.json').exists()
 assert not (ROOT/'runs/class_patch_dispersion/protocol.json').exists()
 assert not Path('/proc/14083/cmdline').exists()
 recovery=read('results/dispersion_scientific_code_inventory_recovery_v2.json')
 assert sha('results/dispersion_scientific_code_inventory_recovery_v2.json')=='84baa373b7ef8ad64faecfe2be63aab7e2b26104df2713d79dcdadfeec121a6a'
 assert recovery['passed'] and set(recovery['excluded_mutable_reporting'])==EXCLUDED
 code={};artifacts={}
 for name,expected in recovery['scientific_code_sha256'].items():assert sha(name)==expected;code[name]=expected
 for name,expected in recovery['input_artifact_sha256'].items():assert sha(name)==expected;artifacts[name]=expected
 names=['results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json',
  'runs/class_patch_dispersion_preflight/protocol.json','verification_class_patch_dispersion_model.json',
  'runs/class_patch_dispersion_engine_preflight/protocol.json','verification_class_patch_dispersion_confirmation_plan.json',
  'verification_class_patch_dispersion_startup.json','verification_class_patch_dispersion_continuation.json',
  'verification_dispersion_after_direct_patch.json','verification_dispersion_after_direct_patch_v2.json',
  'results/dispersion_bridge_identity_review_recovery_v2.json','runs/class_patch_dispersion_saved_review/protocol.json',
  'results/class_patch_dispersion_queue_root_review.json','runs/dispersion_after_direct_patch_v2/protocol.json',
  'results/dispersion_scientific_code_inventory_recovery_v2.json',
  'verification_class_patch_dispersion_startup_v2.json','verification_class_patch_dispersion_continuation_v2.json',
  'verification_dispersion_after_direct_patch_v3.json',
  'runs/class_query_patch_evidence/protocol.json','runs/class_patch_evidence_continuation/protocol.json',
  'runs/class_query_patch_evidence_confirmation/protocol.json',
  'runs/class_patch_evidence_confirmation_recovery_v2/protocol.json',
  'results/class_patch_evidence_confirmation_recovery_v2_root_review.json',
  'results/class_patch_evidence_confirmation_formal_recovery_v2_root_review.json']
 for name in names:
  record=read(name);artifacts[str(ROOT/name)]=sha(name)
  for section in ['code_sha256','scientific_code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','seed_inventory_sha256']:
   mapping=record.get(section,{})
   if isinstance(mapping,dict):
    for path,expected in mapping.items():
     if section=='code_sha256' and path in EXCLUDED:
      assert expected==recovery['excluded_mutable_reporting'][path]['frozen_sha256'];continue
     assert sha(path)==expected,path
     if section in ['code_sha256','scientific_code_sha256']:
      if path in code:assert code[path]==expected,path
      code[path]=expected
     else:
      p=Path(path);artifacts[str(p if p.is_absolute() else ROOT/p)]=expected
   elif section=='code_sha256':
    assert name=='runs/class_patch_dispersion_saved_review/protocol.json'
    assert sha('review_class_patch_dispersion_outputs.py')==mapping;code['review_class_patch_dispersion_outputs.py']=mapping
 model=read('verification_class_patch_dispersion_model.json');planning=read('verification_class_patch_dispersion_confirmation_plan.json')
 assert model['passed'] and planning['passed'] and len(model['cases'])==4 and model['engineering_optimizer_steps']==2
 assert model['equal_mean_different_variance_toy_passed'] and model['constant_value_cases']==4
 for row in model['cases']:
  assert row['actual_images']==256 and row['ordinary_cuda_restore_bitwise']
  assert row['trainable_parameters']==row['mean_control_trainable_parameters']==4098
  assert row['zero_source_probability_subtype_and_complete_task_bitwise']
  assert sha(row['first_task_witness_path'])==row['first_task_witness_sha256'];artifacts[row['first_task_witness_path']]=row['first_task_witness_sha256']
 assert len(planning['plans'])==6 and len(planning['synthetic_formula_cases'])==4
 assert len(planning['development_gate_cases'])==40 and len(planning['confirmation_gate_cases'])==25
 assert len(planning['actual_original_fresh_control_engine_cases'])==4
 assert planning['supervised_fits']==108 and planning['actual_task_witnesses']==72
 for record in [model,planning]:assert record['classification_fits']==record['classification_scores']==record['external_images_read']==0
 for name,key,count in [('verification_class_patch_dispersion_startup_v2.json','guard_cases',7),('verification_class_patch_dispersion_continuation_v2.json','cases',7),('verification_dispersion_after_direct_patch_v3.json','cases',6)]:
  proof=read(name);assert proof['passed'] and len(proof[key])==count and proof['child_processes_started']==0
 bridge_proof=read('verification_dispersion_after_direct_patch_v3.json')
 predecessor=read('runs/class_patch_evidence_confirmation_recovery_v2/launch_record.json')['pid']
 assert bridge_proof['actual_predecessor_observer_pid_checked']==predecessor==14661 and bridge_proof['actual_predecessor_cmdline_checked']
 assert str(ROOT/'continue_class_patch_evidence_confirmation_recovery_v2.py').encode() in Path(f'/proc/{predecessor}/cmdline').read_bytes().split(b'\0')
 prereg=read('results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json');draft=read('results/next_class_patch_dispersion_hypothesis.json')
 for key in ['architecture_plan','configuration_plan','fixed_training','original_selection','confirmation_plan']:assert prereg[key]==draft[key]
 assert not draft['training_launched'] and draft['classification_fits']==draft['scores_evaluated']==0
 registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
 assert read(registry['active_reference'])['retained_development_model']=='M8';artifacts[str(ROOT/registry['active_reference'])]=registry['active_reference_sha256']
 explicit=['prepare_dispersion_queue_v3.py','class_patch_dispersion_inventory_recovery_v2.py',
  'continue_dispersion_after_direct_patch_v3.py','verify_dispersion_after_direct_patch_v3.py',
  'continue_class_patch_dispersion_v2.py','verify_class_patch_dispersion_continuation_v2.py',
  'start_class_patch_dispersion_v2.py','start_class_patch_dispersion_confirmation_v2.py','verify_class_patch_dispersion_confirmation_v2.py',
  'verify_class_patch_dispersion_startup_v2.py','verify_class_patch_dispersion_startup_success_v2.py']
 scientific=set(code)|set(explicit);pending=list(scientific)
 while pending:
  name=pending.pop()
  for node in ast.walk(ast.parse((ROOT/name).read_text())):
   mods=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
   for mod in mods:
    dep=mod+'.py'
    if dep not in scientific and dep not in EXCLUDED and (ROOT/dep).exists():scientific.add(dep);pending.append(dep)
 assert not (scientific & EXCLUDED)
 for name in scientific:
  if name in code:assert sha(name)==code[name],name
  code[name]=sha(name)
 review={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'finding':'Preserve original dispersion engineering proofs, failed V2 bridge and all scientific source bytes. Versioned startup/selected/fresh/observer copies apply explicit four-reporting-module inventory correction, bind to actual fresh-confirmation recovery observer and repair an unexecuted selected-proof preregistration filename. No architecture, data, fit, selection or gate changes.',
  'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
  'mutable_reporting_excluded':sorted(EXCLUDED),'unchanged_preexisting_scientific_modules':len(recovery['scientific_code_sha256']),
  'model_actual256_cases':4,'engineering_optimizer_steps':2,'patient_plans':6,'actual_original_fresh_control_engine_cases':4,
  'startup_rejection_cases':7,'observer_wait_loop_cases':7,'predecessor_decision_cases':6,
  'actual_predecessor_observer_pid_checked':predecessor,'actual_predecessor_cmdline_checked':True,
  'classification_fits':0,'classification_scores':0,'model_predictions':0,'child_processes_started':0,'external_images_read':0,
  'scope':'Root existing engineering record/code/input byte and current identity review. No actual model/task replay. Full original protocol success proof remains compulsory only after terminal negative fresh predecessor. No classifier or reference promotion.'}
 write('results/class_patch_dispersion_queue_v3_root_review.json',review)
 artifacts[str(ROOT/'results/class_patch_dispersion_queue_v3_root_review.json')]=sha('results/class_patch_dispersion_queue_v3_root_review.json')
 protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'continue_dispersion_after_direct_patch_v3.py',
  'existing_predecessor_observer_pid':predecessor,'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
  'policy':'Wait same new selected fresh-confirmation recovery observer. Positive requires root reference review. Negative requires original saved-output review and actual no-launch protocol-success proof, then start24newvariancefits once, reusing24completed meancontrols. Freeze highest eligible originalCV once;108freshfits/72witnesses with fresh corresponding meancontrol. No automatic external scoring, promotion or original worker restart.'}
 write('runs/dispersion_after_direct_patch_v3/protocol.json',protocol)
 draft.update(status='recovered_queue_waiting_selected_fresh_confirmation',
  recovered_queue_root_review='results/class_patch_dispersion_queue_v3_root_review.json',
  recovered_queue='runs/dispersion_after_direct_patch_v3/protocol.json',
  recovered_original_starter='start_class_patch_dispersion_v2.py')
 (ROOT/'results/next_class_patch_dispersion_hypothesis.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2))
 env=dict(os.environ);env.pop('CLASS_PATCH_DISPERSION_PREFLIGHT_ONLY',None);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 with (ROOT/'dispersion_after_direct_patch_v3.log').open('a') as log:
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('runs/dispersion_after_direct_patch_v3/launch_record.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'existing_predecessor_observer_pid':predecessor,'protocol_sha256':sha('runs/dispersion_after_direct_patch_v3/protocol.json')})
 print(json.dumps({'review_passed':True,'bridge_pid':process.pid,'actual_predecessor_observer_pid':predecessor,'code_modules':len(code),'frozen_artifacts':len(artifacts),'protocol_sha256':sha('runs/dispersion_after_direct_patch_v3/protocol.json')}),flush=True)
if __name__=='__main__':main()

"""Launch one preregistered24-fit dispersion study from the verified completed negative result."""
import ast,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from class_patch_confirmation_recovery_state_v3 import recovery,RECOVERY,RECOVERY_SHA
OUT=ROOT/'runs/dispersion_after_confirmed_negative_v4'
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
def read(name):return json.loads((ROOT/name).read_text())
def write(name,record):
 p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
def status(value,**details):dump(OUT/'state.json',{'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False})
def main():
 assert not (OUT/'protocol.json').exists() and not (ROOT/'runs/class_patch_dispersion/protocol.json').exists()
 recovered=recovery();assert sha(ROOT/RECOVERY)==RECOVERY_SHA
 for pid in [14745,14661,15237]:assert not Path(f'/proc/{pid}/cmdline').exists(),pid
 proof=read('verification_class_patch_dispersion_startup_success_v3.json')
 assert proof['passed'] and proof['success_path_protocol_constructed_without_launch'] and proof['child_processes_started']==0
 preview=read(proof['protocol_preview']);assert sha(ROOT/proof['protocol_preview'])==proof['protocol_preview_sha256']
 assert preview['planned_new_seed_fits']==24 and preview['reused_mean_control_trajectories']==24 and preview['predeclared_scores']==4
 prereg=read('results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json')
 for target,source in [('architecture','architecture_plan'),('configuration_plan','configuration_plan'),('training','fixed_training'),('selection','original_selection'),('confirmation_plan','confirmation_plan'),('fixed_fusions','fixed_fusions')]:assert preview[target]==prereg[source]
 code={};artifacts={}
 for record in [recovered,preview]:
  for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
   for name,expected in record.get(section,{}).items():
    p=Path(name);p=p if p.is_absolute() else ROOT/p;assert sha(p)==expected,name
    if section=='code_sha256':
     if name in code:assert code[name]==expected
     code[name]=expected
    else:artifacts[str(p)]=expected
 for n in [RECOVERY,'verification_class_patch_dispersion_startup_success_v3.json',proof['protocol_preview'],
  'verification_class_patch_dispersion_startup_v3.json','verification_class_patch_dispersion_continuation_v3.json']:
  record=read(n);artifacts[str(ROOT/n)]=sha(ROOT/n)
  for path,expected in record.get('code_sha256',{}).items():assert sha(ROOT/path)==expected;code[path]=expected
 assert read('verification_class_patch_dispersion_startup_v3.json')['passed'] and len(read('verification_class_patch_dispersion_startup_v3.json')['guard_cases'])==7
 assert read('verification_class_patch_dispersion_continuation_v3.json')['passed'] and len(read('verification_class_patch_dispersion_continuation_v3.json')['cases'])==7
 code['launch_class_patch_dispersion_after_negative_v4.py']=sha(__file__)
 code['continue_class_patch_dispersion_v3.py']=sha(ROOT/'continue_class_patch_dispersion_v3.py')
 assert not set(code).intersection(EXCLUDED)
 review={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'recovered_negative_record_sha256':RECOVERY_SHA,
  'scientific_predecessor_completed_negative':True,'original_failed_worker_observer_bridge_states_preserved':True,
  'actual_success_path_protocol_constructed_without_launch':True,'preregistered_architecture_training_selection_confirmation_unchanged':True,
  'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
  'planned_new_seed_fits':24,'reused_mean_control_trajectories':24,'predeclared_scores':4,
  'classification_fits':0,'classification_scores':0,'model_predictions':0,'external_images_read':0,
  'scope':'Root completed-negative recovery, actual full-success startup proof and all science/input byte review before new classifier launch. No model/task replay, external score or reference promotion.'}
 write('results/class_patch_dispersion_after_negative_v4_prelaunch_review.json',review)
 artifacts[str(ROOT/'results/class_patch_dispersion_after_negative_v4_prelaunch_review.json')]=sha(ROOT/'results/class_patch_dispersion_after_negative_v4_prelaunch_review.json')
 control={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'launch_class_patch_dispersion_after_negative_v4.py',
  'code_sha256':dict(sorted(code.items())),'frozen_artifacts':dict(sorted(artifacts.items())),
  'policy':'Completed negative126fit scientific result recovered without rewriting original failure states. Start preregistered24newvariancefits once and observe same actual worker; no restart, external scoring or automatic promotion.'}
 write('runs/dispersion_after_confirmed_negative_v4/protocol.json',control);status('starting_preregistered_variance_study_once')
 env=dict(os.environ);env.pop('CLASS_PATCH_DISPERSION_PREFLIGHT_ONLY',None);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 subprocess.run([sys.executable,'-u',str(ROOT/'start_class_patch_dispersion_v3.py')],cwd=ROOT,env=env,check=True)
 formal_path=ROOT/'runs/class_patch_dispersion/protocol.json';formal=read(formal_path)
 assert {k:v for k,v in formal.items() if k!='created_at_utc'}=={k:v for k,v in preview.items() if k!='created_at_utc'}
 draft=read('results/next_class_patch_dispersion_hypothesis.json');worker=draft['pid']
 assert str(ROOT/'class_patch_dispersion_experiments.py').encode() in Path(f'/proc/{worker}/cmdline').read_bytes().split(b'\0')
 observer=ROOT/'runs/class_patch_dispersion_continuation_v3';assert not (observer/'protocol.json').exists();observer.mkdir(exist_ok=True)
 observer_protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution_entrypoint':'continue_class_patch_dispersion_v3.py','existing_worker_pid':worker,
  'code_sha256':{**control['code_sha256'],**formal['code_sha256']},'frozen_artifacts':{**control['frozen_artifacts'],**formal['frozen_artifacts'],str(formal_path):sha(formal_path)},
  'policy':'Observe the same24fit variance worker, independently review outputs, confirm only highest eligible originalCV under108fit/72witness rules. No worker restart, external score or automatic reference promotion.'}
 write('runs/class_patch_dispersion_continuation_v3/protocol.json',observer_protocol)
 with (ROOT/'class_patch_dispersion_continuation_v3.log').open('a') as log:
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/'continue_class_patch_dispersion_v3.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('runs/class_patch_dispersion_continuation_v3/launch_record.json',{'pid':process.pid,'existing_worker_pid':worker,'protocol_sha256':sha(observer/'protocol.json')})
 formal_review={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'formal_protocol_sha256':sha(formal_path),
  'protocol_preview_sha256':proof['protocol_preview_sha256'],'formal_matches_actual_success_preview_except_creation_time':True,
  'preregistered_architecture_training_selection_confirmation_unchanged':True,'worker_pid':worker,'actual_worker_cmdline_verified':True,
  'observer_pid':process.pid,'observer_same_worker_binding_verified':True,'observer_protocol_sha256':sha(observer/'protocol.json'),
  'formal_scientific_modules':len(formal['code_sha256']),'formal_frozen_artifacts':len(formal['frozen_artifacts']),
  'recovered_negative_predecessor_record_sha256':RECOVERY_SHA,'external_images_read':0,'reference_changed':False,
  'scope':'Actual formal protocol vs proven full-success preview and launched worker/observer identity. No completed-fit, full model/task audit or gain claim.'}
 write('results/class_patch_dispersion_formal_v4_root_review.json',formal_review)
 status('handed_off_to_dispersion_observer',worker_pid=worker,observer_pid=process.pid,reference_changed=False)
 print(json.dumps(formal_review),flush=True)
if __name__=='__main__':main()

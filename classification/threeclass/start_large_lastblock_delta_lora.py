"""Construct the frozen last-block attention/token-control study from completed proofs."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from experiment import ROOT,dump
from start_class_query_patch_evidence import scientific_closure
EXCLUDED={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
OUT=ROOT/'runs/large_lastblock_delta_lora'
PREREG='results/large_lastblock_delta_lora_preregistration_v2.json'
CACHE={}
def sha(name):
 path=Path(name);path=path if path.is_absolute() else ROOT/path;stat=path.stat()
 key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
 if key not in CACHE:
  with path.open('rb') as stream:CACHE[key]=hashlib.file_digest(stream,'sha256').hexdigest()
 return CACHE[key]
def checked(record,code,artifacts):
 for section in ['code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','package_implementation_sha256']:
  mapping=record.get(section,{})
  if not isinstance(mapping,dict):continue
  for name,expected in mapping.items():
   path=Path(name);path=path if path.is_absolute() else ROOT/path
   assert sha(path)==expected,name
   if section=='code_sha256':assert name not in EXCLUDED;code.add(name)
   else:artifacts.add(path)
def image_inputs(frame,mode):
 ids=np.flatnonzero(frame.split.isin(['train','val']) if mode=='original' else frame.split.eq('train'))
 assert len(ids)==(2091 if mode=='original' else 1717)
 return {str(Path(frame.iloc[index].cache_path)):sha(frame.iloc[index].cache_path) for index in ids}
def build_protocol():
 code=set();artifacts=set()
 def read(name):
  path=ROOT/name;artifacts.add(path);return json.loads(path.read_text())
 prereg=read(PREREG);assert sha(ROOT/PREREG)=='ee784341a64d46619fa28990e7452ac76e5410faeea6d39790df3d43d21d72c3'
 assert prereg['planned_original_new_seed_fits']==24
 previous=read('results/research_class_patch_log_moment/independent_saved_output_review.json')
 previous_root=read('results/class_patch_log_moment_completed_root_review.json')
 assert previous['review_passed'] and previous['gate_passers']==0 and previous_root['passed']
 assert previous_root['saved_review_sha256']==sha(ROOT/'results/research_class_patch_log_moment/independent_saved_output_review.json')
 for name,expected in prereg['predecessor_proofs'].items():assert sha(ROOT/name)==expected,name
 state=read('runs/class_patch_log_moment/state.json');assert state['status']=='completed' and state['new_seed_fits']==24
 assert read('runs/class_patch_log_moment_observer/state.json')['status']=='completed_negative_requires_next_exploration'
 engineering=read('results/large_lastblock_delta_lora_engineering_root_review.json');assert engineering['passed']
 assert engineering['original_fits_planned']==24 and engineering['fresh_fits_if_eligible']==108 and engineering['fresh_task_witnesses_if_eligible']==72
 for name,expected in engineering['proof_sha256'].items():assert sha(ROOT/name)==expected,name
 artifacts.add(ROOT/'review_large_lastblock_delta_lora_engineering_root.py')
 assert engineering['code_sha256']==sha(ROOT/'review_large_lastblock_delta_lora_engineering_root.py')
 for name in engineering['proof_sha256']:
  proof=read(name);assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==proof['external_raw_images_read']==0
  checked(proof,code,artifacts)
  for case in proof.get('cases',[]):
   artifacts.add(Path(case['actual_first_task_witness_path']))
   assert sha(case['actual_first_task_witness_path'])==case['actual_first_task_witness_sha256']
 for family in ['large_lastblock_delta_lora_cache_preflight_v2','large_lastblock_delta_lora_model_preflight','large_lastblock_delta_lora_engine_preflight']:
  read('runs/'+family+'/protocol.json');assert read('runs/'+family+'/state.json')['status']=='completed'
 fresh_rule=read('runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json')
 assert fresh_rule['expected_completed_models']==108 and fresh_rule['expected_actual_task_witnesses']==72
 assert fresh_rule['code_sha256']['review_large_lastblock_delta_lora_fresh_outputs.py']==sha(ROOT/'review_large_lastblock_delta_lora_fresh_outputs.py')
 parent=read('runs/class_patch_log_moment/protocol.json')
 checkpoints=parent['source_checkpoint_sha256'];predictions=parent['source_prediction_sha256']
 assert len(checkpoints)==36 and len(predictions)==8
 for name,expected in {**checkpoints,**predictions}.items():assert sha(name)==expected,name
 for name in checkpoints:
  folder=Path(name).parent
  for artifact in folder.iterdir():
   if artifact.is_file():artifacts.add(artifact)
 for phase in ['validation','fold_0','fold_1','fold_2']:
  for root in ['runs/large_token_evidence/geometry','runs/register_patch_query/split_plans']:
   path=ROOT/root/(phase if 'geometry' in root else phase+'.json')
   if path.is_dir():artifacts.update(x for x in path.rglob('*') if x.is_file())
   else:artifacts.add(path)
  for arm in ['registers','registers_m7replace']:
   artifacts.add(ROOT/'runs/large_token_evidence'/arm/phase/'predictions.npz')
 registry=read('results/research_reference_registry.json');current=read(registry['active_reference'])
 assert sha(ROOT/registry['active_reference'])==registry['active_reference_sha256'] and current['retained_development_model']=='M8'
 read('results/milestone5/selection_before_test.json')
 import inspect,timm,torch,torchvision,PIL,numpy,torch.utils.checkpoint
 import torch.nn.functional,torch.nn.modules.linear,torch.nn.modules.normalization
 import torchvision.transforms.functional,torchvision.transforms.transforms,PIL.Image
 package_modules=[timm,torch,torchvision,PIL,numpy,torch.utils.checkpoint,torch.nn.functional,torch.nn.modules.linear,torch.nn.modules.normalization,torchvision.transforms.functional,torchvision.transforms.transforms,PIL.Image]
 package_implementations={str(Path(inspect.getfile(module))):sha(inspect.getfile(module)) for module in package_modules}
 artifacts.update(Path(name) for name in package_implementations)
 package_versions={name:module.__version__ for name,module in [('timm',timm),('torch',torch),('torchvision',torchvision),('PIL',PIL),('numpy',numpy)]}
 frame=pd.read_csv(ROOT/'metadata/manifest.csv');images=image_inputs(frame,'original')
 assert len(images)==2091
 for name in images:artifacts.add(Path(name))
 artifacts.add(ROOT/'metadata/manifest.csv')
 for name in ['start_large_lastblock_delta_lora.py','verify_large_lastblock_delta_lora_startup.py','start_large_lastblock_delta_lora_confirmation.py','verify_large_lastblock_delta_lora_confirmation.py','review_large_lastblock_delta_lora_outputs.py','review_large_lastblock_delta_lora_fresh_outputs.py','observe_large_lastblock_delta_lora.py']:
  code.add(name)
 code=scientific_closure(code-EXCLUDED);assert not set(code).intersection(EXCLUDED)
 return {'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
  'retained_model':'M8','retained_reference':registry['active_reference'],'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
  'hypothesis':prereg['hypothesis'],'architecture':prereg['architecture_plan'],'configuration_plan':prereg['configuration_plan'],
  'preregistered_configuration_plan':prereg['configuration_plan'],'planned_new_seed_fits':24,'predeclared_scores':4,'upgrade_candidate_scores':2,
  'training':prereg['fixed_training'],'fixed_fusions':prereg['fixed_fusions'],'selection':prereg['original_selection'],'confirmation_plan':prereg['confirmation_plan'],
  'package_versions':package_versions,'additional_package_implementation_sha256':package_implementations,
  'ram_activation_cache':prereg['ram_activation_cache'],'actual_development_image_sha256':images,'source_checkpoint_sha256':checkpoints,'source_prediction_sha256':predictions,
  'code_sha256':{name:sha(ROOT/name) for name in code},'frozen_artifacts':{str(path):sha(path) for path in sorted(artifacts)},
  'execution_entrypoint':'large_lastblock_delta_lora_experiments.py','excluded_reporting_modules_before_closure':sorted(EXCLUDED),
  'external_test_policy':'Only2091developmentrawimages;1717train/374val. Sourceimage/backbone/cacheproducer/packagebytes frozen before extraction; RAM cache/content digests frozen before all fits. No external scoring or automatic promotion.'}
def main():
 assert not (OUT/'protocol.json').exists(),'Never duplicate a formal study.'
 proof=json.loads((ROOT/'verification_large_lastblock_delta_lora_startup.json').read_text())
 assert proof['passed'] and proof['success_path_protocol_constructed_without_launch']
 for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected,name
 preview=ROOT/proof['protocol_preview'];assert sha(preview)==proof['protocol_preview_sha256']
 protocol=build_protocol();expected=json.loads(preview.read_text())
 assert {k:v for k,v in protocol.items() if k!='created_at_utc'}=={k:v for k,v in expected.items() if k!='created_at_utc'}
 assert shutil.disk_usage('/autodl-fs/data').free>128*1024**2
 storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/large_lastblock_delta_lora');storage.mkdir(exist_ok=True,parents=True)
 if not OUT.exists():OUT.symlink_to(storage,target_is_directory=True)
 assert OUT.resolve()==storage.resolve()
 rule=ROOT/'runs/large_lastblock_delta_lora_saved_review/protocol.json';assert not rule.exists()
 dump(rule,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':protocol['manifest_sha256'],
  'code_sha256':sha(ROOT/'review_large_lastblock_delta_lora_outputs.py'),'frozen_artifacts':protocol['frozen_artifacts'],
  'purpose':'After completed24model/task audit reconstruct saved16metricphases/8fusions and M8+newtokencontrol gates; control never nominated. No model/image replay.'})
 fresh_rule=ROOT/'runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json';assert fresh_rule.exists()
 data=json.loads(fresh_rule.read_text());assert data['expected_completed_models']==108 and data['expected_actual_task_witnesses']==72
 assert data['code_sha256']['review_large_lastblock_delta_lora_fresh_outputs.py']==sha(ROOT/'review_large_lastblock_delta_lora_fresh_outputs.py')
 dump(OUT/'protocol.json',protocol);dump(OUT/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 with (ROOT/'large_lastblock_delta_lora.log').open('x') as log:
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 dump(OUT/'launch_record.json',{'pid':process.pid,'protocol_sha256':sha(OUT/'protocol.json'),'actual_success_preview_matched':True})
 path=ROOT/'results/next_large_lastblock_delta_lora_hypothesis.json';hypothesis=json.loads(path.read_text())
 dump(path,{**hypothesis,'status':'running','training_launched':True,'pid':process.pid,'protocol':str((OUT/'protocol.json').relative_to(ROOT)),'protocol_sha256':sha(OUT/'protocol.json')})
 print(json.dumps({'pid':process.pid,'planned_new_seed_fits':24,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)
if __name__=='__main__':main()

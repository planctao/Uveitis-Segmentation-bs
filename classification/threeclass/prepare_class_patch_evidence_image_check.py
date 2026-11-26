"""Freeze original-image engineering checks before any new inference is executed."""
import ast,json,hashlib,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CACHE={}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in CACHE:
  with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return CACHE[key]
def read(n):return json.loads((ROOT/n).read_text())
def main():
 out=ROOT/'runs/class_patch_evidence_image_inference_check';assert not (out/'protocol.json').exists()
 previous=read('runs/research_patch_image_inference_check/protocol.json');single=read('runs/research_patch_single_image_check/protocol.json')
 assert read('results/research_patch_image_inference_check.json')['passed'] and read('results/research_patch_single_image_check.json')['passed']
 assert read('runs/class_query_patch_evidence/state.json')['status']=='completed'
 audit=read('results/research_class_query_patch_evidence/verification.json');assert audit['passed'] and audit['actual_seed_heads_checked']==48
 checkpoints=[{'arm':'class_query_pool','phase':'validation','seed':s} for s in range(3)]+[{'arm':arm,'phase':'validation','seed':1} for arm in ['uniform_pool','source_query_pool','class_query_contrast']]
 artifacts={**previous['frozen_artifacts'],**single['frozen_artifacts']};images={**previous['image_sha256'],**single['image_sha256']}
 for section in [artifacts,images]:
  for name,expected in section.items():assert sha(name)==expected,name
 for n in ['runs/research_patch_image_inference_check/protocol.json','runs/research_patch_single_image_check/protocol.json',
  'results/research_patch_image_inference_check.json','results/research_patch_single_image_check.json',
  'runs/class_query_patch_evidence/protocol.json','runs/class_query_patch_evidence/state.json',
  'results/research_class_query_patch_evidence/verification.json','results/research_class_query_patch_evidence/independent_saved_output_review.json',
  'results/research_class_query_patch_evidence/selection_before_repeat.json','verification_class_query_patch_evidence_model.json']:
  artifacts[str(ROOT/n)]=sha(n)
 for case in checkpoints:
  matches=[r for r in audit['fits'] if all(r[k]==v for k,v in case.items())];assert len(matches)==1
  path=ROOT/'runs/class_query_patch_evidence'/case['arm']/case['phase']/f"seed_{case['seed']}"/'model.pt'
  assert sha(path)==matches[0]['checkpoint_sha256'];artifacts[str(path)]=sha(path)
 explicit={'research_class_patch_evidence_image_inference.py','verify_class_patch_evidence_image_inference.py','prepare_class_patch_evidence_image_check.py'}
 code=set(previous['code_sha256'])|set(single['code_sha256'])|explicit
 pending=list(code);excluded={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
 while pending:
  name=pending.pop()
  for node in ast.walk(ast.parse((ROOT/name).read_text())):
   mods=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
   for mod in mods:
    dep=mod+'.py'
    if dep not in code and dep not in excluded and (ROOT/dep).exists():code.add(dep);pending.append(dep)
 assert not (code&excluded)
 protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha('metadata/manifest.csv'),
  'purpose':'Actual raw-image direct-patch classifier numerical engineering; no accuracy, ranking, new fit or reference promotion.',
  'development_batches':previous['development_batches'],'validation_indices':single['validation_indices'],'checkpoint_cases':checkpoints,
  'feature_tolerance':{'atol':.05,'rtol':.005},'matched_batch_probability_tolerance':{'atol':5e-4,'rtol':.002},'single_image_probability_tolerance':{'atol':.005,'rtol':.01},
  'decision_gate':'Every API cache case bitwise original engine; independent formulas2e-6. Raw-vs-cache probabilities use inherited numerical tolerances; every decision identical. Six arrays exact only at original8-image batches. Do not tune gates after outcomes.',
  'planned_actual_backbone_view_forwards':24,'planned_production_head_batch_forwards':90,'planned_independent_formula_batch_forwards':18,'planned_head_image_evaluations':612,
  'classification_fits':0,'new_candidate_scores':0,'external_images_read':0,'code_sha256':{n:sha(n) for n in sorted(code)},'frozen_artifacts':artifacts,'image_sha256':images}
 out.mkdir(exist_ok=True,parents=True)
 with (out/'protocol.json').open('x') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
 env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
 with (ROOT/'class_patch_evidence_image_inference_check.log').open('a') as log:
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/'verify_class_patch_evidence_image_inference.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 with (out/'launch_record.json').open('x') as f:json.dump({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json')},f)
 print(json.dumps({'engineering_worker_pid':process.pid,'code_modules':len(code),'frozen_artifacts':len(artifacts),'image_files':len(images),'protocol_sha256':sha(out/'protocol.json')}),flush=True)
if __name__=='__main__':main()

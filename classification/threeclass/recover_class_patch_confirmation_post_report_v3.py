"""Preserve a completed126fit scientific result whose final mutable reporting refresh failed."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CACHE={}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in CACHE:
  with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return CACHE[key]
def read(name):return json.loads((ROOT/name).read_text())
def main():
 dest=ROOT/'results/class_patch_confirmation_post_report_recovery_v3.json';assert not dest.exists()
 for pid in [14745,14661,15237]:assert not Path(f'/proc/{pid}/cmdline').exists(),pid
 formal=read('runs/class_query_patch_evidence_confirmation/protocol.json')
 assert sha('runs/class_query_patch_evidence_confirmation/protocol.json')=='aa3b38617b7b312620324aa039c01ceae8fde8d0cc5d2021d3a25f27931ccfe0'
 audit=read('results/research_class_query_patch_evidence_confirmation/verification.json');comparison=read('results/research_class_query_patch_evidence_confirmation/comparison.json');freeze=read('results/research_class_query_patch_evidence_confirmation/freeze_integrity_after_confirmation.json')
 assert audit['passed'] and audit['actual_seed_heads_checked']==len(audit['fits'])==126 and audit['actual_training_witnesses_checked']==90 and audit['generative_fits_checked']==18
 assert audit['matched_controls_required']==['matched_uniform','matched_source_query']
 assert comparison['protocol']==formal and comparison['new_seed_fits']==126 and not comparison['confirmation_gate_passed']
 assert freeze['passed'] and freeze['all_source_and_code_hashes_unchanged']
 failed=read('runs/class_query_patch_evidence_confirmation/state.json')
 assert failed['status']=='failed' and 'ImportError' in failed['error'] and 'load_class_patch_confirmation_recovery' in failed['error']
 code={};inputs={}
 for name in ['runs/class_query_patch_evidence_confirmation/protocol.json','runs/class_patch_evidence_confirmation_recovery_v2/protocol.json','runs/dispersion_after_direct_patch_v3/protocol.json']:
  record=read(name);inputs[str(ROOT/name)]=sha(name)
  for section in ['code_sha256','frozen_artifacts','split_plan_sha256']:
   for path,expected in record.get(section,{}).items():
    assert sha(path)==expected,path
    if section=='code_sha256':code[path]=expected
    else:
     p=Path(path);inputs[str(p if p.is_absolute() else ROOT/p)]=expected
 products=read('runs/class_query_patch_evidence_confirmation/training_products_freeze_before_audit.json')['input_artifact_sha256']
 assert len(products)==684
 for path,expected in products.items():assert sha(path)==expected;inputs[path]=expected
 witnesses=0
 for fit in audit['fits']:
  assert sha(fit['checkpoint'])==fit['checkpoint_sha256'];inputs[fit['checkpoint']]=fit['checkpoint_sha256']
  if 'training_witness_path' in fit:
   assert sha(fit['training_witness_path'])==fit['training_witness_sha256'];inputs[fit['training_witness_path']]=fit['training_witness_sha256'];witnesses+=1
  if 'source_checkpoint' in fit:assert sha(fit['source_checkpoint'])==fit['source_checkpoint_sha256']
 # Some source L witnesses are in checkpoint-adjacent records; the completed actual audit declares90.
 assert witnesses>=54
 for path,expected in freeze['geometry_sha256'].items():assert sha(path)==expected;inputs[path]=expected
 names=['runs/class_query_patch_evidence_confirmation/state.json','runs/class_patch_evidence_confirmation_recovery_v2/state.json','runs/dispersion_after_direct_patch_v3/state.json',
  'class_query_patch_evidence_confirmation.log','class_patch_evidence_confirmation_recovery_v2.log','dispersion_after_direct_patch_v3.log',
  'results/research_class_query_patch_evidence_confirmation/comparison.json','results/research_class_query_patch_evidence_confirmation/verification.json',
  'results/research_class_query_patch_evidence_confirmation/freeze_integrity_after_confirmation.json','results/research_class_query_patch_evidence_confirmation/geometry_diagnostics.json',
  'runs/class_query_patch_evidence_confirmation/training_products_freeze_before_audit.json']
 for n in names:inputs[str(ROOT/n)]=sha(n)
 rows={r['arm']:r for r in comparison['arms']};paired=comparison['paired_candidate_vs_controls']
 controls=['m8','m7','m6','reference','fixed_gaussian','matched_uniform_control','matched_source_query_control']
 assert set(paired)==set(controls)
 gate=all(rows['candidate']['mean_macro_f1']>rows[n]['mean_macro_f1'] and paired[n]['macro_f1_ci95'][0]>0 for n in controls)
 assert gate==comparison['confirmation_gate_passed']==False
 registry=read('results/research_reference_registry.json');assert sha(registry['active_reference'])==registry['active_reference_sha256']
 assert read(registry['active_reference'])['retained_development_model']=='M8'
 record={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'finding':'All126fits,90actualtaskwitnesses,18Gaussianfits, frozen products and full independent confirmation audit completed and decision saved. Only subsequent mutable reporting failed because a long-lived interpreter cached integrity_reporting before new reporting helpers were added. Original failed worker/observer/bridge states and logs preserved; no restart, fit, prediction or scientific gate change.',
  'formal_protocol_sha256':sha('runs/class_query_patch_evidence_confirmation/protocol.json'),
  'recovered_completed_confirmation_state':{'status':'completed','new_seed_fits':126,'generative_fits':18,'confirmation_gate_passed':False,'external_test_evaluated':False,'original_failed_state_preserved':True},
  'recovered_continuation_state':{'status':'completed_class_patch_evidence_decision_requires_next_exploration','reference_changed':False,'external_test_evaluated':False},
  'actual_seed_heads_checked':126,'actual_training_witnesses_checked':90,'generative_fits_checked':18,'training_products_checked':684,
  'actual_checkpoint_and_source_records_checked':126,'candidate_mean_macro_f1':rows['candidate']['mean_macro_f1'],'matched_m8_mean_macro_f1':rows['m8']['mean_macro_f1'],
  'paired_vs_m8':paired['m8'],'confirmation_gate_passed':False,'reference_changed':False,
  'code_sha256':{**dict(sorted(code.items())),'recover_class_patch_confirmation_post_report_v3.py':sha('recover_class_patch_confirmation_post_report_v3.py')},
  'frozen_artifacts':dict(sorted(inputs.items())),'classification_fits':0,'classification_scores':0,'model_predictions':0,'external_images_read':0,
  'scope':'Completed scientific-output/frozen-byte and saved-gate recovery; no model/task replay. Distinguishes failed reporting execution from completed negative scientific result. Original failure snapshots remain immutable.'}
 with dest.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
 print(json.dumps({'passed':True,'record_sha256':sha(dest),'code_modules':len(record['code_sha256']),'frozen_artifacts':len(inputs),'candidate_mean_macro_f1':rows['candidate']['mean_macro_f1'],'m8_mean_macro_f1':rows['m8']['mean_macro_f1'],'confirmation_gate_passed':False}),flush=True)
if __name__=='__main__':main()

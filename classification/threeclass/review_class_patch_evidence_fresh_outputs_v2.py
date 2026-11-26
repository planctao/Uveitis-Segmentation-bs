"""Independent saved-output reconstruction and descriptive corrections after negative fresh confirmation."""
import argparse,fcntl,hashlib,json,os,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/class_query_patch_evidence_confirmation'
RESULT=ROOT/'results/research_class_query_patch_evidence_confirmation'
OUT=ROOT/'runs/class_patch_fresh_saved_review_v2'
DEST=RESULT/'independent_saved_output_review.json'
CACHE={}
def sha(name):
 p=Path(name);p=p if p.is_absolute() else ROOT/p;p=p.resolve(strict=True);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in CACHE:
  with p.open('rb') as f:CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return CACHE[key]
def read(name):return json.loads(Path(name).read_text())
def write(name,data):
 p=Path(name);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def status(value,**details):
 p=OUT/'state.json';p.write_text(json.dumps({'status':value,'pid':os.getpid(),**details,'external_test_evaluated':False}))
def prepare():
 assert not (OUT/'protocol.json').exists() and not DEST.exists()
 formal=read(RUN/'protocol.json');comparison=read(RESULT/'comparison.json');audit=read(RESULT/'verification.json')
 assert audit['passed'] and audit['actual_seed_heads_checked']==126 and audit['actual_training_witnesses_checked']==90
 assert not comparison['confirmation_gate_passed'] and comparison['protocol']==formal
 inputs={}
 def add(p,expected=None):
  digest=sha(p)
  if expected is not None:assert digest==expected,str(p)
  inputs[str(Path(p).resolve())]=digest
 for p in [ROOT/'results/class_patch_fresh_saved_review_metric_recovery_v2.json',ROOT/'review_class_patch_evidence_fresh_outputs.py',ROOT/'runs/class_patch_fresh_saved_review/protocol.json',ROOT/'runs/class_patch_fresh_saved_review/state.json',ROOT/'experiment.py',ROOT/'metadata/manifest.csv',RUN/'protocol.json',RUN/'state.json',RESULT/'comparison.json',RESULT/'verification.json',RESULT/'freeze_integrity_after_confirmation.json',ROOT/'results/class_patch_confirmation_post_report_recovery_v3.json',RUN/'training_products_freeze_before_audit.json']:
  add(p)
 assert sha(ROOT/'results/class_patch_confirmation_post_report_recovery_v3.json')=='4fd35859b88917466202600c9e4a2f14050681f79d7c61e9eb0308e72d52399a'
 for p,expected in formal['split_plan_sha256'].items():add(p,expected)
 products=read(RUN/'training_products_freeze_before_audit.json')['input_artifact_sha256'];assert len(products)==684
 for p,expected in products.items():add(p,expected)
 for arm in formal['arms']:
  for repeat in formal['outer_split_seeds']:
   for fold in range(3):
    folder=RUN/arm/f'repeat_{repeat}/fold_{fold}'
    for name in ['outer_predictions.npz','metrics.json']:add(folder/name)
 for item in audit['fits']:
  add(item['checkpoint'],item['checkpoint_sha256'])
  if 'training_witness_path' in item:add(item['training_witness_path'],item['training_witness_sha256'])
 for arm in ['value','matched_uniform','matched_source_query']:
  for repeat in formal['outer_split_seeds']:
   for fold in range(3):
    for seed in range(3):add(RUN/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}/summary.json')
 rule={'created_at_utc':datetime.now(timezone.utc).isoformat(),'family':'class_query_patch_evidence_confirmation',
  'outcome_known_before_review_rule':True,'known_confirmation_gate_passed':False,
  'scope':'Post-outcome independent saved-output/metric/fixed-fusion/paired-CI reconstruction and descriptive correction accounting. No model/task replay, new predictions, new architecture, hyperparameter selection or external data. Does not establish causal error mechanisms.',
  'expected_saved_fold_metric_sets':84,'expected_fixed_fusion_checks':36,'expected_paired_comparisons':7,'expected_patient_plans':6,
  'expected_completed_models':126,'expected_actual_task_witnesses':90,'expected_training_products':684,'expected_adapter_summaries':54,
  'fixed_fusions':{'m6':{'reference':.25,'subspace':.75},'m7':{'reference':.1875,'subspace':.5625,'large':.25},'m8':{'reference':.1875,'subspace':.5625,'token':.25},
   'candidate':{'reference':.1875,'subspace':.5625,'value':.25},'matched_uniform_control':{'reference':.1875,'subspace':.5625,'matched_uniform':.25},'matched_source_query_control':{'reference':.1875,'subspace':.5625,'matched_source_query':.25}},
  'paired_seed':20270124,'paired_bootstrap_draws':2000,'formula_tolerance':{'atol':2e-7,'rtol':2e-7},'metric_tolerance':1e-12,
  'correction_controls':['m8','matched_uniform_control','matched_source_query_control'],
  'code_sha256':{'review_class_patch_evidence_fresh_outputs_v2.py':sha(__file__)},'frozen_artifacts':dict(sorted(inputs.items()))}
 write(OUT/'protocol.json',rule);print(json.dumps({'frozen_saved_review':True,'inputs':len(inputs),'known_confirmation_negative':True}),flush=True)
def check_frozen(rule):
 for name,expected in rule['code_sha256'].items():assert sha(ROOT/name)==expected,name
 for name,expected in rule['frozen_artifacts'].items():assert sha(name)==expected,name
def cm_for(y,p):
 result=np.zeros((3,3),dtype='int64')
 for truth,guess in zip(y,p):result[int(truth),int(guess)]+=1
 return result
def scores(cm):
 diagonal=np.diagonal(cm,axis1=-2,axis2=-1).astype(float);denominator=cm.sum(-1)+cm.sum(-2)
 f1=np.divide(2*diagonal,denominator,out=np.zeros_like(diagonal),where=denominator>0).mean(-1)
 return diagonal.sum(-1)/cm.sum((-2,-1)),f1
def independent_metrics(y,p):
 cm=cm_for(y,p.argmax(1));acc,f1=scores(cm);recall=np.diag(cm)/cm.sum(1)
 normalized=np.clip(np.asarray(p,dtype='float64'),1e-7,1);normalized=normalized/normalized.sum(1,keepdims=True)
 loss=-np.log(normalized[np.arange(len(y)),np.asarray(y,dtype=int)]).mean()
 return {'n':len(y),'accuracy':float(acc),'balanced_accuracy':float(recall.mean()),'macro_f1':float(f1),
  'log_loss':float(loss),'confusion_matrix':cm.tolist(),'recall':dict(zip(['infectious','noninfectious','normal'],map(float,recall)))}
def compare_metrics(a,b,tolerance):
 assert a.keys()==b.keys()
 for key in a:
  if key in ['n','confusion_matrix']:assert a[key]==b[key],key
  elif key=='recall':
   for label in a[key]:assert abs(a[key][label]-b[key][label])<=tolerance,(key,label)
  else:assert abs(a[key]-b[key])<=tolerance,(key,a[key],b[key])
def review():
 started=time.time();assert not DEST.exists();rule=read(OUT/'protocol.json');check_frozen(rule);status('reviewing_saved_outputs')
 formal=read(RUN/'protocol.json');comparison=read(RESULT/'comparison.json');audit=read(RESULT/'verification.json')
 frame=pd.read_csv(ROOT/'metadata/manifest.csv');original=np.flatnonzero(frame.split.eq('train'));assert len(original)==1717
 labels=frame.iloc[original].label.to_numpy();original_set=set(original.tolist());row_map={r['arm']:r for r in comparison['arms']}
 arms=formal['arms'];repeats=formal['outer_split_seeds'];assert len(arms)==14 and len(repeats)==2
 bank={arm:{} for arm in arms};metric_rows=[];fusion_rows=[];plans_checked=0
 for repeat in repeats:
  seen=[]
  for fold in range(3):
   plan=read(RUN/'split_plans'/f'repeat_{repeat}_fold_{fold}.json')
   partitions=[np.asarray(plan[k],dtype='int64') for k in ['fit_indices','stop_indices','outer_indices']]
   assert set(np.concatenate(partitions).tolist())==original_set and sum(map(len,partitions))==len(original)
   for i in range(3):
    for j in range(i):
     assert not set(partitions[i]).intersection(partitions[j])
     for key in ['patient_group','split_group']:assert not set(frame.iloc[partitions[i]][key]).intersection(frame.iloc[partitions[j]][key]),key
   plans_checked+=1;outer=partitions[2];seen.extend(outer.tolist());probabilities={}
   for arm in arms:
    folder=RUN/arm/f'repeat_{repeat}/fold_{fold}';saved=np.load(folder/'outer_predictions.npz');ids=saved['indices'];p=saved['p']
    assert np.array_equal(ids,outer) and np.array_equal(saved['pred'],p.argmax(1))
    assert p.shape==(len(ids),3) and np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1,atol=2e-6,rtol=0)
    computed=independent_metrics(frame.iloc[ids].label.to_numpy(),p);existing=read(folder/'metrics.json')
    assert existing['repeat']==repeat and existing['fold']==fold;compare_metrics(computed,existing['metrics'],rule['metric_tolerance'])
    matching=[r for r in row_map[arm]['folds'] if r['repeat']==repeat and r['fold']==fold];assert len(matching)==1
    compare_metrics(computed,matching[0]['metrics'],rule['metric_tolerance'])
    metric_rows.append({'arm':arm,'repeat':repeat,'fold':fold,'metrics':computed});probabilities[arm]=p
    bank[arm].setdefault(repeat,[]).append((ids.copy(),p.copy()))
   for name,weights in rule['fixed_fusions'].items():
    manual=sum(np.asarray(probabilities[source],dtype='float64')*weight for source,weight in weights.items()).astype('float32')
    assert np.allclose(manual,probabilities[name],**rule['formula_tolerance']) and np.array_equal(manual.argmax(1),probabilities[name].argmax(1)),(name,repeat,fold)
    fusion_rows.append({'arm':name,'repeat':repeat,'fold':fold,'maximum_probability_error':float(np.abs(manual-probabilities[name]).max())})
  assert sorted(seen)==original.tolist()
  for arm in arms:
   parts=bank[arm][repeat];ids=np.concatenate([x[0] for x in parts]);p=np.concatenate([x[1] for x in parts]);order=ids.argsort();assert np.array_equal(ids[order],original);bank[arm][repeat]=p[order]
 for arm in arms:
  values=[r['metrics'] for r in metric_rows if r['arm']==arm]
  assert abs(np.mean([r['macro_f1'] for r in values])-row_map[arm]['mean_macro_f1'])<=1e-12
  assert abs(np.std([r['macro_f1'] for r in values],ddof=1)-row_map[arm]['sd_macro_f1'])<=1e-12
  assert abs(np.mean([r['accuracy'] for r in values])-row_map[arm]['mean_accuracy'])<=1e-12
 _,groups=np.unique(frame.iloc[original].split_group.to_numpy(),return_inverse=True);group_count=int(groups.max()+1)
 confusions={}
 for arm in arms:
  cm=np.zeros((group_count,3,3),dtype='int64')
  for repeat in repeats:
   for group,truth,guess in zip(groups,labels,bank[arm][repeat].argmax(1)):cm[group,truth,guess]+=1
  confusions[arm]=cm
 draws=np.random.default_rng(rule['paired_seed']).integers(0,group_count,size=(rule['paired_bootstrap_draws'],group_count))
 pairs={}
 for control,existing in comparison['paired_candidate_vs_controls'].items():
  a,b=confusions[control],confusions['candidate'];a_acc,a_f1=scores(a.sum(0));b_acc,b_f1=scores(b.sum(0))
  aa,af=scores(np.stack([a[draw].sum(0) for draw in draws]));ba,bf=scores(np.stack([b[draw].sum(0) for draw in draws]))
  computed={'accuracy_difference':float(b_acc-a_acc),'macro_f1_difference':float(b_f1-a_f1),
   'accuracy_ci95':np.quantile(ba-aa,[.025,.975]).tolist(),'macro_f1_ci95':np.quantile(bf-af,[.025,.975]).tolist()}
  for key in computed:assert np.allclose(computed[key],existing[key],atol=1e-12,rtol=0),(control,key)
  pairs[control]=computed
 gate=all(row_map['candidate']['mean_macro_f1']>row_map[name]['mean_macro_f1'] and pair['macro_f1_ci95'][0]>0 for name,pair in pairs.items())
 assert gate==comparison['confirmation_gate_passed']==False
 correction_rows=[];case_rows=[]
 for repeat in repeats:
  candidate=bank['candidate'][repeat];guess=candidate.argmax(1);correct=guess==labels
  for control in rule['correction_controls']:
   reference=bank[control][repeat];refguess=reference.argmax(1);refcorrect=refguess==labels;changed=guess!=refguess
   fixed=(~refcorrect)&correct;introduced=refcorrect&(~correct)
   correction_rows.append({'repeat':repeat,'control':control,'images':1717,'decision_changes':int(changed.sum()),
    'errors_corrected':int(fixed.sum()),'errors_introduced':int(introduced.sum()),'net_correct_change':int(fixed.sum()-introduced.sum()),
    'changed_patients':int(frame.iloc[original[changed]].patient_group.nunique()),
    'normal_boundary_changes':int((changed&((guess==2)|(refguess==2))).sum()),
    'infectious_noninfectious_changes':int((changed&(guess!=2)&(refguess!=2)).sum()),
    'maximum_probability_change':float(np.abs(candidate-reference).max()),'mean_absolute_probability_change':float(np.abs(candidate-reference).mean())})
  for i,index in enumerate(original):
   row={'repeat':repeat,'manifest_index':int(index),'patient_group':frame.iloc[index].patient_group,'split_group':frame.iloc[index].split_group,'label':int(labels[i]),'candidate_pred':int(guess[i])}
   for control in rule['correction_controls']:row[control+'_pred']=int(bank[control][repeat][i].argmax())
   case_rows.append(row)
 summaries={}
 for arm in ['value','matched_uniform','matched_source_query']:
  rows=[]
  for repeat in repeats:
   for fold in range(3):
    for seed in range(3):
     s=read(RUN/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}/summary.json');rows.append({'repeat':repeat,'fold':fold,'seed':seed,'selected_epoch':s['epoch'],'trainable_parameters':s['trainable_parameters']})
  summaries[arm]={'heads':18,'selected_epoch_zero_heads':sum(r['selected_epoch']==0 for r in rows),'rows':rows}
 assert len(metric_rows)==84 and len(fusion_rows)==36 and len(pairs)==7 and plans_checked==6 and len(case_rows)==3434
 table=RESULT/'fresh_training_oof_corrections.csv';assert not table.exists();pd.DataFrame(case_rows).to_csv(table,index=False)
 check_frozen(rule)
 result={'review_passed':True,'family':'class_query_patch_evidence_confirmation','created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),
  'saved_fold_metric_sets_recomputed':84,'fixed_fusion_phases_recomputed':36,'paired_patient_comparisons_reconstructed':7,'patient_and_duplicate_group_plans_checked':6,
  'completed_classification_fits':126,'completed_actual_task_witnesses':90,'training_products_checked':684,'checkpoint_records_checked':126,
  'confirmation_gate_independently_checked':True,'confirmation_gate_passed':False,'rows':metric_rows,'fusion_checks':fusion_rows,'paired_comparisons':pairs,
  'descriptive_correction_rows':correction_rows,'adapter_selected_epoch_summaries':summaries,'training_images_per_repeat':1717,'repeats':repeats,
  'case_table':str(table.relative_to(ROOT)),'case_table_sha256':sha(table),'case_rows':3434,
  'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_images_read':0,'external_test_evaluated':False,
  'input_sha256':rule['frozen_artifacts'],'code_sha256':rule['code_sha256'],'outcome_known_before_review_rule':True,
  'scope_limit':'Recomputes84already-scoredsavedmetricsets,36fixedfusions and7previouslydeclaredpairedstatistics, not new candidate evaluations. Uses exactly1717originaltrainimages twice;3434rows are repeated observations, not independent images/patients. Per-seed averaging and Gaussian internal predictions are not replayed here; actual126model/90task semantics rely on completed original audit. Selected-epoch and correction counts are post-outcome descriptive, not causal evidence or a new confirmation.',
  'seconds':time.time()-started}
 write(DEST,result);status('completed',review_passed=True)
 print(json.dumps({'review_passed':True,'saved_metric_sets':84,'fixed_fusions':36,'paired_comparisons':7,'confirmation_gate_passed':False,'corrections':correction_rows,'selected_epoch_zero_heads':{a:s['selected_epoch_zero_heads'] for a,s in summaries.items()}}),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 if args.prepare:prepare();return
 with (OUT/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:review()
  except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()

"""Independent saved-output reconstruction and descriptive corrections after completed fresh confirmation, using a rule frozen before its outcomes."""
import argparse,fcntl,hashlib,json,os,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/large_lastblock_readout_coupling_confirmation'
RESULT=ROOT/'results/research_large_lastblock_readout_coupling_confirmation'
OUT=ROOT/'runs/large_lastblock_readout_coupling_fresh_saved_review'
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
def prepare_rule():
 assert not (OUT/'rule_before_outcome.json').exists()
 assert not (RESULT/'comparison.json').exists(),'Freeze rule before fresh confirmation outcome.'
 prereg=ROOT/'results/large_lastblock_readout_coupling_preregistration.json'
 assert sha(prereg)=='180e3bbd1f9fbbf3e70d428c117d4c6c5950ef1c510d0fda0eff5d67266d19e0'
 rule={'created_at_utc':datetime.now(timezone.utc).isoformat(),'family':'large_lastblock_readout_coupling_confirmation',
  'outcome_known_before_review_rule':False,'expected_saved_fold_metric_sets':84,'expected_fixed_fusion_checks':36,
  'expected_paired_comparisons':7,'expected_patient_plans':6,'expected_completed_models':126,'expected_actual_task_witnesses':90,
  'expected_training_products':684,'expected_adapter_summaries':54,'preregistration_sha256':sha(prereg),
  'candidate_replacement_weights_allowed':[.25,1.],'matched_controls':['matched_token_joint','matched_readout_only'],
  'paired_seed':20270124,'paired_bootstrap_draws':2000,'formula_tolerance':{'atol':2e-7,'rtol':2e-7},'metric_tolerance':1e-12,
  'correction_controls':['m8','matched_token_joint','matched_readout_only'],
  'code_sha256':{'review_large_lastblock_readout_coupling_fresh_outputs.py':sha(__file__)},
  'scope':'Freeze saved-output/metric/fusion/paired-CI and descriptive correction review before fresh outcomes. Bind all completed inputs only after126model/90task audit. No new model predictions or candidate evaluations. Positive or negative outcomes handled identically.'}
 write(OUT/'rule_before_outcome.json',rule)
 print(json.dumps({'pre_outcome_review_rule_frozen':True,'rule_sha256':sha(OUT/'rule_before_outcome.json')}),flush=True)
def prepare():
 assert not (OUT/'protocol.json').exists() and not DEST.exists()
 rule=read(OUT/'rule_before_outcome.json')
 for name,expected in rule['code_sha256'].items():assert sha(ROOT/name)==expected,name
 state=read(RUN/'state.json');assert state['status']=='completed' and state['new_seed_fits']==126
 formal=read(RUN/'protocol.json');comparison=read(RESULT/'comparison.json');audit=read(RESULT/'verification.json')
 assert audit['passed'] and audit['actual_seed_heads_checked']==126 and audit['actual_training_witnesses_checked']==90
 assert audit['matched_controls_required']==['token_joint','readout_only'] and comparison['protocol']==formal
 assert formal['planned_new_seed_fits']==126 and formal['planned_training_witnesses']==90
 weight=formal['selected_configuration']['replacement_weight'];assert weight in rule['candidate_replacement_weights_allowed']
 assert formal['outer_split_seeds']==[20294701,20294711]
 inputs={}
 def add(p,expected=None):
  digest=sha(p)
  if expected is not None:assert digest==expected,str(p)
  inputs[str(Path(p).resolve())]=digest
 for p in [OUT/'rule_before_outcome.json',ROOT/'experiment.py',ROOT/'metadata/manifest.csv',RUN/'protocol.json',RUN/'state.json',RESULT/'comparison.json',RESULT/'verification.json',RESULT/'freeze_integrity_after_confirmation.json',RUN/'training_products_freeze_before_audit.json',RUN/'ram_activation_cache_before_any_fit.json']:
  add(p)
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
 for arm in ['attention_joint','token_joint','readout_only']:
  for repeat in formal['outer_split_seeds']:
   for fold in range(3):
    for seed in range(3):add(RUN/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}/summary.json')
 fusions={'m6':{'reference':.25,'subspace':.75},'m7':{'reference':.1875,'subspace':.5625,'large':.25},'m8':{'reference':.1875,'subspace':.5625,'token':.25}}
 for name,arm in [('candidate','attention_joint'),('matched_token_joint','token_joint'),('matched_readout_only','readout_only')]:
  fusions[name]={'reference':.1875,'subspace':.5625,arm:.25} if weight==.25 else {arm:1.}
 write(OUT/'protocol.json',{**rule,'inputs_bound_at_utc':datetime.now(timezone.utc).isoformat(),'rule_before_outcome_sha256':sha(OUT/'rule_before_outcome.json'),'fixed_fusions':fusions,'frozen_artifacts':dict(sorted(inputs.items()))})
 print(json.dumps({'frozen_saved_review_inputs':True,'inputs':len(inputs),'outcome_rule_changed':False}),flush=True)
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
 cache=read(RUN/'ram_activation_cache_before_any_fit.json');assert cache['mode']=='fresh' and cache['classification_fits_so_far']==0 and cache['metadata']['indices']==original.tolist()
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
 assert set(pairs)=={'m8','m7','m6','reference','fixed_gaussian','matched_token_joint','matched_readout_only'}
 assert gate==comparison['confirmation_gate_passed']
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
 compact_checked=0
 for item in audit['fits']:
  if item['arm'] not in ['attention_joint','token_joint','readout_only']:continue
  arm=item['arm'];folder=Path(item['checkpoint']).parent
  checkpoint=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
  source_path=Path(checkpoint['source_checkpoint'])
  assert sha(source_path)==checkpoint['source_checkpoint_sha256']==item['source_checkpoint_sha256']
  source=torch.load(source_path,map_location='cpu',weights_only=False)
  plan=read(RUN/'split_plans'/f"repeat_{item['repeat']}_fold_{item['fold']}.json")
  fit=np.asarray(plan['fit_indices'],dtype='int64');assert np.array_equal(checkpoint['fit_indices'],fit)
  assert checkpoint['arm']==arm and checkpoint['seed']==item['seed']
  expected_source=RUN/'sources/token'/f"repeat_{item['repeat']}/fold_{item['fold']}/seed_{item['seed']}/model.pt"
  assert source_path==expected_source
  names=['head.flat.weight','head.flat.bias']
  if arm!='readout_only':names=['adapter_a','adapter_b']+names
  values=checkpoint['trainable_state_dict'];leaves=len(source['diseases'])
  count=257*leaves+(0 if arm=='readout_only' else 16384)
  assert list(values)==names and sum(value.numel() for value in values.values())==count
  assert values['head.flat.weight'].shape==(leaves,256) and values['head.flat.bias'].shape==(leaves,)
  assert all(torch.isfinite(value).all() for value in values.values())
  warm=read(folder/'warmstart_freeze_before_adapter.json')
  assert warm['arm']==arm and warm['seed']==item['seed'] and warm['source_checkpoint_sha256']==sha(source_path)
  assert warm['fit_indices']==fit.tolist() and warm['actual_images']==256
  with np.load(folder/'training_input_witness.npz') as witness:
   ids=fit[np.random.default_rng(20294600+item['seed']).permutation(len(fit))[:256]]
   assert np.array_equal(witness['indices'],ids) and np.array_equal(witness['fit_indices'],fit)
   assert str(witness['arm'])==arm and int(witness['seed'])==item['seed']
   assert witness['parameter_names'].tolist()==names and witness['gradient'].shape==(count,)
   assert np.isfinite(witness['gradient']).all() and np.isfinite(witness['loss'])
   for name in ['head.flat.weight','head.flat.bias']:
    initial=source['state_dict'][name.removeprefix('head.')].detach().cpu().numpy()
    assert str(witness['initial_sha256_'+name])==hashlib.sha256(initial.tobytes()).hexdigest()
  assert item['witness']['only_prescribed_adapter_and_readout_trainable']
  compact_checked+=1
 assert compact_checked==54
 summaries={}
 for arm in ['attention_joint','token_joint','readout_only']:
  rows=[]
  for repeat in repeats:
   for fold in range(3):
    for seed in range(3):
     s=read(RUN/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}/summary.json');rows.append({'repeat':repeat,'fold':fold,'seed':seed,'selected_epoch':s['epoch'],'trainable_parameters':s['trainable_parameters']})
  summaries[arm]={'heads':18,'selected_epoch_zero_heads':sum(r['selected_epoch']==0 for r in rows),'rows':rows}
 assert len(metric_rows)==84 and len(fusion_rows)==36 and len(pairs)==7 and plans_checked==6 and len(case_rows)==3434
 table=RESULT/'fresh_training_oof_corrections.csv';assert not table.exists();pd.DataFrame(case_rows).to_csv(table,index=False)
 check_frozen(rule)
 result={'review_passed':True,'family':'large_lastblock_readout_coupling_confirmation','created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),
  'saved_fold_metric_sets_recomputed':84,'fixed_fusion_phases_recomputed':36,'paired_patient_comparisons_reconstructed':7,'patient_and_duplicate_group_plans_checked':6,
  'completed_classification_fits':126,'completed_actual_task_witnesses':90,'training_products_checked':684,'checkpoint_records_checked':126,
  'compact_adapter_readout_checkpoints_checked':54,'confirmation_gate_independently_checked':True,'confirmation_gate_passed':gate,'rows':metric_rows,'fusion_checks':fusion_rows,'paired_comparisons':pairs,
  'descriptive_correction_rows':correction_rows,'adapter_selected_epoch_summaries':summaries,'training_images_per_repeat':1717,'repeats':repeats,
  'case_table':str(table.relative_to(ROOT)),'case_table_sha256':sha(table),'case_rows':3434,
  'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_images_read':0,'external_test_evaluated':False,
  'input_sha256':rule['frozen_artifacts'],'code_sha256':rule['code_sha256'],'outcome_known_before_review_rule':False,
  'scope_limit':'Recomputes84already-scoredsavedmetricsets,36fixedformulas and7previouslydeclaredpairedstatistics, not new candidate evaluations. Uses exactly1717originaltrainimages twice;3434rows are repeated observations, not independent images/patients. Per-seed averaging and Gaussian internal predictions are not replayed here; actual126model/90task semantics rely on completed original audit. Selected-epoch and correction counts are post-outcome descriptive, not causal evidence or a new confirmation.',
  'seconds':time.time()-started}
 write(DEST,result);status('completed',review_passed=True)
 print(json.dumps({'review_passed':True,'saved_metric_sets':84,'fixed_fusions':36,'paired_comparisons':7,'confirmation_gate_passed':gate,'corrections':correction_rows,'selected_epoch_zero_heads':{a:s['selected_epoch_zero_heads'] for a,s in summaries.items()}}),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--prepare-rule',action='store_true');args=parser.parse_args()
 if args.prepare_rule:prepare_rule();return
 if args.prepare:prepare();return
 with (OUT/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:review()
  except BaseException:status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()

"""Evaluate the protected multiscale compact candidate after development selection."""
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from experiment import ROOT,load_data,dump
from continuous import known_cv,signature
from evaluate import evaluate_group
from report import paired_ci
from milestone2 import paired_oof

def main():
    torch.set_num_threads(4)
    candidate=json.loads((ROOT/'results/development_round3/candidate.json').read_text());feature=candidate['feature']
    cv_dir=ROOT/'runs'/feature/'candidate_crossvalidation'/('continuous_'+candidate['candidate_id'])
    audit=json.loads((cv_dir/'provenance_audit.json').read_text());assert audit['passed']
    cv=json.loads((cv_dir/'summary.json').read_text())
    previous=json.loads((ROOT/'results/milestone2/selection_before_test.json').read_text())
    registry=known_cv();key=signature(feature,candidate['config'])
    prior=max([previous['cv']['mean_macro_f1']]+[r['mean_macro_f1'] for k,r in registry.items() if k!=key])
    out=ROOT/'results/milestone3';out.mkdir(parents=True,exist_ok=True)
    frozen=out/'selection_before_test.json'
    if frozen.exists():
        decision=json.loads(frozen.read_text());assert decision['selected']['candidate_id']==candidate['candidate_id']
    else:
        decision={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'selected':candidate,'cv':cv,'prior_best_development_cv_macro_f1':prior,'eligible':cv['mean_macro_f1']>prior and candidate['val']['macro_f1']>=previous['selected']['val']['macro_f1'],'rule':'Maximum completed development CV macro-F1 plus nondecreasing development validation macro-F1, frozen before this candidate external evaluation; external results never select the model.','architecture':'Shared frozen DINOv3 ViT-B/16 at 320 and 448; aligned regional features; compact subtype head with probability marginalization to three classes','resolved_subtype_auxiliary_weight':candidate['config'].get('aux',.6),'provenance_audit_passed':True}
        dump(frozen,decision)
    if not decision['eligible']:
        dump(out/'decision.json',{'promoted':False,'reason':'Another completed development CV candidate is stronger; no external evaluation.'});return
    for i,path in enumerate(candidate['checkpoints']):assert hashlib.sha256(open(path,'rb').read()).hexdigest()==candidate['checkpoint_sha256'][str(i)]
    frame,features,digest=load_data(feature);target=out/'primary'
    if not (target/'metrics.json').exists():evaluate_group(candidate,frame,features,digest,target)
    report=json.loads((target/'metrics.json').read_text());assert abs(report['results']['val']['image']['macro_f1']-candidate['val']['macro_f1'])<1e-6
    comparisons={}
    for split in ['test_independent','test_ningbo']:
        comparisons[split]={'versus_direct_mlp':paired_ci(ROOT/'results/vitb_320/baseline/flat'/f'{split}_images.csv',target/f'{split}_images.csv'),'versus_second_primary':paired_ci(ROOT/'results/milestone2/primary'/f'{split}_images.csv',target/f'{split}_images.csv')}
    dump(out/'paired_external_comparison.json',comparisons)
    previous_cv=ROOT/'runs'/previous['selected']['feature']/'candidate_crossvalidation'/('continuous_'+previous['selected']['candidate_id'])
    dump(out/'paired_outer_cv.json',paired_oof(previous_cv,cv_dir,frame))
    print(json.dumps({'selected':candidate['candidate_id'],'mean_cv_macro_f1':cv['mean_macro_f1'],'external':{s:r['image'] for s,r in report['results'].items() if s.startswith('test')}}),flush=True)

if __name__=='__main__':main()

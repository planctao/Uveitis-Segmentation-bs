"""Protect and evaluate the strongest completed development-CV candidate once."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from experiment import ROOT,load_data,train_one,summarize_group,dump
from continuous import known_cv,signature
from evaluate import evaluate_group
from report import paired_ci
from milestone2 import paired_oof

def main():
    torch.set_num_threads(4);out=ROOT/'results/milestone4';out.mkdir(parents=True,exist_ok=True)
    frozen=out/'selection_before_test.json'
    if frozen.exists():decision=json.loads(frozen.read_text());candidate=decision['selected']
    else:
        pool=known_cv();key,winner=max(pool.items(),key=lambda item:item[1]['mean_macro_f1'])
        cv_path=Path(winner['source']);cv=json.loads(cv_path.read_text());identifier=cv['candidate'].removeprefix('continuous_')
        records=json.loads((ROOT/'runs/continuous_search/leaderboard.json').read_text())
        matches=[r for r in records if signature(r['feature'],r['config'])==key]
        if not matches:raise ValueError('CV champion has no matching full-development candidate to protect')
        exact=[r for r in matches if r['candidate_id']==identifier]
        record=exact[0] if exact else max(matches,key=lambda r:r['val']['macro_f1'])
        frame,features,digest=load_data(winner['feature']);fit=out/'protected_model'
        source_available=all(Path(p).exists() for p in record['checkpoints'])
        if source_available:
            for seed,path in enumerate(record['checkpoints']):
                target=fit/f'seed_{seed}';target.mkdir(parents=True,exist_ok=True)
                for name in ['model.pt','summary.json','val_predictions.npz']:shutil.copy2(Path(path).parent/name,target/name)
        else:
            for seed in range(3):train_one(frame,features,digest,{**winner['config'],'seed':seed},fit/f'seed_{seed}')
        candidate=summarize_group(fit,frame)
        candidate.update(feature=winner['feature'],candidate_id=identifier,source_candidate_id=record['candidate_id'],source_cv_summary=str(cv_path),protected_from_pruning=True,full_fit_refitted=not source_available)
        candidate['checkpoint_sha256']={str(seed):hashlib.sha256(Path(path).read_bytes()).hexdigest() for seed,path in enumerate(candidate['checkpoints'])}
        request=out/'candidate.json';dump(request,candidate)
        subprocess.run([sys.executable,str(ROOT/'audit_candidate_cv.py'),'--candidate-json',str(request)],cwd=ROOT,check=True)
        audit_path=cv_path.parent/'provenance_audit.json';assert json.loads(audit_path.read_text())['passed']
        reference=json.loads((ROOT/'results/milestone2/selection_before_test.json').read_text())
        decision={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'rule':'Freeze the maximum completed development outer-CV macro-F1 candidate; require improvement over the last published development CV and nondecreasing development validation macro-F1. No external metric enters this choice.','selected':candidate,'cv':cv,'eligible':cv['mean_macro_f1']>reference['cv']['mean_macro_f1'] and candidate['val']['macro_f1']>=reference['selected']['val']['macro_f1'],'pool_at_selection':pool,'provenance_audit_passed':True,'inference_views':'DINOv3 ViT-B/16 at 320 and 448 for multiscale features; three small-head seeds share both backbone feature passes.'}
        dump(frozen,decision)
    if not decision['eligible']:
        dump(out/'decision.json',{'promoted':False,'reason':'Development criteria failed; no external evaluation.'});return
    for seed,path in enumerate(candidate['checkpoints']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==candidate['checkpoint_sha256'][str(seed)]
    frame,features,digest=load_data(candidate['feature']);target=out/'primary'
    if not (target/'metrics.json').exists():evaluate_group(candidate,frame,features,digest,target)
    metrics=json.loads((target/'metrics.json').read_text());assert abs(metrics['results']['val']['image']['macro_f1']-candidate['val']['macro_f1'])<1e-6
    paired={}
    for split in ['test_independent','test_ningbo']:
        paired[split]={'versus_direct_mlp':paired_ci(ROOT/'results/vitb_320/baseline/flat'/f'{split}_images.csv',target/f'{split}_images.csv'),'versus_second_primary':paired_ci(ROOT/'results/milestone2/primary'/f'{split}_images.csv',target/f'{split}_images.csv')}
    dump(out/'paired_external_comparison.json',paired)
    reference=json.loads((ROOT/'results/milestone2/selection_before_test.json').read_text())
    previous_cv=ROOT/'runs'/reference['selected']['feature']/'candidate_crossvalidation'/('continuous_'+reference['selected']['candidate_id'])
    dump(out/'paired_outer_cv.json',paired_oof(previous_cv,Path(candidate['source_cv_summary']).parent,frame))
    print(json.dumps({'selected':candidate['candidate_id'],'feature':candidate['feature'],'val':candidate['val'],'cv_macro_f1':decision['cv']['mean_macro_f1'],'external':{s:r['image'] for s,r in metrics['results'].items() if s.startswith('test')}}),flush=True)

if __name__=='__main__':main()

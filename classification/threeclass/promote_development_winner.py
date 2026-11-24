"""Protect a stronger completed development candidate before any held-out scoring."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from continuous import known_cv,signature
from experiment import ROOT,load_data,train_one,summarize_group,dump
from evaluate import evaluate_group
from report import paired_ci
from milestone2 import paired_oof


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'results/milestone5';out.mkdir(parents=True,exist_ok=True)
    frozen=out/'selection_before_test.json'
    reference=json.loads((ROOT/'results/milestone4/selection_before_test.json').read_text())
    if frozen.exists():decision=json.loads(frozen.read_text())
    else:
        pool=known_cv();key,winner=max(pool.items(),key=lambda item:item[1]['mean_macro_f1'])
        cv_path=Path(winner['source']);cv=json.loads(cv_path.read_text())
        records=json.loads((ROOT/'runs/continuous_search/leaderboard.json').read_text())
        matching=[r for r in records if signature(r['feature'],r['config'])==key]
        if not matching:raise ValueError('Strongest completed CV candidate has no full development model in this promotion scope')
        record=max(matching,key=lambda r:r['val']['macro_f1'])
        eligible=cv['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        if not eligible:
            dump(out/'decision.json',{'promoted':False,'reason':'Development CV and validation gate failed','external_test_evaluated':False,'candidate':record['candidate_id']});return
        frame,features,digest=load_data(winner['feature']);fit=out/'protected_model'
        available=all(Path(p).exists() for p in record['checkpoints'])
        if available:
            for seed,path in enumerate(record['checkpoints']):
                destination=fit/f'seed_{seed}';destination.mkdir(parents=True,exist_ok=True)
                for name in ['model.pt','summary.json','val_predictions.npz']:shutil.copy2(Path(path).parent/name,destination/name)
        else:
            for seed in range(3):train_one(frame,features,digest,{**record['config'],'seed':seed},fit/f'seed_{seed}')
        candidate=summarize_group(fit,frame)
        candidate.update(feature=winner['feature'],candidate_id=cv['candidate'].removeprefix('continuous_'),source_candidate_id=record['candidate_id'],source_cv_summary=str(cv_path),protected_from_pruning=True,full_fit_refitted=not available)
        assert candidate['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        candidate['checkpoint_sha256']={str(seed):hashlib.sha256(Path(path).read_bytes()).hexdigest() for seed,path in enumerate(candidate['checkpoints'])}
        dump(out/'candidate.json',candidate)
        subprocess.run([sys.executable,str(ROOT/'audit_candidate_cv.py'),'--candidate-json',str(out/'candidate.json')],cwd=ROOT,check=True)
        assert json.loads((cv_path.parent/'provenance_audit.json').read_text())['passed']
        decision={'time':datetime.now(timezone.utc).isoformat(),'rule':'Maximum completed development outer-CV macro-F1 at this cutoff; require improvement over milestone4 and nondecreasing full development validation macro-F1. No external metric participates.','manifest_sha256':digest,'selected':candidate,'cv':cv,'eligible':True,'reference_candidate':reference['selected']['candidate_id'],'reference_cv_macro_f1':reference['cv']['mean_macro_f1'],'reference_validation_macro_f1':reference['selected']['val']['macro_f1'],'pool_at_selection':pool,'provenance_audit_passed':True,'external_evaluation_at_freeze':False}
        dump(frozen,decision)
        dump(out/'paired_outer_cv.json',paired_oof(Path(reference['selected']['source_cv_summary']).parent,cv_path.parent,frame))
    candidate=decision['selected']
    for seed,path in enumerate(candidate['checkpoints']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==candidate['checkpoint_sha256'][str(seed)]
    if not args.evaluate:
        print(json.dumps({'frozen':str(frozen),'candidate':candidate['candidate_id'],'val':candidate['val'],'cv':decision['cv']['mean_macro_f1'],'external_test_evaluated':False}),flush=True);return
    frame,features,digest=load_data(candidate['feature']);target=out/'primary'
    if not (target/'metrics.json').exists():evaluate_group(candidate,frame,features,digest,target)
    document=json.loads((target/'metrics.json').read_text())
    assert abs(document['results']['val']['image']['macro_f1']-candidate['val']['macro_f1'])<1e-6
    paired={}
    for split in ['test_independent','test_ningbo']:
        paired[split]={'versus_direct_mlp':paired_ci(ROOT/'results/vitb_320/baseline/flat'/f'{split}_images.csv',target/f'{split}_images.csv'),'versus_milestone4':paired_ci(ROOT/'results/milestone4/primary'/f'{split}_images.csv',target/f'{split}_images.csv')}
    dump(out/'paired_external_comparison.json',paired)
    print(json.dumps({'candidate':candidate['candidate_id'],'cv_macro_f1':decision['cv']['mean_macro_f1'],'external':{s:r['image'] for s,r in document['results'].items() if s.startswith('test')}}),flush=True)


if __name__=='__main__':main()

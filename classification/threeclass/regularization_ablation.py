"""Isolate the development champion's capacity and regularization choices."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from experiment import ROOT,load_data,train_one,summarize_group,dump
from continuous import canonical_cfg,signature


def main():
    frozen=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    reference=frozen['selected'];feature=reference['feature'];base=reference['config']
    changes=[('remove_fine_contrastive',{'fine_contrastive':0.}),('remove_label_smoothing',{'label_smoothing':0.}),('lower_dropout',{'dropout':.3}),('narrow_head',{'hidden':64}),('lower_weight_decay',{'wd':.01})]
    out=ROOT/'runs'/feature/'regularization_ablation'
    dump(out/'protocol.json',{'hypothesis':'Isolate which development choices contribute to the stronger compact taxonomy configuration; no attribution from joint search alone.','source':'Frozen development configuration and its patient CV; external outcomes excluded.','reference_candidate':reference['candidate_id'],'reference_cv_macro_f1':frozen['cv']['mean_macro_f1'],'reference_validation_macro_f1':reference['val']['macro_f1'],'controls':[{'name':name,'change':change} for name,change in changes],'evaluation':'All five controls receive three seeds and three outer patient folds; existing identical completed fits reused, no external evaluation.'})
    frame,features,digest=load_data(feature);records=[]
    old=json.loads((ROOT/'runs/continuous_search/leaderboard.json').read_text())
    for index,(name,change) in enumerate(changes):
        cfg=canonical_cfg({**base,**change});identifier=signature(feature,cfg);group=out/f'{index:02d}_{name}'
        matches=[r for r in old if signature(r['feature'],r['config'])==identifier]
        existing=next((r for r in matches if all(Path(p).exists() for p in r['checkpoints'])),None)
        for seed in range(3):
            destination=group/f'seed_{seed}'
            if existing and not (destination/'summary.json').exists():
                destination.mkdir(parents=True,exist_ok=True)
                for filename in ['model.pt','summary.json','val_predictions.npz']:shutil.copy2(Path(existing['checkpoints'][seed]).parent/filename,destination/filename)
            train_one(frame,features,digest,{**cfg,'seed':seed},destination)
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id=identifier,control=name,reused_existing_fit=bool(existing))
        records.append(record);dump(group/'candidate.json',record);dump(out/'leaderboard.json',records)
    for record in records:
        candidate=out/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',feature]
            subprocess.run(args,cwd=ROOT,check=True)
    dump(out/'state.json',{'status':'completed','controls':len(records),'external_test_evaluated':False})
    print('All five single-factor development controls completed',flush=True)


if __name__=='__main__':main()

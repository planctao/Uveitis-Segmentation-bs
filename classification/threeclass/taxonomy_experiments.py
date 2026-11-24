"""Compact subtype marginalization, motivated by pre-test branch contribution."""
import hashlib
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    frame,features,digest=load_data('vitb_320')
    base=configuration('attention','flat',128,taxonomy_only=True)
    configs=[{**base,'aux':v} for v in [0.,.2,.6,1.]]
    configs += [{**base,'arch':'global','aux':.6},{**base,'aux':.6,'fine_contrastive':.025},{**base,'aux':.6,'contrastive':.05}]
    out=ROOT/'runs/vitb_320/taxonomy';records=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_{cfg["arch"]}_taxonomy'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature='vitb_320',candidate_id='taxonomy_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True))
    best=max(records,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'selection_rule':'Three-seed validation macro-F1; compact head tests subtype-probability aggregation without redundant routing heads.','external_test_evaluated':False})
    dump(out/'cv_challenger.json',best)

if __name__=='__main__':main()

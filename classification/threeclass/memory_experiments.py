"""Evaluate patient-balanced retrieval as an adaptive branch beside a DINO MLP."""
import hashlib
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    frame,features,digest=load_data('vitb_320')
    configs=[configuration('global','flat',128),configuration('global','flat',256)]
    configs += [configuration('memory_hybrid','flat',h,memory_topk=k) for h,k in [(128,3),(128,15),(256,3),(256,15)]]
    configs += [configuration('memory_hybrid','dual',128,aux=.4,consistency=.3,memory_topk=k) for k in [3,15]]
    configs += [configuration('memory_hybrid','dual',128,aux=.4,consistency=.3,memory_topk=5,memory_cap=.5,contrastive=.05),configuration('memory_hybrid','dual',128,aux=.4,consistency=.3,memory_topk=5,memory_cap=0.)]
    out=ROOT/'runs/vitb_320/memory';records=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_{cfg["arch"]}_{cfg["mode"]}_h{cfg["hidden"]}'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame)
        record.update(feature='vitb_320',candidate_id='memory_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True))
    memory=[r for r in records if r['config']['arch']=='memory_hybrid' and r['config'].get('memory_cap',.75)>0]
    best=max(memory,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':digest,'selection_rule':'Validation image macro-F1, three-seed ensemble; patient prototypes built only from fit patients; own patient excluded for every training query.','best':best,'selected':{'best':best},'external_test_evaluated':False})
    dump(out/'cv_challenger.json',best)

if __name__=='__main__':main()

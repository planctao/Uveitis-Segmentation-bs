"""Disease-to-etiology marginalization and patient-aware fine contrastive ablations."""
import hashlib
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    frame,features,digest=load_data('vitb_320')
    base=configuration('attention','dual',128,aux=.4,consistency=.3)
    configs=[base,{**base,'contrastive':.05},
             {**base,'subtype_fusion':True},
             {**base,'subtype_fusion':True,'subtype_consistency':.1},
             {**base,'fine_contrastive':.025},
             {**base,'contrastive':.05,'fine_contrastive':.025},
             {**base,'subtype_fusion':True,'contrastive':.05,'fine_contrastive':.025},
             {**base,'subtype_fusion':True,'aux':.2,'contrastive':.05,'fine_contrastive':.05},
             {**base,'subtype_fusion':True,'aux':.6,'subtype_consistency':.1}]
    out=ROOT/'runs/vitb_320/subtype_hierarchy';records=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_attention_dual'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame)
        record.update(feature='vitb_320',candidate_id='subtype_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True))
    best=max(records,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'selection_rule':'Three-seed validation macro-F1; subtype mapping constructed only from fit patients; no external tests read.','external_test_evaluated':False})
    dump(out/'cv_challenger.json',best)

if __name__=='__main__':main()

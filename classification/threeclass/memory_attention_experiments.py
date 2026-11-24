"""Combine patient retrieval with the strongest region-aware hierarchy, with controls."""
import hashlib
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    frame,features,digest=load_data('vitb_320');base=configuration('attention','dual',128,aux=.4,consistency=.3)
    configs=[base]
    configs += [{**base,'arch':'memory_attention','memory_topk':k} for k in [3,15]]
    configs += [{**base,'arch':'memory_attention','memory_topk':5,'memory_gate_evidence':True},
                {**base,'arch':'memory_attention','memory_topk':5,'memory_gate_evidence':True,'contrastive':.05},
                {**base,'arch':'memory_attention','memory_topk':5,'memory_cap':0.}]
    out=ROOT/'runs/vitb_320/memory_attention';records=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_{cfg["arch"]}'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame)
        record.update(feature='vitb_320',candidate_id='memory_attention_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True))
    candidates=[r for r in records if r['config']['arch']=='memory_attention' and r['config'].get('memory_cap',.75)>0]
    best=max(candidates,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'external_test_evaluated':False,'selection_rule':'Three-seed validation Macro-F1; fit-patient prototypes only, exclude own patient; matched region-aware head and zero-memory controls.'})
    dump(out/'cv_challenger.json',best)

if __name__=='__main__':main()

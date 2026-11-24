"""Matched ablations for denser regions, foreground masking and local evidence."""
import json
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    frame,features,digest=load_data('vitb_patch7')
    configs=[configuration('global','flat',256),
             configuration('attention','dual',128,aux=.4,consistency=.3,region_heads=3,use_region_mask=False),
             configuration('attention','dual',128,aux=.4,consistency=.3,region_heads=3,use_region_mask=True),
             configuration('evidence_mil','flat',128,evidence_aux=.1,topk=5),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=.1,topk=5),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=0.,topk=5),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=.1,topk=5,global_conditioning=False),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=.1,topk=3,region_dropout=.1),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=.1,topk=8,contrastive=.05),
             configuration('evidence_mil','dual',128,aux=.4,consistency=.3,evidence_aux=.1,topk=5,use_region_mask=False)]
    out=ROOT/'runs/vitb_patch7/explore';records=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_{cfg["arch"]}_{cfg["mode"]}'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record['feature']='vitb_patch7';record['ablation_index']=i;records.append(record)
        dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True))
    best=max(records,key=lambda x:x['val']['macro_f1'])
    dump(out/'selection.json',{'manifest_sha256':digest,'selection_rule':'Three-seed ensemble validation image macro-F1; no external evaluation; matched controls for region density, mask, auxiliary local loss and global conditioning.','best':best,'selected':{'best':best}})
    dump(ROOT/'runs/patch_round.json',{'best':best,'external_tests_evaluated':False,'ablation_hypothesis':'Finer foreground regions and class-specific sparse evidence may recover small lesions diluted by 3x3 pooling.'})

if __name__=='__main__':main()

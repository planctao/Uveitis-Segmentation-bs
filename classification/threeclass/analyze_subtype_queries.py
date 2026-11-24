"""Measure query usage and frozen-branch ablations using development images only."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,metrics,dump
from milestone2 import paired_oof


def main():
    torch.set_num_threads(4);frame,features,digest=load_data('multiscale_vit');valid=np.flatnonzero(frame.split.eq('val').to_numpy());train=frame.split.eq('train').to_numpy()
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());records=json.loads((ROOT/'runs/multiscale_vit/subtype_queries/leaderboard.json').read_text());rows=[]
    for record in records:
        full=[];base=[];local=[];gates=[];entropies=[]
        for path in record['checkpoints']:
            state=torch.load(path,map_location='cpu',weights_only=False);model=Model(state['cfg'],state['dims'],len(state['diseases']));model.load_state_dict(state['state_dict'],strict=True);model.eval()
            data,_=normalize(features,train,stats=state['stats']);fp=[];bp=[];lp=[];gg=[];ee=[]
            with torch.no_grad():
                for start in range(0,len(valid),64):
                    x={k:torch.tensor(v[valid[start:start+64]]) for k,v in data.items()};out=model(x)
                    fp.append(out['p'].numpy());bp.append((out['base_subtype_logits'].softmax(-1)@model.subtype_to_class).numpy());lp.append((out['local_subtype_logits'].softmax(-1)@model.subtype_to_class).numpy());gg.append(out['query_gate'].numpy())
                    a=out['query_attention'];ee.append((-(a*a.clamp(min=1e-9).log()).sum(-1)/np.log(a.shape[-1])).mean((1,2)).numpy())
            full.append(np.concatenate(fp));base.append(np.concatenate(bp));local.append(np.concatenate(lp));gates.append(np.concatenate(gg));entropies.append(np.concatenate(ee))
        f,b,l=[np.mean(v,axis=0) for v in [full,base,local]];y=frame.iloc[valid].label.to_numpy();fm=metrics(y,f);assert abs(fm['macro_f1']-record['val']['macro_f1'])<1e-6
        cp=ROOT/'runs/multiscale_vit/candidate_crossvalidation'/('continuous_'+record['candidate_id']);cv=json.loads((cp/'summary.json').read_text());assert json.loads((cp/'provenance_audit.json').read_text())['passed']
        row={k:record[k] for k in ['candidate_id','ablation_index','config','val']};row['cv']={k:cv[k] for k in ['mean_macro_f1','sd_macro_f1','mean_accuracy']};row['query_branch_removed']=metrics(y,b);row['query_branch_only']=metrics(y,l)
        row['mean_query_gate']=float(np.mean(gates));row['mean_normalized_attention_entropy']=float(np.mean(entropies));row['mean_class_probability_l1_change']=float(np.abs(f-b).sum(1).mean());row['changed_validation_decisions']=int(np.sum(f.argmax(1)!=b.argmax(1)))
        row['passes_frozen_reference_gate']=cv['mean_macro_f1']>reference['cv']['mean_macro_f1'] and fm['macro_f1']>=reference['selected']['val']['macro_f1'];rows.append(row)
    best=max(rows,key=lambda r:r['cv']['mean_macro_f1']);p=ROOT/'runs/multiscale_vit/candidate_crossvalidation'/('continuous_'+best['candidate_id'])
    report={'manifest_sha256':digest,'reference_candidate':reference['selected']['candidate_id'],'reference_cv_macro_f1':reference['cv']['mean_macro_f1'],'reference_val_macro_f1':reference['selected']['val']['macro_f1'],'candidates':rows,'best_cv_candidate':best['candidate_id'],'paired_outer_cv_versus_reference':paired_oof(Path(reference['selected']['source_cv_summary']).parent,p,frame),'external_test_evaluated':False,'decision':'Keep the frozen reference; no disease-query candidate passes both development gates.'}
    assert len(rows)==6 and not any(r['passes_frozen_reference_gate'] for r in rows)
    dump(ROOT/'results/research_subtype_queries/comparison.json',report)
    print(json.dumps([{'candidate':r['candidate_id'],'cv':r['cv']['mean_macro_f1'],'gate':r['mean_query_gate'],'branch_removed_f1':r['query_branch_removed']['macro_f1'],'changed_decisions':r['changed_validation_decisions']} for r in rows]),flush=True)


if __name__=='__main__':main()

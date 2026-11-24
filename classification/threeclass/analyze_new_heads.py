"""Development-only comparison of support, scale sharing and scale interactions."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,metrics,dump
from milestone2 import paired_oof


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    reference=ROOT/'runs/multiscale_vit/candidate_crossvalidation/continuous_4726000e84fe'
    incumbent=json.loads((reference/'summary.json').read_text())
    incumbent_val=.8804553513465684
    # Exact validation reference is read from the frozen development decision only.
    frozen=json.loads((ROOT/'results/milestone4/selection_before_test.json').read_text())
    incumbent_val=frozen['selected']['val']['macro_f1']
    train=frame.split.eq('train').to_numpy();valid=np.flatnonzero(frame.split.eq('val').to_numpy())
    families=[]
    for family in ['support_taxonomy','scale_consensus','scale_interaction']:
        path=ROOT/'runs/multiscale_vit'/family/'leaderboard.json'
        if not path.exists():continue
        records=json.loads(path.read_text());rows=[]
        for record in sorted(records,key=lambda x:x['ablation_index']):
            ckpt=torch.load(record['checkpoints'][0],map_location='cpu',weights_only=False)
            model=Model(ckpt['cfg'],ckpt['dims'],len(ckpt['diseases']));model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
            row={k:record[k] for k in ['candidate_id','group','config','val','ablation_index']}
            row['head_parameters_per_seed']=sum(p.numel() for p in model.parameters())
            cv_path=ROOT/'runs/multiscale_vit/candidate_crossvalidation'/('continuous_'+record['candidate_id'])
            row['cv_complete']=(cv_path/'summary.json').exists()
            if row['cv_complete']:
                cv=json.loads((cv_path/'summary.json').read_text())
                row['cv']={k:cv[k] for k in ['mean_macro_f1','sd_macro_f1','mean_accuracy']}
                audit=json.loads((cv_path/'provenance_audit.json').read_text())
                assert audit['passed'] and audit['manifest_sha256']==digest
                row['audit_passed']=True
                row['fit_patient_counts_by_fold']=[f['inner_patients'] for f in audit['folds']]
                row['eligible_for_external_evaluation']=cv['mean_macro_f1']>incumbent['mean_macro_f1'] and record['val']['macro_f1']>=incumbent_val
            if family.startswith('scale_'):
                weights=[];predictions=[];gates=[]
                for checkpoint in record['checkpoints']:
                    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    model=Model(state['cfg'],state['dims'],len(state['diseases']));model.load_state_dict(state['state_dict'],strict=True);model.eval()
                    normalized,_=normalize(features,train,stats=state['stats'])
                    ps=[];ws=[]
                    with torch.no_grad():
                        for start in range(0,len(valid),64):
                            batch={k:torch.tensor(v[valid[start:start+64]]) for k,v in normalized.items()}
                            out=model(batch);ws.append(out['scale_weights'].numpy());ps.append(out['p'].numpy())
                    weights.append(np.concatenate(ws));predictions.append(np.concatenate(ps))
                    if hasattr(model.scale_pool,'interaction_gate'):gates.append(model.scale_pool.interaction_gate.detach().sigmoid().mean().item())
                w=np.mean(weights,axis=0);p=np.mean(predictions,axis=0)
                computed=metrics(frame.iloc[valid].label.to_numpy(),p)
                assert abs(computed['macro_f1']-record['val']['macro_f1'])<1e-6
                row['validation_scale_weights']={'scale_order':[320,448],'mean':w.mean(0).tolist(),'sd_across_images':w.std(0).tolist(),'mean_entropy':float((-w*np.log(w.clip(1e-9))).sum(1).mean())}
                if gates:row['mean_interaction_gate_by_seed']=gates
            if family=='support_taxonomy':
                counts=ckpt['dims']['disease_patient_counts'];k=ckpt['cfg'].get('support_shrink',10.)
                row['support_shrinkage']={d:{'fit_patients':n,'shrink_factor':float(np.sqrt(n/(n+k)))} for d,n in zip(ckpt['diseases'],counts)}
            rows.append(row)
        complete=[r for r in rows if r['cv_complete']]
        best_val=max(rows,key=lambda r:r['val']['macro_f1'])
        result={'family':family,'planned_configurations':8 if family=='support_taxonomy' else (7 if family=='scale_consensus' else 6),'completed_validation_configurations':len(rows),'completed_cv_configurations':len(complete),'best_validation_candidate':best_val['candidate_id'],'candidates':rows,'external_test_evaluated':False}
        if complete:
            best=max(complete,key=lambda r:r['cv']['mean_macro_f1']);result['best_cv_candidate']=best['candidate_id']
            if len(complete)==result['planned_configurations']:
                directory=ROOT/'runs/multiscale_vit/candidate_crossvalidation'/('continuous_'+best['candidate_id'])
                result['paired_outer_cv_versus_incumbent']=paired_oof(reference,directory,frame)
                result['inference']='Exploratory paired patient bootstrap after family search; intervals do not adjust for candidate selection.'
                if not any(r['eligible_for_external_evaluation'] for r in complete):result['decision']='Keep incumbent; no candidate passed both development CV improvement and nondecreasing validation macro-F1.'
        families.append(result)
    report={'time':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest,'selection_policy':'Select with development evidence only; compare all predeclared ablations; no external evaluation without improved CV and nondecreasing development validation macro-F1.','incumbent':{'candidate_id':'4726000e84fe','cv_macro_f1':incumbent['mean_macro_f1'],'cv_sd':incumbent['sd_macro_f1'],'val_macro_f1':incumbent_val},'families':families,'external_files_read':False,'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['taxonomy_head.py','scale_head.py','interaction_head.py','experiment.py','audit_candidate_cv.py']}}
    dump(ROOT/'results/research_round5/comparison.json',report)
    print(json.dumps({'families':[{k:v for k,v in f.items() if k in ['family','completed_cv_configurations','best_cv_candidate','decision','paired_outer_cv_versus_incumbent']} for f in families]}),flush=True)


if __name__=='__main__':main()

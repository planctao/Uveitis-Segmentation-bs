"""Ablate shared scale encoders, uncertainty gating and cross-scale supervision."""
import hashlib
import subprocess
import sys
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    feature='multiscale_vit';frame,features,digest=load_data(feature)
    base=configuration('scale_consensus','flat',64,taxonomy_only=True,aux=1.)
    configs=[{**base,'learn_scale_gate':False,'scale_aux':0.,'scale_consistency':0.},
             {**base,'scale_aux':0.,'scale_consistency':0.},
             {**base,'scale_aux':.3,'scale_consistency':0.},
             {**base,'scale_aux':.3,'scale_consistency':.05},
             {**base,'scale_aux':.3,'scale_consistency':.05,'scale_dropout':.15},
             {**base,'scale_aux':.3,'scale_consistency':.1,'scale_probability_blend':.5},
             {**base,'scale_aux':.1,'scale_consistency':.05,'hidden':128}]
    out=ROOT/'runs'/feature/'scale_consensus';records=[]
    dump(out/'protocol.json',{'hypothesis':'Share regional encoders across scales and learn per-image reliability instead of concatenating all feature channels.','controls':'Equal averaging, learned gate, auxiliary per-scale subtype targets, scale consistency and scale dropout.','evaluation':'Three seeds per fixed configuration; all candidates get patient-group outer CV; no external tests.'})
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_shared_scales'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='scale_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True));dump(group/'candidate.json',record)
    best=max(records,key=lambda x:x['val']['macro_f1']);dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'external_test_evaluated':False})
    for record in records:
        candidate=out/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',feature]
            subprocess.run(args,cwd=ROOT,check=True)

if __name__=='__main__':main()

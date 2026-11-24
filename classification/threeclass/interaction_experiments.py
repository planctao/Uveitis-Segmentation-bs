"""Test whether shared scale pooling loses useful cross-scale disagreements."""
import hashlib
import subprocess
import sys
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump


def main():
    feature='multiscale_vit';frame,features,digest=load_data(feature)
    base=configuration('scale_interaction','flat',64,taxonomy_only=True,aux=1.,learn_scale_gate=False,scale_aux=0.,scale_consistency=0.,interaction_rank=64)
    configs=[{**base,'interaction_kind':kind} for kind in ['difference','product','both']]
    configs.append({**configs[2],'learn_scale_gate':True})
    configs.append({**configs[3],'scale_aux':.3,'scale_consistency':.05})
    configs.append({**configs[4],'interaction_dropout':.2})
    out=ROOT/'runs'/feature/'scale_interaction';records=[]
    dump(out/'protocol.json',{'hypothesis':'Shared scale means may discard scale disagreements; low-rank absolute differences and products may recover complementary local evidence.','controls':'Difference only, product only, both; equal versus learned scale weights; per-scale supervision and interaction dropout.','evaluation':'All six predeclared configurations receive three seeds and three patient-group outer folds, with inner holdout early stopping. No external test access.'})
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_scale_interaction'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='interaction_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True));dump(group/'candidate.json',record)
    best=max(records,key=lambda x:x['val']['macro_f1']);dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'external_test_evaluated':False})
    for record in records:
        candidate=out/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',feature]
            subprocess.run(args,cwd=ROOT,check=True)
    print('All interaction ablations and CV audits completed',flush=True)


if __name__=='__main__':main()

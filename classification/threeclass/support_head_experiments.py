"""Predeclared ablations for patient-support partial pooling of disease heads."""
import hashlib
import subprocess
import sys
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump

def main():
    feature='multiscale_vit';frame,features,digest=load_data(feature)
    base=configuration('attention','flat',64,taxonomy_only=True,taxonomy_residual=True,aux=1.)
    settings=[('uniform',0.,0.,1.),('support',0.,0.,1.),('support',10.,.001,1.),('support',30.,.001,1.),('uniform',10.,.001,1.),('support',10.,.01,1.),('support',10.,.001,.6)]
    configs=[{**base,'support_prior':p,'support_shrink':k,'residual_penalty':penalty,'aux':aux} for p,k,penalty,aux in settings]
    configs.append({**configs[2],'fine_contrastive':.01})
    out=ROOT/'runs'/feature/'support_taxonomy';records=[]
    dump(out/'protocol.json',{'hypothesis':'A coarse-family classifier plus support-shrunk, centered subtype residuals may generalize better on the long tail.','controls':'Uniform/support prior; zero/10/30 shrinkage; residual penalty; auxiliary weight; fine contrastive.','counts_policy':'Unique fit patients per subtype; rebuilt inside every CV fit mask.','selection':'Compare all predeclared variants with three seeds and three patient-group outer folds; external tests not accessed.'})
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_support_residual'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='support_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(out/'leaderboard.json',sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True));dump(group/'candidate.json',record)
    best=max(records,key=lambda x:x['val']['macro_f1']);dump(out/'selection.json',{'manifest_sha256':digest,'best':best,'selected':{'best':best},'external_test_evaluated':False})
    for record in records:
        candidate=out/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',feature]
            subprocess.run(args,cwd=ROOT,check=True)
    print('All support residual ablations and patient-group CV audits completed',flush=True)

if __name__=='__main__':main()

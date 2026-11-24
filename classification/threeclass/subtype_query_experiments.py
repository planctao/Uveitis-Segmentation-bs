"""Predeclared disease-specific region evidence experiments, development only."""
import hashlib
import json
import subprocess
import sys
from experiment import ROOT,load_data,train_one,summarize_group,dump


def main():
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    base={**reference['selected']['config'],'arch':'subtype_attention','query_heads':2,'query_temperature':.2,'query_dropout':.1,'query_aux':0.}
    configs=[{**base,'query_heads':1},base,{**base,'query_aux':.2},{**base,'wd':.01},{**base,'wd':.01,'query_aux':.2},{**base,'wd':.01,'query_aux':.2,'query_temperature':.5,'query_dropout':.3}]
    feature='multiscale_vit';frame,features,digest=load_data(feature);out=ROOT/'runs'/feature/'subtype_queries';records=[]
    dump(out/'protocol.json',{'hypothesis':'Shared generic regional pooling may hide small disease-specific evidence; bind separate spatial queries to each subtype and gate their residual scores into the unchanged global taxonomy classifier.','controls':'One/two queries per subtype, local subtype supervision, high/lower weight decay and query temperature/dropout.','initialization':'All base parameters use the same initialization as attention; zero residual readout starts with exactly the base prediction.','reference_candidate':reference['selected']['candidate_id'],'second_development_control':'b9ae533d779d, same compact head with weight decay .01; higher CV but lower development validation.','configuration_plan':configs,'evaluation':'All six candidates receive three seeds and three outer patient folds with independent inner stopping patients; no external tests.'})
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_subtype_queries'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='query_'+hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);dump(group/'candidate.json',record);dump(out/'leaderboard.json',sorted(records,key=lambda r:r['val']['macro_f1'],reverse=True))
    best=max(records,key=lambda r:r['val']['macro_f1']);dump(out/'selection.json',{'best':best,'selected':{'best':best},'external_test_evaluated':False})
    for record in records:
        candidate=out/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',feature]
            subprocess.run(args,cwd=ROOT,check=True)
    dump(out/'state.json',{'status':'completed','candidates':6,'external_test_evaluated':False})
    print('All disease-query candidates and provenance audits completed',flush=True)


if __name__=='__main__':main()

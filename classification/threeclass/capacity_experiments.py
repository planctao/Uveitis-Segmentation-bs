"""Compare a larger frozen DINO backbone and complementary DINO feature fusion."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from experiment import ROOT,configuration,load_data,train_one,summarize_group,dump


def plan():
    large='dinov2_vitl_322';fusion='dinov2l_dinov3b'
    return [(large,configuration('global','flat',256)),
            (large,configuration('attention','flat',64,taxonomy_only=True,aux=1.)),
            (large,configuration('attention','flat',128,taxonomy_only=True,aux=1.)),
            (large,configuration('attention','dual',128,aux=.4,consistency=.3)),
            (fusion,configuration('attention','flat',64,taxonomy_only=True,aux=1.)),
            (fusion,configuration('attention','flat',128,taxonomy_only=True,aux=1.,dropout=.45))]


def fuse():
    frame,large,digest=load_data('dinov2_vitl_322');other,base,other_digest=load_data('vitb_320')
    assert digest==other_digest and np.array_equal(frame.path,other.path)
    out=ROOT/'features/dinov2l_dinov3b.npz'
    arrays={k:np.concatenate([large[k],base[k]],axis=-1).astype('float16') for k in large}
    arrays.update(path=frame.path.to_numpy(dtype=str),manifest_sha256=np.array(digest))
    with out.with_suffix('.tmp').open('wb') as target:np.savez_compressed(target,**arrays)
    os.replace(out.with_suffix('.tmp'),out)
    dump(out.with_suffix('.json'),{'backbone':'Frozen DINOv2 ViT-L/14 plus DINOv3 ViT-B/16','source_features':['dinov2_vitl_322','vitb_320'],'modality_dimensions':[1024,768],'manifest_sha256':digest,'shapes':{k:list(v.shape) for k,v in arrays.items()},'policy':'Concatenate corresponding regional features; no equal-half cross-attention or scale-sharing on unequal modalities.'})


def main():
    fuse();records=[]
    for i,(feature,cfg) in enumerate(plan()):
        frame,features,digest=load_data(feature);out=ROOT/'runs'/feature/'explore';group=out/f'capacity_{i:02d}_{cfg["arch"]}'
        for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='capacity_'+hashlib.sha256(str(sorted(cfg.items())).encode()+feature.encode()).hexdigest()[:12],ablation_index=i)
        records.append(record);own=[r for r in records if r['feature']==feature];best=max(own,key=lambda r:r['val']['macro_f1'])
        dump(out/'leaderboard.json',sorted(own,key=lambda r:r['val']['macro_f1'],reverse=True));dump(out/'selection.json',{'best':best,'selected':{'best':best},'external_test_evaluated':False});dump(group/'candidate.json',record)
        dump(ROOT/'runs/backbone_capacity/leaderboard.json',records)
    for record in records:
        candidate=ROOT/'runs'/record['feature']/'explore'/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',record['feature']]
            subprocess.run(args,cwd=ROOT,check=True)
    print('All capacity candidates and patient-group CV audits completed',flush=True)


if __name__=='__main__':main()

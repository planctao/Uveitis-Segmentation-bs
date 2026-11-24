"""Compare genuine-resolution features with matched upsampling controls."""
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
from experiment import ROOT,load_data,train_one,summarize_group,dump


def fuse(arm,size=768):
    frame,base,digest=load_data('multiscale_vit');other,high,other_digest=load_data(f'vitb_{arm}{size}')
    assert digest==other_digest and np.array_equal(frame.path,other.path)
    destination=ROOT/'features'/f'multiscale_{arm}{size}.npz'
    arrays={key:np.concatenate([base[key],high[key]],axis=-1).astype('float16') for key in base}
    arrays.update(path=frame.path.to_numpy(dtype=str),manifest_sha256=np.array(digest))
    with destination.with_suffix('.tmp').open('wb') as target:np.savez_compressed(target,**arrays)
    os.replace(destination.with_suffix('.tmp'),destination)
    dump(destination.with_suffix('.json'),{'backbone':'DINOv3 ViT-B/16 at 320, 448 and 768','source_features':['multiscale_vit',f'vitb_{arm}{size}'],'view_dimensions':[768,768,768],'high_resolution_arm':arm,'manifest_sha256':digest,'shapes':{key:list(value.shape) for key,value in arrays.items()}})


def main():
    fuse('raw');fuse('up')
    frozen=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());cfg=frozen['selected']['config'];records=[]
    for feature in ['vitb_raw768','vitb_up768','multiscale_raw768','multiscale_up768']:
        frame,features,digest=load_data(feature);group=ROOT/'runs'/feature/'explore'/'00_matched_highres'
        for seed in range(3):
            dump(ROOT/'runs/raw_highres/state.json',{'status':'training_heads','pid':os.getpid(),'feature':feature,'seed':seed,'external_test_evaluated':False})
            train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        record=summarize_group(group,frame);record.update(feature=feature,candidate_id='highres_'+hashlib.sha256((feature+str(sorted(cfg.items()))).encode()).hexdigest()[:12])
        records.append(record);dump(group/'candidate.json',record)
        dump(group.parent/'selection.json',{'best':record,'selected':{'best':record},'external_test_evaluated':False});dump(group.parent/'leaderboard.json',[record])
        dump(ROOT/'runs/raw_highres/leaderboard.json',records)
    for record in records:
        candidate=ROOT/'runs'/record['feature']/'explore'/record['group']/'candidate.json'
        for script in ['crossvalidate.py','audit_candidate_cv.py']:
            args=[sys.executable,'-u',str(ROOT/script),'--candidate-json',str(candidate)]
            if script=='crossvalidate.py':args+=['--features',record['feature']]
            subprocess.run(args,cwd=ROOT,check=True)
    dump(ROOT/'runs/raw_highres/state.json',{'status':'completed_development_experiments','pid':os.getpid(),'configurations':4,'external_test_evaluated':False})
    print('All four raw-resolution/upsampling control configurations and patient CV completed',flush=True)


if __name__=='__main__':main()

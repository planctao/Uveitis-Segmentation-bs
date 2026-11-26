"""Image-only near-duplicate grouping plus separate annotation disagreement audit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.dataset import decode_mask_array


def main():
    out=ROOT/'outputs/severity_research_20261013/data'
    frame=pd.read_csv(out/'images.csv',dtype={'dhash':str})
    pairs=pd.read_csv(out/'near_duplicate_candidates.csv')
    uid_index={s:i for i,s in enumerate(frame.uid)}
    def read(p):
        return np.asarray(Image.open(p).convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=float)/255
    with ThreadPoolExecutor(max_workers=4) as ex:images=list(ex.map(read,frame.image_path))
    parent=list(range(len(frame)))
    def find(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        parent[find(i)]=find(j)
    verified=[]
    for row in pairs.itertuples():
        i,j=uid_index[row.uid_a],uid_index[row.uid_b]
        ssim=structural_similarity(images[i],images[j],data_range=1.)
        if ssim>=.97:
            join(i,j)
            verified.append(dict(uid_a=row.uid_a,uid_b=row.uid_b,ssim=float(ssim),cross_fold=row.cross_fold,exact=row.exact))
    pd.DataFrame(verified,columns=['uid_a','uid_b','ssim','cross_fold','exact']).to_csv(out/'ssim_verified_near_pairs.csv',index=False)
    frame['near_group']=[find(i) for i in range(len(frame))]
    frame['exclude_exact_group']=frame.pixel_sha256.duplicated(keep=False)
    frame.to_csv(out/'image_groups.csv',index=False)
    evaluation=pd.read_csv(out/'mask_burden_evaluation_only.csv')
    disagreements=[]
    for sha,g in frame.groupby('pixel_sha256'):
        if len(g)<2:continue
        masks=[]
        for r in g.itertuples():
            ip=Path(r.image_path)
            path=ip.parents[2]/'mask_only_itksnap'/r.fold/(ip.stem+'.png')
            masks.append(decode_mask_array(np.asarray(Image.open(path)),path))
        a,b=masks[:2]
        union_a,union_b=a>0,b>0
        dice=2*(union_a&union_b).sum()/max(union_a.sum()+union_b.sum(),1)
        disagreements.append(dict(uids=list(g.uid),folds=list(g.fold),
                                   pixel_label_disagreement_fraction=float((a!=b).mean()),
                                   union_dice=float(dice),
                                   union_area_difference=float(abs(union_a.mean()-union_b.mean()))))
    (out/'exact_duplicate_annotation_disagreement.json').write_text(json.dumps(disagreements,indent=2)+'\n')
    group_sizes=frame.groupby('near_group').size()
    report=dict(dhash_candidates=len(pairs),ssim_threshold=.97,grid_size=64,
                verified_pairs=len(verified),verified_crossfold_pairs=sum(r['cross_fold'] for r in verified),
                near_groups=int(group_sizes.size),nontrivial_groups=int((group_sizes>1).sum()),
                max_group_size=int(group_sizes.max()),excluded_exact_group_images=int(frame.exclude_exact_group.sum()),
                main_analysis_images=int((~frame.exclude_exact_group).sum()),
                duplicate_union_dice_quantiles=np.quantile([d['union_dice'] for d in disagreements],[0,.25,.5,.75,1]).tolist(),
                duplicate_area_difference_quantiles=np.quantile([d['union_area_difference'] for d in disagreements],[0,.25,.5,.75,1]).tolist(),
                patient_independence_proven=False)
    (out/'group_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

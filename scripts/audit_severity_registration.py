"""Recover conservative image-structure links without masks or patient labels."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_features import retinal_fov


def prepare_image(path):
    raw=np.asarray(Image.open(path).convert('L').resize((512,512),Image.Resampling.BILINEAR))
    enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(raw)
    mask=retinal_fov(raw.astype(float)/255).astype(np.uint8)*255
    orb=cv2.ORB_create(nfeatures=2000,fastThreshold=10,edgeThreshold=19)
    keypoints,descriptors=orb.detectAndCompute(enhanced,mask)
    points=np.array([k.pt for k in keypoints],dtype=np.float32).reshape(-1,2)
    return dict(raw=raw,mask=mask,points=points,descriptors=descriptors)


def check_pair(item,prepared):
    i,j=item;a,b=prepared[i],prepared[j]
    result=dict(i=i,j=j,confirmed=False,reason='few_descriptors',n_good=0,n_inliers=0)
    if a['descriptors'] is None or b['descriptors'] is None or min(len(a['descriptors']),len(b['descriptors']))<40:return result
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    forward=matcher.knnMatch(a['descriptors'],b['descriptors'],k=2)
    backward=matcher.knnMatch(b['descriptors'],a['descriptors'],k=2)
    reverse={(m.trainIdx,m.queryIdx) for pair in backward if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
    good=[m for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and (m.queryIdx,m.trainIdx) in reverse]
    result['n_good']=len(good)
    if len(good)<40:result['reason']='few_mutual_matches';return result
    pa=a['points'][[m.queryIdx for m in good]];pb=b['points'][[m.trainIdx for m in good]]
    matrix,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=2000,confidence=.995,refineIters=10)
    if matrix is None:result['reason']='ransac_failed';return result
    inliers=inliers.ravel().astype(bool);count=int(inliers.sum());ratio=float(inliers.mean())
    predicted=pa@matrix[:,:2].T+matrix[:,2]
    error=float(np.median(np.linalg.norm(predicted[inliers]-pb[inliers],axis=1))) if count else float('inf')
    scale=float(np.linalg.norm(matrix[:,0]));result.update(n_inliers=count,inlier_fraction=ratio,median_error=error,scale=scale)
    if count<40 or ratio<.6 or error>1.5 or not .7<=scale<=1.4:
        result['reason']='geometry_rejected';return result
    warped_mask=cv2.warpAffine(a['mask'],matrix,(512,512),flags=cv2.INTER_NEAREST)
    overlap=(warped_mask>0)&(b['mask']>0)
    overlap_fraction=float(overlap.sum()/max(min((warped_mask>0).sum(),(b['mask']>0).sum()),1))
    result['overlap_fraction']=overlap_fraction
    if overlap_fraction<.2:result['reason']='small_overlap';return result
    warped=cv2.warpAffine(cv2.GaussianBlur(a['raw'],(5,5),1.5),matrix,(512,512))
    other=cv2.GaussianBlur(b['raw'],(5,5),1.5)
    x,y=warped[overlap].astype(float),other[overlap].astype(float)
    correlation=float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>1e-8 else 0.
    result['aligned_correlation']=correlation
    if correlation<.4:result['reason']='photometry_rejected';return result
    result.update(confirmed=True,reason='confirmed',matrix=matrix.tolist())
    return result


def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(20261013)
    root=ROOT/'outputs/severity_research_20261013';out=root/'registration_audit';out.mkdir(exist_ok=True)
    frame=pd.read_csv(root/'data/image_groups.csv',dtype={'dhash':str})
    cache=np.load(root/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    assert np.array_equal(frame.uid.to_numpy(),cache['uids'])
    uid_index={uid:i for i,uid in enumerate(frame.uid)}
    nn=NearestNeighbors(n_neighbors=31,metric='cosine',n_jobs=4).fit(normalize(cache['embeddings'][0]))
    indices=nn.kneighbors(return_distance=False)
    pairs={tuple(sorted((i,int(j)))) for i,row in enumerate(indices) for j in row if i!=j}
    prior=pd.read_csv(root/'data/near_duplicate_candidates.csv')
    pairs|={tuple(sorted((uid_index[r.uid_a],uid_index[r.uid_b]))) for r in prior.itertuples()}
    pairs=sorted(pairs);start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:prepared=list(pool.map(prepare_image,frame.image_path))
    joblib.dump(prepared,out/'orb_descriptors_and_gray.joblib',compress=3)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k,result in enumerate(pool.map(lambda pair:check_pair(pair,prepared),pairs)):
            result['uid_a']=frame.uid.iloc[result['i']];result['uid_b']=frame.uid.iloc[result['j']]
            result['cross_fold']=bool(frame.fold.iloc[result['i']]!=frame.fold.iloc[result['j']])
            result['exact_pixels']=bool(frame.pixel_sha256.iloc[result['i']]==frame.pixel_sha256.iloc[result['j']])
            results.append(result)
            if (k+1)%1000==0:
                status=dict(completed_pairs=k+1,total_pairs=len(pairs),confirmed=sum(r['confirmed'] for r in results),elapsed_seconds=time.monotonic()-start)
                (out/'progress.json').write_text(json.dumps(status,indent=2)+'\n');print(status,flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='matrix'} for r in results]).to_csv(out/'all_candidate_results.csv',index=False)
    confirmed=[r for r in results if r['confirmed']]
    (out/'confirmed_with_transforms.json').write_text(json.dumps(confirmed,indent=2)+'\n')
    parent=list(range(len(frame)))
    def find(i):
        while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for r in confirmed:parent[find(r['i'])]=find(r['j'])
    frame['registration_group']=[find(i) for i in range(len(frame))]
    frame.to_csv(out/'image_registration_groups.csv',index=False)
    sizes=frame.groupby('registration_group').size()
    cross=frame.groupby('registration_group').fold.nunique()
    report=dict(n_images=len(frame),candidate_pairs=len(pairs),confirmed_pairs=len(confirmed),
        crossfold_confirmed=sum(r['cross_fold'] for r in confirmed),nonexact_crossfold=sum(r['cross_fold'] and not r['exact_pixels'] for r in confirmed),
        known_exact_pair_candidates=sum(r['exact_pixels'] for r in results),known_exact_confirmed=sum(r['exact_pixels'] for r in confirmed),
        registration_groups=len(sizes),nontrivial_groups=int((sizes>1).sum()),max_group_size=int(sizes.max()),
        crossfold_groups=int((cross>1).sum()),elapsed_seconds=time.monotonic()-start,
        true_patient_ids_recovered=False,masks_read=False,opencv_version=cv2.__version__,
        protocol_sha256=hashlib.sha256((ROOT/'docs/severity_registration_protocol_20261013.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    selected=[r for r in confirmed if r['cross_fold'] and not r['exact_pixels']][:12]
    montage=Image.new('RGB',(3*768,4*280));draw=ImageDraw.Draw(montage)
    for k,r in enumerate(selected):
        x,y=k%3*768,k//3*280
        a=Image.fromarray(prepared[r['i']]['raw']).resize((256,256));b=Image.fromarray(prepared[r['j']]['raw']).resize((256,256))
        matrix=np.asarray(r['matrix']);warped=cv2.warpAffine(prepared[r['i']]['raw'],matrix,(512,512))
        overlay=np.stack([warped,prepared[r['j']]['raw'],prepared[r['j']]['raw']],axis=-1)
        montage.paste(a,(x,y));montage.paste(b,(x+256,y));montage.paste(Image.fromarray(overlay).resize((256,256)),(x+512,y))
        draw.text((x+3,y+258),f'{r["uid_a"][:7]} {r["uid_b"][:7]} inliers={r["n_inliers"]} rho={r["aligned_correlation"]:.2f}',fill='white')
    montage.save(out/'crossfold_registration_examples.jpg')


if __name__=='__main__':main()

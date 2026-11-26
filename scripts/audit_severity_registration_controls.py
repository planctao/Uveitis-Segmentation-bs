"""Fixed detector controls; do not alter the completed registration graph."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_features import retinal_fov
from audit_severity_registration import check_pair


def prepare(raw):
    enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(raw)
    mask=retinal_fov(raw.astype(float)/255).astype(np.uint8)*255
    orb=cv2.ORB_create(nfeatures=2000,fastThreshold=10,edgeThreshold=19)
    keypoints,descriptors=orb.detectAndCompute(enhanced,mask)
    points=np.array([k.pt for k in keypoints],dtype=np.float32).reshape(-1,2)
    return dict(raw=raw,mask=mask,points=points,descriptors=descriptors)


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261013)
    root=ROOT/'outputs/severity_research_20261013'
    out=root/'registration_audit'
    prepared=joblib.load(out/'orb_descriptors_and_gray.joblib')
    frame=pd.read_csv(out/'image_registration_groups.csv')
    rng=np.random.default_rng(20261013)
    indices=rng.choice(len(frame),32,replace=False)
    results=[]
    for idx in indices:
        a=prepared[idx]
        for gamma in [.8,1.2]:
            raw=np.rint((a['raw'].astype(float)/255)**gamma*255).astype(np.uint8)
            result=check_pair((0,1),[a,prepare(raw)])
            results.append(dict(uid=frame.uid.iloc[idx],condition=f'gamma{gamma}',**result))
        matrix=cv2.getRotationMatrix2D((256,256),5.,1.)
        matrix[:,2]+=np.array([12.,-8.])
        raw=cv2.warpAffine(a['raw'],matrix,(512,512))
        result=check_pair((0,1),[a,prepare(raw)])
        results.append(dict(uid=frame.uid.iloc[idx],condition='rotate5_shift12minus8',**result))
        b={**a,'points':a['points'][rng.permutation(len(a['points']))]}
        result=check_pair((0,1),[a,b])
        results.append(dict(uid=frame.uid.iloc[idx],condition='coordinate_shuffle_null',**result))
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='matrix'} for r in results])
    table.to_csv(out/'detector_controls.csv',index=False)
    sources=[Path(__file__),ROOT/'scripts/audit_severity_registration.py',ROOT/'docs/severity_registration_detector_controls_20261013.md']
    summary=dict(n_images=32,n_tests=len(table),masks_read=False,thresholds_changed=False,
        condition_summary=table.groupby('condition').agg(n=('uid','size'),confirmed=('confirmed','sum')).reset_index().to_dict('records'),
        rejection_reasons=table.groupby(['condition','reason']).size().rename('n').reset_index().to_dict('records'),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        real_different_eye_specificity_estimated=False,real_patient_recall_estimated=False)
    (out/'detector_controls_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()

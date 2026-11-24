"""DINOv3 ConvNeXt-Tiny frozen-feature multiclass disease classification."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SEED=20261021
RUN=ROOT/'runs/dinov3_convnext_tiny_binary'
OUT=ROOT/'runs/dinov3_convnext_tiny_multiclass'


def patient_probs(df, probs):
    x=df[['patient_group','disease']].copy()
    x['row']=np.arange(len(x))
    rows=[]
    for patient,g in x.groupby('patient_group',sort=True):
        rows.append((patient,g.disease.iloc[0],probs[g.row.to_numpy()].mean(axis=0)))
    return pd.DataFrame({'patient_group':[x[0] for x in rows], 'disease':[x[1] for x in rows],
                         'probability':[x[2] for x in rows]})


def report(df, probs, le, split, c):
    y=le.transform(df.disease);pred=probs.argmax(1)
    patient=patient_probs(df,probs);py=le.transform(patient.disease);pp=np.stack(patient.probability);ppred=pp.argmax(1)
    metrics=dict(split=split,C=c,images=len(df),patients=len(patient),
                 image_accuracy=float(accuracy_score(y,pred)),
                 image_balanced_accuracy=float(recall_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
                 image_macro_f1=float(f1_score(y,pred,labels=np.arange(len(le.classes_)),average='macro',zero_division=0)),
                 patient_accuracy=float(accuracy_score(py,ppred)),
                 patient_balanced_accuracy=float(recall_score(py,ppred,labels=np.unique(py),average='macro',zero_division=0)),
                 patient_macro_f1=float(f1_score(py,ppred,labels=np.arange(len(le.classes_)),average='macro',zero_division=0)),
                 patient_top3_accuracy=float(top_k_accuracy_score(py,pp,k=3,labels=np.arange(len(le.classes_)))),
                 patient_log_loss=float(log_loss(py,pp,labels=np.arange(len(le.classes_)))),
                 confusion_matrix=confusion_matrix(py,ppred,labels=np.arange(len(le.classes_))).tolist())
    per=[]
    for i,disease in enumerate(le.classes_):
        ii=py==i
        per.append(dict(split=split,disease=disease,true_patients=int(ii.sum()),
                        correct_patients=int((ppred[ii]==i).sum()),
                        patient_accuracy=float((ppred[ii]==i).mean()) if ii.any() else None))
    return metrics,per


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'metadata/manifest.csv')
    arrays=np.load(RUN/'frozen_features.npz')
    disease_patient_counts=d.groupby('patient_group').disease.nunique()
    disease_conflicts=set(disease_patient_counts[disease_patient_counts>1].index)
    frames={}
    feats={}
    for split, frame in d.groupby('split',sort=True):
        frame=frame.reset_index(drop=True)
        if len(frame)!=len(arrays[split]):
            raise ValueError(f'{split}: {len(frame)} manifest rows but {len(arrays[split])} feature rows')
        mask=~frame.patient_group.isin(disease_conflicts).to_numpy()
        frames[split]=frame.loc[mask].reset_index(drop=True)
        feats[split]=arrays[split][mask]
    labels=sorted(frames['train'].disease.unique())
    le=LabelEncoder().fit(labels)
    scaler=StandardScaler().fit(feats['train'])
    patient_counts=frames['train'].drop_duplicates('patient_group').disease.value_counts()
    train_weights=np.array([len(frames['train'])/(len(labels)*patient_counts[r.disease]*frames['train'].patient_group.value_counts()[r.patient_group]) for r in frames['train'].itertuples()])
    trials=[];best=None
    for c in [.0003,.001,.003,.01,.03,.1,1.]:
        # lbfgs automatically uses multinomial loss for >2 classes in current sklearn.
        clf=LogisticRegression(C=c,max_iter=4000,solver='lbfgs',random_state=SEED)
        clf.fit(scaler.transform(feats['train']),le.transform(frames['train'].disease),sample_weight=train_weights)
        prob=clf.predict_proba(scaler.transform(feats['val']))
        pat=patient_probs(frames['val'],prob); y=le.transform(pat.disease)
        score=f1_score(y,pat.probability.map(lambda x:int(np.argmax(x))),labels=np.arange(len(le.classes_)),average='macro',zero_division=0)
        trials.append(dict(C=c,val_patient_macro_f1=float(score)))
        if best is None or score>best[0]:best=(score,c,clf)
    _,c,clf=best
    joblib.dump(dict(scaler=scaler,classifier=clf,label_encoder=le,classes=labels),OUT/'multiclass_probe.joblib')
    (OUT/'search.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2))
    reports=[];per=[]
    for split,frame in frames.items():
        prob=clf.predict_proba(scaler.transform(feats[split]))
        m,p=report(frame,prob,le,split,c);reports.append(m);per.extend(p)
        out=frame.copy();out['prediction']=le.inverse_transform(prob.argmax(1));out['confidence']=prob.max(1)
        out.to_csv(OUT/f'{split}_images.csv',index=False)
        pat=patient_probs(frame,prob);pat['prediction']=le.inverse_transform(np.stack(pat.probability).argmax(1))
        pat.to_csv(OUT/f'{split}_patients.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    pd.DataFrame(per).to_csv(OUT/'per_disease_patient_metrics.csv',index=False)
    config=dict(task='13-class specific disease classification',classes=labels,
                excluded=['IU','脉络膜视网膜炎','视网膜血管炎','Normal','Other'],
                feature_source=str(RUN/'frozen_features.npz'),selection='validation patient macro-F1',seed=SEED,
                excluded_conflicting_disease_patients=sorted(disease_conflicts),
                note='Frozen DINOv3 ConvNeXt-Tiny backbone plus multinomial logistic probe; external test labels are not used for C selection.')
    (OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print(json.dumps({'best_C':c,'classes':labels,'reports':reports},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

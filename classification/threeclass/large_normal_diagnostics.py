"""Training-only normal geometry in the already frozen DINOv3-L feature cache."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT,KEYS,normalize,dump
from normal_subspace import fit_normal_geometry,patient_centers
from dinov3_large_data import load_derived
from kernel_residual_experiments import sha


def main():
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv');mask=frame.split.eq('train').to_numpy();tf=frame.loc[mask].reset_index(drop=True)
    prior=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());results=[]
    for arm in prior['configuration_plan']:
        features=load_derived(ROOT,frame,digest,arm['feature']);normalized,_=normalize(features,mask);data={k:v[mask] for k,v in normalized.items()}
        geometry=fit_normal_geometry(tf,data);streams={}
        for key in KEYS:
            names,normal=patient_centers(tf,data,key);normal=normal-normal.mean(0)
            b=geometry[key+'_basis'].astype('float64');total=float(np.sum(normal**2))
            x=data[key].astype('float64');x=x.mean(1) if x.ndim==3 else x
            patient_rows=[];labels=[]
            for (patient,label),group in tf.groupby(['patient_group','label']):patient_rows.append(x[group.index].mean(0));labels.append(int(label))
            patient_rows=np.stack(patient_rows);labels=np.array(labels)
            means=np.stack([patient_rows[labels==j].mean(0) for j in range(3)]);contrasts=means[[0,0,1]]-means[[1,2,2]]
            fractions=np.square(contrasts@b).sum(1)/np.square(contrasts).sum(1)
            streams[key]={'dimension':x.shape[-1],'normal_patients':len(names),'normal_centroid_variance_fraction_in_top32':float(np.square(normal@b).sum()/total),'coarse_mean_contrast_fraction_in_normal32':dict(zip(['infection_vs_noninfection','infection_vs_normal','noninfection_vs_normal'],fractions.tolist()))}
        results.append({'feature':arm['feature'],'streams':streams});print({'feature':arm['feature'],'global':streams['global']},flush=True)
        del features,normalized,data
    dump(ROOT/'results/large_normal_diagnostics.json',{'manifest_sha256':digest,'scope':'Train split only, equal normal-patient centers and equal patient-label coarse means. Description of feature geometry; neither classification-generalization evidence nor causal interpretation. All four prior cached feature arms retained without choosing among them.','arms':results,'code_sha256':{p:sha(ROOT/p) for p in ['large_normal_diagnostics.py','dinov3_large_data.py','normal_subspace.py']},'source_protocol_sha256':sha(ROOT/'runs/dinov3_large/protocol.json'),'external_test_evaluated':False,'new_classification_heads':0})


if __name__=='__main__':main()

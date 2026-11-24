"""Fit-only patient-balanced diagnostic signal in discarded normal coordinates."""
import numpy as np
from experiment import ROOT,KEYS,load_data,normalize,dump
from normal_subspace import fit_normal_geometry
from kernel_residual_experiments import sha


def main():
    frame,features,digest=load_data('multiscale_vit')
    fit=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[fit].reset_index(drop=True)
    for v in features.values():v[2091:]=np.nan
    mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,_=normalize(features,mask);data={k:v[fit] for k,v in normalized.items()}
    geometry=fit_normal_geometry(tf,data);streams={}
    for key in KEYS:
        x=data[key].astype('float64')
        if x.ndim==3:x=x.mean(1)
        rows=[];labels=[]
        for (patient,label),group in tf.groupby(['patient_group','label']):
            rows.append(x[group.index].mean(0));labels.append(int(label))
        rows=np.stack(rows);labels=np.array(labels);b=geometry[key+'_basis'].astype('float64')
        class_means=np.stack([rows[labels==label].mean(0) for label in range(3)])
        contrasts=class_means[[0,0,1]]-class_means[[1,2,2]]
        numerator=np.sum(np.square(contrasts@b),1);denominator=np.sum(np.square(contrasts),1)
        streams[key]={'patient_label_groups':len(rows),'fraction_pairwise_class_mean_squared_contrast_removed':dict(zip(['infection_vs_noninfection','infection_vs_normal','noninfection_vs_normal'],(numerator/denominator).tolist()))}
    dump(ROOT/'results/normal_coordinate_diagnostics.json',{'manifest_sha256':digest,'scope':'Train split only; equal patient-label centers. Descriptive contrast energy; no performance or causal inference. No new model fits.','fit_images':len(fit),'streams':streams,'code_sha256':sha(ROOT/'normal_coordinate_diagnostics.py'),'external_test_evaluated':False})
    print(streams,flush=True)


if __name__=='__main__':main()

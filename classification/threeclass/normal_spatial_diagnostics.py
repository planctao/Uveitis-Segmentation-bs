"""Train-only held-normal-patient spatial prediction diagnostic."""
import numpy as np
from sklearn.model_selection import KFold
from experiment import ROOT,load_data,normalize,dump
from normal_spatial_reference import fit_spatial_geometry,spatial_patient_data
from normal_subspace import independent_transform
from kernel_residual_experiments import sha


def main():
    frame,features,digest=load_data('multiscale_vit');train=frame.split.eq('train').to_numpy();normal=frame.label.eq(2).to_numpy()
    for v in features.values():v[2091:]=np.nan
    names=np.array(sorted(frame.loc[train&normal,'patient_group'].unique()));folds=[];errors=[]
    for fold,(a,b) in enumerate(KFold(n_splits=5,shuffle=True,random_state=20283024).split(names)):
        fit=train&~frame.patient_group.isin(names[b]).to_numpy();held=train&normal&frame.patient_group.isin(names[b]).to_numpy()
        assert not set(frame.loc[fit,'patient_group'])&set(frame.loc[held,'patient_group'])
        normalized,_=normalize(features,fit);geometry=fit_spatial_geometry(frame.loc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()})
        results=[]
        for name in names[b]:
            ids=np.flatnonzero(held&frame.patient_group.eq(name).to_numpy());r=independent_transform(normalized['regions'][ids],geometry['regions_center'],geometry['regions_basis'],1.).astype('float64');target=(r-r.mean(1,keepdims=True)).mean(0)
            x=normalized['global'][ids].astype('float64').mean(0);q=np.clip(((x-geometry['global_center'])@geometry['global_basis'][:,:8])/geometry['spatial_deviation'],-5,5)
            unconditional=geometry['spatial_center'].astype('float64');conditional=unconditional+np.einsum('k,ktd->td',q-geometry['spatial_qmean'],geometry['spatial_coefficients'].astype('float64'))
            row={'patient':name,'zero_mse':float(np.mean(target**2)),'mean_mse':float(np.mean((target-unconditional)**2)),'conditional_mse':float(np.mean((target-conditional)**2))};results.append(row);errors.append(row)
        folds.append({'fold':fold,'fit_normal_patients':len(geometry['normal_patients']),'held_normal_patients':len(b),'patients':results});print({'fold':fold,'patients':len(b)},flush=True)
    summary={k:float(np.mean([r[k] for r in errors])) for k in ['zero_mse','mean_mse','conditional_mse']}
    summary['conditional_reduction_vs_zero']=1-summary['conditional_mse']/summary['zero_mse'];summary['conditional_reduction_vs_mean']=1-summary['conditional_mse']/summary['mean_mse']
    dump(ROOT/'results/normal_spatial_diagnostics.json',{'manifest_sha256':digest,'scope':'Training only; five held-normal-patient folds; all normalization and four normal PCA spaces fitted excluding held patients. Equal normal patient scores. Frozen global8 coordinate ridge lambda1. Diagnostic of spatial prediction,not classification.','summary':summary,'folds':folds,'new_classification_heads':0,'code_sha256':{p:sha(ROOT/p) for p in ['normal_spatial_reference.py','normal_spatial_diagnostics.py']},'external_test_evaluated':False});print(summary,flush=True)


if __name__=='__main__':main()

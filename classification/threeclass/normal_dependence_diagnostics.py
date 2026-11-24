"""Train-only patient-level conditional dependence on removed normal axes."""
import numpy as np
import torch
from experiment import ROOT,KEYS,load_data,normalize,dump
from normal_subspace import NormalSuppressedModel,fit_normal_geometry
from audit_normal_coordinate import numpy_coordinates
from kernel_residual_experiments import sha


def score(x,h,groups):
    values=[]
    for group in sorted(set(groups)):
        ids=np.flatnonzero(groups==group)
        if len(ids)<4:continue
        a=x[ids]-x[ids].mean(0);b=h[ids]-h[ids].mean(0)
        cross=a.T@b/len(ids);vx=np.mean(a*a,0);vh=np.mean(b*b,0)
        r=cross/np.sqrt(vx[:,None]*vh[None,:]+1e-4)
        values.append(float(np.mean(r*r)))
    return float(np.mean(values)),len(values)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[fit].reset_index(drop=True)
    for v in features.values():v[2091:]=np.nan
    source=ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt';c=torch.load(source,map_location='cpu',weights_only=False)
    normalized,_=normalize({k:v[fit] for k,v in features.items()},None,c['stats']);geometry=fit_normal_geometry(tf,normalized)
    model=NormalSuppressedModel(c['cfg'],c['dims'],len(c['diseases']));model.load_state_dict(c['state_dict']);model.eval()
    state=dict(c['state_dict'])
    for key in KEYS:state['coordinate_deviation_'+key]=torch.from_numpy(np.maximum(np.sqrt(geometry[key+'_eigenvalues']),.05))
    coordinates=numpy_coordinates(normalized,state);hidden=[]
    for begin in range(0,len(fit),256):hidden.append(model({k:torch.from_numpy(v[begin:begin+256]) for k,v in normalized.items()})['embedding'].numpy())
    hidden=np.concatenate(hidden);x=[];h=[];coarse=[];fine=[]
    for (patient,disease),group in tf.groupby(['patient_group','disease']):
        x.append(coordinates[group.index].mean(0));h.append(hidden[group.index].mean(0));coarse.append(int(group.label.iloc[0]));fine.append(disease)
    x=np.stack(x).astype('float64');h=np.stack(h).astype('float64');results={}
    for scope,groups in [('coarse',np.array(coarse)),('disease',np.array(fine))]:
        observed,count=score(x,h,groups);rng=np.random.default_rng(20282924);null=[]
        for repeat in range(20):
            perm=np.arange(len(x))
            for group in sorted(set(groups)):
                ids=np.flatnonzero(groups==group);perm[ids]=rng.permutation(ids)
            null.append(score(x[perm],h,groups)[0])
        results[scope]={'observed_mean_squared_conditional_correlation':observed,'within_stratum_permutation_mean':float(np.mean(null)),'within_stratum_permutation_max':float(np.max(null)),'strata_with_at_least_four_patient_disease_groups':count}
    dump(ROOT/'results/normal_dependence_diagnostics.json',{'manifest_sha256':digest,'source_sha256':sha(source),'patient_disease_centers':len(x),'scope':'Original training split only,trained normal-reference model; descriptive in-sample dependence,not causal or performance proof.20 fixed within-stratum patient permutations.0 new models.','results':results,'code_sha256':sha(ROOT/'normal_dependence_diagnostics.py'),'external_test_evaluated':False})
    print(results,flush=True)


if __name__=='__main__':main()

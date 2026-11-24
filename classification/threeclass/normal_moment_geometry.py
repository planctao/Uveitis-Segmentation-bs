"""Patient-balanced normal-image and within-normal-patient nuisance moments."""
import numpy as np
from experiment import KEYS


def moment_rows(frame,normalized,key,mode):
    assert mode in ['images','within_patient']
    normal=frame.label.eq(2).to_numpy();patients=frame.patient_group.to_numpy()
    names=sorted(set(patients[normal]));x=np.asarray(normalized[key],dtype='float64')
    if x.ndim==3:x=x.mean(1)
    patient_means={name:x[normal&(patients==name)].mean(0) for name in names}
    center=np.mean(list(patient_means.values()),axis=0)
    rows=[]
    for name in names:
        values=x[normal&(patients==name)]
        residual=values-(center if mode=='images' else patient_means[name])
        rows.append(residual/np.sqrt(len(names)*len(values)))
    return names,center,np.concatenate(rows)


def fit_moment_geometry(frame,normalized,mode,rank=32):
    geometry={}
    for key in KEYS:
        names,center,rows=moment_rows(frame,normalized,key,mode)
        _,singular,right=np.linalg.svd(rows,full_matrices=False)
        eigenvalues=singular**2
        usable=int(min(rank,np.sum(eigenvalues>max(float(eigenvalues[0])*1e-6,1e-12))))
        basis=np.zeros((rows.shape[-1],rank),dtype='float64')
        if usable:
            basis[:,:usable]=right[:usable].T
            signs=np.sign(basis[np.abs(basis[:,:usable]).argmax(0),np.arange(usable)])
            basis[:,:usable]*=np.where(signs==0,1,signs)
        spectrum=np.zeros(rank,dtype='float32');spectrum[:usable]=eigenvalues[:usable]
        geometry[key+'_center']=center.astype('float32')
        geometry[key+'_basis']=basis.astype('float32')
        geometry[key+'_eigenvalues']=spectrum
        geometry[key+'_effective_rank']=np.array(usable,dtype='int64')
    geometry['normal_patients']=np.asarray(names,dtype=str)
    geometry['moment_mode']=np.array(mode)
    return geometry


def independent_moment_geometry(frame,normalized,geometry):
    mode=str(geometry['moment_mode']);normal=frame.label.eq(2).to_numpy()
    names=sorted(frame.loc[normal,'patient_group'].unique());assert np.array_equal(names,geometry['normal_patients'])
    errors={}
    for key in KEYS:
        x=np.asarray(normalized[key],dtype='float64')
        if x.ndim==3:x=np.sum(x,axis=1)/x.shape[1]
        groups=[np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy()) for name in names]
        means=[np.sum(x[ids],axis=0)/len(ids) for ids in groups];center=np.sum(means,axis=0)/len(means)
        weighted=[]
        for index,ids in enumerate(groups):
            target=center if mode=='images' else means[index]
            for row in ids:weighted.append((x[row]-target)/(len(groups)*len(ids))**.5)
        a=np.stack(weighted);eigenvalues,left=np.linalg.eigh(a@a.T)
        order=np.argsort(eigenvalues)[::-1];values=np.maximum(eigenvalues[order],0)
        rank=int(min(32,np.sum(values>max(float(values[0])*1e-6,1e-12))))
        assert rank==int(geometry[key+'_effective_rank'])
        basis=geometry[key+'_basis'].astype('float64')
        if rank:
            independent=a.T@left[:,order[:rank]]/np.sqrt(values[:rank])[None]
            angles=np.linalg.svd(independent.T@basis[:,:rank],compute_uv=False)
            assert np.allclose(angles,1,atol=2e-6)
            assert np.allclose(basis[:,:rank].T@basis[:,:rank],np.eye(rank),atol=1e-6)
        else:angles=np.array([1.])
        assert not basis[:,rank:].any() and not geometry[key+'_eigenvalues'][rank:].any()
        assert np.allclose(values[:rank],geometry[key+'_eigenvalues'][:rank],atol=2e-5,rtol=2e-6)
        assert np.allclose(center,geometry[key+'_center'],atol=1e-6)
        errors[key]={'effective_rank':rank,'normal_patients':len(names),
            'maximum_principal_cosine_error':float(np.max(np.abs(angles-1))),
            'weighted_covariance_trace':float(np.square(a).sum())}
    return errors

"""Fit-only normal-patient nuisance directions with fixed low-rank suppression."""
import numpy as np
import torch
from experiment import Model,KEYS


def patient_centers(frame,normalized,key):
    normal=frame.label.eq(2).to_numpy();patients=np.asarray(frame.patient_group)
    names=sorted(set(patients[normal]));x=np.asarray(normalized[key],dtype='float64')
    if x.ndim==3:x=x.mean(1)
    centers=np.stack([x[normal&(patients==name)].mean(0) for name in names])
    return names,centers


def fit_normal_geometry(frame,normalized,rank=32):
    assert len(frame)==len(normalized['global'])
    geometry={}
    for key in KEYS:
        names,centers=patient_centers(frame,normalized,key)
        assert len(names)>rank
        mean=centers.mean(0);centered=centers-mean
        _,singular,right=np.linalg.svd(centered,full_matrices=False)
        basis=right[:rank].T
        signs=np.sign(basis[np.abs(basis).argmax(0),np.arange(rank)])
        basis*=np.where(signs==0,1,signs)
        geometry[key+'_center']=mean.astype('float32')
        geometry[key+'_basis']=basis.astype('float32')
        geometry[key+'_eigenvalues']=(singular[:rank]**2/(len(names)-1)).astype('float32')
    geometry['normal_patients']=np.asarray(names,dtype=str)
    return geometry


def independent_transform(values,center,basis,strength):
    if strength==0:return np.array(values,copy=True)
    x=np.asarray(values,dtype='float64');b=np.asarray(basis,dtype='float64')
    coordinates=np.einsum('...d,dk->...k',x-np.asarray(center,dtype='float64'),b)
    correction=np.einsum('...k,dk->...d',coordinates,b)
    return np.clip(x-strength*correction,-10,10).astype('float32')


class NormalSuppressedModel(Model):
    def __init__(self,cfg,dims,n_diseases,geometry=None):
        super().__init__(cfg,dims,n_diseases)
        rank=cfg['normal_rank'];assert rank in [8,32] and cfg['normal_strength'] in [0.,.5,1.]
        for key in KEYS:
            self.register_buffer('normal_center_'+key,torch.zeros(dims[key]))
            self.register_buffer('normal_basis_'+key,torch.zeros(dims[key],rank))
        if geometry is not None:
            for key in KEYS:
                getattr(self,'normal_center_'+key).copy_(torch.as_tensor(geometry[key+'_center']))
                getattr(self,'normal_basis_'+key).copy_(torch.as_tensor(geometry[key+'_basis'][:,:rank]))

    def transform(self,x):
        alpha=self.cfg['normal_strength']
        if alpha==0:return x
        transformed={}
        for key,value in x.items():
            center=getattr(self,'normal_center_'+key);basis=getattr(self,'normal_basis_'+key)
            transformed[key]=(value-alpha*((value-center)@basis)@basis.T).clamp(-10,10)
        return transformed

    def forward(self,x):
        return super().forward(self.transform(x))

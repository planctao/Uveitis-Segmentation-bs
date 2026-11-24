"""Conditional mixtures of normal-patient tangent spaces; no query labels."""
import numpy as np
import torch
from experiment import KEYS,Model
from normal_subspace import patient_centers,fit_normal_geometry


def distances(points,centers):
    return np.square(points[:,None]-centers[None]).mean(-1)


def cluster_normals(points,count):
    selected=[int(np.argmax(np.square(points).sum(1)))]
    while len(selected)<count:selected.append(int(np.argmax(distances(points,points[selected]).min(1))))
    centers=points[selected].copy();old=None
    for iteration in range(100):
        assignment=distances(points,centers).argmin(1)
        assert set(assignment)==set(range(count))
        if old is not None and np.array_equal(assignment,old):break
        centers=np.stack([points[assignment==j].mean(0) for j in range(count)]);old=assignment.copy()
    else:raise RuntimeError('Normal atlas clustering did not converge')
    return centers,assignment,iteration+1


def numpy_gate(global_values,geometry,temperature):
    x=global_values.astype('float64');basis=geometry['gate_basis'].astype('float64')
    points=((x-geometry['gate_center'])@basis)/geometry['gate_deviation']
    squared=distances(points,geometry['gate_clusters'].astype('float64'))
    logits=-squared/(temperature*float(geometry['gate_scale']))
    logits-=logits.max(1,keepdims=True);weights=np.exp(logits);weights/=weights.sum(1,keepdims=True)
    return weights,points


def fit_atlas_geometry(frame,normalized,count,temperature):
    assert count in [2,4] and temperature in [.5,2.]
    reference=fit_normal_geometry(frame,normalized);geometry={}
    for key in ['center','basis','eigenvalues']:geometry['gate_'+key]=reference['global_'+key].copy()
    geometry['gate_deviation']=np.sqrt(reference['global_eigenvalues']).astype('float32')
    names,global_centers=patient_centers(frame,normalized,'global')
    points=((global_centers-geometry['gate_center'])@geometry['gate_basis'])/geometry['gate_deviation']
    clusters,assignment,iterations=cluster_normals(points,count)
    geometry['gate_clusters']=clusters.astype('float32')
    geometry['gate_scale']=np.asarray(max(float(distances(points,clusters).min(1).mean()),1e-6),dtype='float32')
    responsibility,_=numpy_gate(global_centers,geometry,temperature)
    # Positive global shrinkage keeps the local covariance full enough for the same rank32.
    weights=.25/len(names)+.75*responsibility/responsibility.sum(0,keepdims=True)
    assert np.allclose(weights.sum(0),1)
    for key in KEYS:
        actual,centers=patient_centers(frame,normalized,key);assert actual==names
        local_centers=[];local_bases=[];local_eigen=[]
        for j in range(count):
            w=weights[:,j];mean=np.sum(centers*w[:,None],axis=0)
            rows=(centers-mean)*np.sqrt(w)[:,None];_,singular,right=np.linalg.svd(rows,full_matrices=False)
            basis=right[:32].T;sign=np.sign(basis[np.abs(basis).argmax(0),np.arange(32)])
            basis*=np.where(sign==0,1,sign)
            local_centers.append(mean);local_bases.append(basis)
            local_eigen.append(singular[:32]**2/(1-np.square(w).sum()))
        geometry['local_center_'+key]=np.asarray(local_centers,dtype='float32')
        geometry['local_basis_'+key]=np.asarray(local_bases,dtype='float32')
        geometry['local_eigen_'+key]=np.asarray(local_eigen,dtype='float32')
    geometry['normal_patients']=np.asarray(names,dtype=str)
    geometry['patient_gate_points']=points.astype('float32')
    geometry['patient_assignment']=assignment.astype('int64')
    geometry['patient_responsibility']=responsibility.astype('float32')
    geometry['patient_local_weights']=weights.astype('float32')
    geometry['cluster_iterations']=np.asarray(iterations,dtype='int64')
    return geometry


def independent_transform(values,geometry,temperature,strength=1.):
    if strength==0:return {k:v.copy() for k,v in values.items()}
    weights,_=numpy_gate(values['global'],geometry,temperature);result={}
    for key in KEYS:
        x=values[key].astype('float64');correction=np.zeros_like(x)
        for cluster in range(weights.shape[1]):
            b=geometry['local_basis_'+key][cluster].astype('float64');center=geometry['local_center_'+key][cluster].astype('float64')
            delta=((x-center)@b)@b.T
            correction+=delta*weights[:,cluster].reshape((-1,)+(1,)*(x.ndim-1))
        result[key]=np.clip(x-strength*correction,-10,10).astype('float32')
    return result


def independent_geometry(frame,normalized,geometry,count,temperature):
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    assert np.array_equal(names,geometry['normal_patients']);centers={}
    for key in KEYS:
        x=normalized[key].astype('float64')
        if x.ndim==3:x=np.sum(x,axis=1)/x.shape[1]
        centers[key]=np.stack([np.sum(x[normal&frame.patient_group.eq(n).to_numpy()],0)/np.sum(normal&frame.patient_group.eq(n).to_numpy()) for n in names])
    mean=centers['global'].mean(0);rows=(centers['global']-mean)/np.sqrt(len(names)-1)
    eigen,left=np.linalg.eigh(rows@rows.T);order=np.argsort(eigen)[::-1]
    b=rows.T@left[:,order[:32]]/np.sqrt(eigen[order[:32]])[None]
    assert np.allclose(mean,geometry['gate_center'],atol=1e-6,rtol=1e-6)
    assert np.allclose(np.linalg.svd(b.T@geometry['gate_basis'],compute_uv=False),1,atol=2e-6)
    assert np.allclose(eigen[order[:32]],geometry['gate_eigenvalues'],atol=2e-5,rtol=2e-6)
    assert np.array_equal(geometry['gate_deviation'],np.sqrt(geometry['gate_eigenvalues']))
    responsibility,points=numpy_gate(centers['global'],geometry,temperature)
    assert np.allclose(points,geometry['patient_gate_points'],atol=2e-6,rtol=2e-6)
    assignment=np.array([np.argmin([np.square(p-center).sum() for center in geometry['gate_clusters']]) for p in points])
    assert np.array_equal(assignment,geometry['patient_assignment'])
    for cluster in range(count):
        assert np.allclose(points[assignment==cluster].mean(0),geometry['gate_clusters'][cluster],atol=2e-6,rtol=2e-6)
    scale=np.mean([min(np.square(p-c).mean() for c in geometry['gate_clusters']) for p in points])
    assert np.isclose(scale,geometry['gate_scale'],atol=1e-7,rtol=2e-6)
    assert np.allclose(responsibility,geometry['patient_responsibility'],atol=1e-6,rtol=1e-6)
    weights=.25/len(names)+.75*responsibility/responsibility.sum(0,keepdims=True)
    assert np.allclose(weights,geometry['patient_local_weights'],atol=1e-7,rtol=1e-6)
    errors=[]
    for key in KEYS:
        for cluster in range(count):
            w=weights[:,cluster];mean=np.sum(centers[key]*w[:,None],0);rows=(centers[key]-mean)*np.sqrt(w)[:,None]
            eigen,left=np.linalg.eigh(rows@rows.T);order=np.argsort(eigen)[::-1]
            recovered=rows.T@left[:,order[:32]]/np.sqrt(eigen[order[:32]])[None]
            basis=geometry['local_basis_'+key][cluster].astype('float64')
            cosine=np.linalg.svd(recovered.T@basis,compute_uv=False)
            assert np.allclose(mean,geometry['local_center_'+key][cluster],atol=1e-6,rtol=1e-6)
            assert np.allclose(cosine,1,atol=2e-6)
            assert np.allclose(basis.T@basis,np.eye(32),atol=1e-6)
            assert np.allclose(eigen[order[:32]]/(1-np.square(w).sum()),geometry['local_eigen_'+key][cluster],atol=2e-5,rtol=2e-6)
            errors.append(float(np.max(np.abs(cosine-1))))
    return {'normal_patients':len(names),'cluster_counts':np.bincount(assignment,minlength=count).tolist(),
        'maximum_principal_cosine_error':max(errors),'mean_max_gate_weight':float(responsibility.max(1).mean()),
        'mean_gate_entropy':float(-(responsibility*np.log(np.maximum(responsibility,1e-15))).sum(1).mean())}


class NormalAtlasModel(Model):
    def __init__(self,cfg,dims,n_diseases,geometry=None):
        super().__init__(cfg,dims,n_diseases)
        count=cfg['normal_atlas_count'];assert count in [2,4] and cfg['normal_atlas_temperature'] in [.5,2.]
        assert cfg['normal_atlas_strength'] in [0.,1.]
        self.register_buffer('atlas_gate_center',torch.zeros(dims['global']))
        self.register_buffer('atlas_gate_basis',torch.zeros(dims['global'],32))
        self.register_buffer('atlas_gate_deviation',torch.ones(32))
        self.register_buffer('atlas_gate_clusters',torch.zeros(count,32))
        self.register_buffer('atlas_gate_scale',torch.tensor(1.))
        for key in KEYS:
            self.register_buffer('atlas_local_center_'+key,torch.zeros(count,dims[key]))
            self.register_buffer('atlas_local_basis_'+key,torch.zeros(count,dims[key],32))
        if geometry is not None:
            for name,value in self.named_buffers():
                if name.startswith('atlas_'):value.copy_(torch.as_tensor(geometry[name[len('atlas_'):]]))

    def transform(self,x):
        points=((x['global']-self.atlas_gate_center)@self.atlas_gate_basis)/self.atlas_gate_deviation
        distance=(points[:,None]-self.atlas_gate_clusters).square().mean(-1)
        self._weights=(-distance/(self.cfg['normal_atlas_temperature']*self.atlas_gate_scale)).softmax(-1)
        if self.cfg['normal_atlas_strength']==0:return x
        result={}
        for key in KEYS:
            value=x[key];v=value[:,None] if value.ndim==2 else value
            b=getattr(self,'atlas_local_basis_'+key);center=getattr(self,'atlas_local_center_'+key)
            coordinates=torch.einsum('btd,kdr->btkr',v,b)-torch.einsum('kd,kdr->kr',center,b)[None,None]
            correction=torch.einsum('btkr,kdr,bk->btd',coordinates,b,self._weights)
            adjusted=(v-self.cfg['normal_atlas_strength']*correction).clamp(-10,10)
            result[key]=adjusted[:,0] if value.ndim==2 else adjusted
        return result

    def forward(self,x):
        out=super().forward(self.transform(x));out['normal_atlas_weights']=self._weights
        return out

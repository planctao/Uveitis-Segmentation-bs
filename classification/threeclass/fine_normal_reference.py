"""Fine foreground region attention with fit-only normal reference geometry."""
import numpy as np
from experiment import KEYS
from normal_subspace import NormalSuppressedModel,fit_normal_geometry


def normal_summaries(normalized):
    result={k:normalized[k] for k in KEYS}
    if 'region_mask' in normalized:
        valid=normalized['region_mask']>.5
        assert valid.any(1).all()
        regions=normalized['regions'].astype('float64')
        summary=(regions*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        result['regions']=summary[:,None]
    return result


def fit_fine_geometry(frame, normalized):
    return fit_normal_geometry(frame,normal_summaries(normalized))


def independent_fine_geometry(frame, normalized, geometry):
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    assert np.array_equal(names,geometry['normal_patients']);diagnostics={}
    for key in KEYS:
        values=normalized[key].astype('float64')
        if key=='regions':
            rows=[]
            for i in range(len(frame)):
                ids=np.flatnonzero(normalized['region_mask'][i]>.5) if 'region_mask' in normalized else np.arange(values.shape[1])
                assert len(ids);rows.append(np.sum(values[i,ids],0)/len(ids))
            values=np.stack(rows)
        centers=[]
        for name in names:
            ids=np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy())
            centers.append(np.sum(values[ids],0)/len(ids))
        centers=np.stack(centers);mean=np.sum(centers,0)/len(names);rows=(centers-mean)/np.sqrt(len(names)-1)
        assert np.allclose(mean,geometry[key+'_center'],atol=1e-6,rtol=1e-6)
        values,left=np.linalg.eigh(rows@rows.T);order=np.argsort(values)[::-1];values=values[order]
        recovered=rows.T@left[:,order[:32]]/np.sqrt(values[:32])[None]
        basis=geometry[key+'_basis'].astype('float64')
        cosine=np.linalg.svd(recovered.T@basis,compute_uv=False)
        assert np.allclose(cosine,1,atol=2e-6)
        assert np.allclose(basis.T@basis,np.eye(32),atol=1e-6)
        assert np.allclose(values[:32],geometry[key+'_eigenvalues'],atol=2e-5,rtol=2e-6)
        diagnostics[key]={'patients':len(names),'maximum_principal_cosine_error':float(np.max(np.abs(cosine-1)))}
    return diagnostics


class FineNormalModel(NormalSuppressedModel):
    def transform(self,x):
        result=super().transform({k:x[k] for k in KEYS})
        if 'region_mask' in x:result['region_mask']=x['region_mask']
        return result

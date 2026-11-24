"""Patient-balanced normal residual energy as a fixed regional attention prior."""
import numpy as np
import torch
from experiment import Model,KEYS
from normal_subspace import NormalSuppressedModel,fit_normal_geometry,independent_transform


def fit_evidence_geometry(frame,normalized,scope):
    assert scope in ['pooled','position']
    geometry=fit_normal_geometry(frame,normalized)
    normal=frame.label.eq(2).to_numpy()
    regions=independent_transform(normalized['regions'][normal],geometry['regions_center'],geometry['regions_basis'],1.).astype('float64')
    patients=frame.loc[normal,'patient_group'].to_numpy();names=geometry['normal_patients']
    first=[];second=[]
    for name in names:
        values=regions[patients==name]
        axes=(0,1) if scope=='pooled' else 0
        first.append(values.mean(axes));second.append(np.square(values).mean(axes))
    mean=np.mean(first,axis=0);variance=np.mean(second,axis=0)-mean**2
    if scope=='pooled':mean=mean[None];variance=variance[None]
    geometry['evidence_center']=mean.astype('float32')
    geometry['evidence_scale']=np.sqrt(np.maximum(variance,.05**2)).astype('float32')
    return geometry


def independent_geometry(frame,normalized,geometry,scope):
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    assert np.array_equal(names,geometry['normal_patients']);diagnostics={}
    for key in KEYS:
        x=normalized[key].astype('float64')
        if x.ndim==3:x=np.sum(x,axis=1)/x.shape[1]
        centers=np.stack([np.sum(x[normal&frame.patient_group.eq(n).to_numpy()],axis=0)/np.sum(normal&frame.patient_group.eq(n).to_numpy()) for n in names])
        mean=np.sum(centers,axis=0)/len(names);rows=(centers-mean)/np.sqrt(len(names)-1)
        eigen,left=np.linalg.eigh(rows@rows.T);order=np.argsort(eigen)[::-1]
        recovered=rows.T@left[:,order[:32]]/np.sqrt(eigen[order[:32]])[None]
        basis=geometry[key+'_basis'].astype('float64');cosine=np.linalg.svd(recovered.T@basis,compute_uv=False)
        assert np.allclose(mean,geometry[key+'_center'],atol=1e-6,rtol=1e-6)
        assert np.allclose(cosine,1,atol=2e-6)
        assert np.allclose(basis.T@basis,np.eye(32),atol=1e-6)
        assert np.allclose(eigen[order[:32]],geometry[key+'_eigenvalues'],atol=2e-5,rtol=2e-6)
        diagnostics[key]={'patients':len(names),'maximum_principal_cosine_error':float(np.max(np.abs(cosine-1)))}
    # Independent explicit patient loops and centered second pass, rather than E[x^2]-E[x]^2.
    moments=[];regions=normalized['regions'].astype('float64');b=geometry['regions_basis'].astype('float64');center=geometry['regions_center'].astype('float64')
    for name in names:
        ids=np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy());rows=[]
        for index in ids:
            image=regions[index];corrected=np.clip(image-((image-center)@b)@b.T,-10,10).astype('float32').astype('float64')
            rows.append(corrected)
        moments.append(np.stack(rows))
    mean=np.zeros_like(geometry['evidence_center'],dtype='float64')
    for patient in moments:
        mean+=patient.mean((0,1),keepdims=False)[None]/len(names) if scope=='pooled' else patient.mean(0)/len(names)
    variance=np.zeros_like(mean)
    for patient in moments:
        squared=np.square(patient-mean)
        variance+=squared.mean((0,1))[None]/len(names) if scope=='pooled' else squared.mean(0)/len(names)
    scale=np.sqrt(np.maximum(variance,.05**2))
    assert np.allclose(mean,geometry['evidence_center'],atol=1e-6,rtol=1e-6)
    assert np.allclose(scale,geometry['evidence_scale'],atol=1e-6,rtol=1e-6)
    diagnostics['evidence']={'scope':scope,'minimum_scale':float(scale.min()),'maximum_scale':float(scale.max())}
    return diagnostics


def numpy_bias(regions,center,scale,strength):
    standardized=(regions.astype('float64')-center.astype('float64'))/scale.astype('float64')
    energy=np.sqrt(np.mean(np.square(standardized),axis=-1)+1e-12)
    raw=strength*np.tanh(np.log(energy+1e-6))
    return energy.astype('float32'),(raw-raw.mean(1,keepdims=True)).astype('float32')


class NormalEvidenceModel(NormalSuppressedModel):
    def __init__(self,cfg,dims,n_diseases,geometry=None):
        super().__init__(cfg,dims,n_diseases,geometry)
        assert cfg['normal_strength']==1. and cfg['normal_rank']==32
        assert cfg['normal_evidence_scope'] in ['pooled','position']
        assert cfg['normal_evidence_strength'] in [0.,.5,2.]
        count=1 if cfg['normal_evidence_scope']=='pooled' else 9
        self.register_buffer('evidence_center',torch.zeros(count,dims['regions']))
        self.register_buffer('evidence_scale',torch.ones(count,dims['regions']))
        if geometry is not None:
            self.evidence_center.copy_(torch.as_tensor(geometry['evidence_center']))
            self.evidence_scale.copy_(torch.as_tensor(geometry['evidence_scale']))
        self.region_gate.register_forward_hook(self._add_evidence)

    def _add_evidence(self,module,args,output):
        self._unbiased_scores=output
        scores=output+self._bias[:,:,None]
        self._attention=scores.softmax(1)
        return scores

    def forward(self,x):
        transformed=self.transform(x)
        standardized=(transformed['regions']-self.evidence_center)/self.evidence_scale
        self._energy=(standardized.square().mean(-1)+1e-12).sqrt()
        raw=self.cfg['normal_evidence_strength']*(self._energy+1e-6).log().tanh()
        self._bias=raw-raw.mean(1,keepdim=True)
        out=Model.forward(self,transformed)
        out.update(normal_region_energy=self._energy,normal_attention_bias=self._bias,
                   normal_region_attention=self._attention,normal_unbiased_scores=self._unbiased_scores)
        return out

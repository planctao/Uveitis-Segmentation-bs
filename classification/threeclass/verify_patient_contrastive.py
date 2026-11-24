"""Independent masked/centroid contrastive geometry and legacy-loss checks."""
import json
import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn.functional as F
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from patient_contrastive import patient_contrastive_loss


def independent(embedding,target,patient,weight,mode):
    unit=embedding/np.linalg.norm(embedding,axis=1,keepdims=True)
    if mode=='patient_centroid':
        keys=sorted(set(zip(patient.tolist(),target.tolist())));inverse=np.array([keys.index((p,t)) for p,t in zip(patient,target)])
        candidates=np.stack([unit[inverse==k].mean(0) for k in range(len(keys))]);candidates/=np.linalg.norm(candidates,axis=1,keepdims=True)
        candidate_patient,candidate_target=np.array(keys).T
    else:
        candidates=unit;candidate_patient,candidate_target=patient,target;inverse=np.arange(len(target))
    anchor_loss=[];valid=[]
    for k in range(len(candidates)):
        allowed=np.flatnonzero((candidate_patient!=candidate_patient[k]) & (candidate_target>=0) & (candidate_target[k]>=0))
        positives=allowed[candidate_target[allowed]==candidate_target[k]]
        valid.append(len(positives)>0)
        if len(positives):
            scores=candidates[allowed]@candidates[k]/.15
            positive_scores=candidates[positives]@candidates[k]/.15
            anchor_loss.append(logsumexp(scores)-positive_scores.mean())
        else:anchor_loss.append(0.)
    valid=np.array(valid)[inverse]
    return np.mean(np.array(anchor_loss)[inverse][valid]*weight[valid]) if valid.any() else 0.


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    embedding=torch.randn(9,5,dtype=torch.float64,requires_grad=True)
    patient=torch.tensor([10,10,20,20,30,30,40,40,50]);target=torch.tensor([0,0,0,0,1,1,1,1,-1])
    weight=torch.linspace(.5,1.5,9,dtype=torch.float64)
    for mode in ['exclude_patient','patient_centroid']:
        actual=patient_contrastive_loss(embedding,target,patient,weight,mode)
        expected=independent(embedding.detach().numpy(),target.numpy(),patient.numpy(),weight.numpy(),mode)
        assert np.isclose(float(actual.detach()),expected,atol=1e-12)
        assert torch.autograd.gradcheck(lambda z:patient_contrastive_loss(z,target,patient,weight,mode),(embedding,),atol=1e-5)
        permutation=torch.tensor([8,3,0,5,2,4,7,6,1])
        assert torch.allclose(actual,patient_contrastive_loss(embedding[permutation],target[permutation],patient[permutation],weight[permutation],mode),atol=1e-12)
        unknown=embedding.detach().clone();unknown[-1]=torch.tensor([100.,-5.,7.,12.,1000.])
        assert torch.allclose(actual,patient_contrastive_loss(unknown,target,patient,weight,mode),atol=1e-12)
        no_positive=patient_contrastive_loss(embedding[:2],torch.tensor([0,1]),torch.tensor([10,20]),weight[:2],mode)
        assert float(no_positive.detach())==0
        gradient=torch.autograd.grad(no_positive,embedding,retain_graph=True)[0];assert not gradient.count_nonzero()
    repeated=torch.arange(9).repeat_interleave(2)
    assert torch.allclose(patient_contrastive_loss(embedding,target,patient,weight,'patient_centroid'),
        patient_contrastive_loss(embedding[repeated],target[repeated],patient[repeated],weight[repeated],'patient_centroid'),atol=1e-12)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');fit=np.flatnonzero(frame.split.eq('train'))
    indices=np.random.default_rng(20261124).choice(fit,256,replace=False)
    x,_=normalize({k:v[indices] for k,v in features.items()},None,c['stats']);inputs={k:torch.from_numpy(v) for k,v in x.items()}
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True);out=model(inputs)
    y=torch.tensor(frame.iloc[indices].label.to_numpy().copy()).long()
    disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[indices].disease])
    _,ids=np.unique(frame.iloc[indices].patient_group,return_inverse=True)
    weights={'leaf':torch.ones(len(indices)),'patient_ids':torch.from_numpy(ids)}
    unit=F.normalize(out['fine_embedding'],dim=-1);sim=unit@unit.T/.15
    identity=torch.eye(len(indices),dtype=torch.bool);legacy_scores=sim.masked_fill(identity,-1e4)
    same_patient=weights['patient_ids'][:,None]==weights['patient_ids'][None,:]
    positive=(disease[:,None]==disease[None,:])&~same_patient;count=positive.sum(1);valid=count>0
    manual=-(F.log_softmax(legacy_scores,dim=1)*positive).sum(1)/count.clamp(min=1)
    base=loss_fn(out,y,disease,weights,{**c['cfg'],'fine_contrastive':0.})
    legacy=loss_fn(out,y,disease,weights,c['cfg'])
    assert torch.allclose(legacy,base+c['cfg']['fine_contrastive']*manual[valid].mean(),atol=1e-7)
    for mode in ['exclude_patient','patient_centroid']:
        cfg={**c['cfg'],'patient_contrastive_mode':mode}
        candidate=Model(cfg,c['dims'],len(c['diseases'])).eval();candidate.load_state_dict(c['state_dict'],strict=True)
        for key,value in model.state_dict().items():assert torch.equal(candidate.state_dict()[key],value)
        assert torch.allclose(candidate(inputs)['p'],out['p'],atol=1e-7,rtol=1e-7)
        actual=loss_fn(out,y,disease,weights,cfg)
        expected=base+c['cfg']['fine_contrastive']*patient_contrastive_loss(out['fine_embedding'],disease,weights['patient_ids'],weights['leaf'],mode)
        assert torch.allclose(actual,expected,atol=1e-7)
        gradient=torch.autograd.grad(actual,list(model.fine_projection.parameters()),retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gradient) and sum(g.abs().sum() for g in gradient)>0
    diagnostics={'sample':'Fixed256originaltrainingimages, descriptive mechanism check only',
        'same_patient_ordered_pairs_excluding_self':int((same_patient&~identity).sum()),
        'legacy_softmax_mass_on_same_patient_mean':float((legacy_scores.softmax(1)*(same_patient&~identity)).sum(1)[valid].mean().detach()),
        'legacy_fine_loss':float(manual[valid].mean().detach()),
        'excluded_patient_fine_loss':float(patient_contrastive_loss(out['fine_embedding'],disease,weights['patient_ids'],weights['leaf'],'exclude_patient').detach()),
        'patient_centroid_fine_loss':float(patient_contrastive_loss(out['fine_embedding'],disease,weights['patient_ids'],weights['leaf'],'patient_centroid').detach())}
    dump(ROOT/'results/research_patient_contrastive/mechanism_diagnostics.json',diagnostics)
    dump(ROOT/'verification_patient_contrastive.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent NumPy candidateconstruction and eligible-only logsumexp match bothlosses',
        'Doubleprecision numericalgradient checks pass for both modes',
        'Image permutation invariance, unknownlabels excluded, absentpositives give differentiablezero',
        'Repeatingallimages preserves patientcentroidloss',
        'OriginalM5loss exactly reconstructed and inferenceparameters/predictions unchanged',
        'Complete classifierloss uses declared contrastive term and gives finite nonzero projectiongradients']})
    print(json.dumps({'verification':'passed','diagnostics':diagnostics}),flush=True)


if __name__=='__main__':main()

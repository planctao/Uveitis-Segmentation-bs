"""Independent patient geometry, parent support, soft-target objective and gradients."""
import json
import numpy as np
from scipy.special import logsumexp
import torch
from experiment import ROOT,load_data,dump
from taxonomy_smoothing import build_transition,transport_penalty


def independent_transition(frame,values,diseases,mode):
    rows=frame.reset_index(drop=True);z=values.astype('float64');z/=np.sqrt((z*z).sum(1,keepdims=True)).clip(1e-12)
    table={}
    for i,row in enumerate(rows.itertuples()):table.setdefault((row.disease,row.patient_group),[]).append(z[i])
    centers=[]
    for name in diseases:
        patient=[]
        for (d,p),vectors in table.items():
            if d==name:
                v=np.sum(vectors,axis=0)/len(vectors);patient.append(v/max(np.sqrt(v@v),1e-12))
        c=np.sum(patient,axis=0)/len(patient);centers.append(c/max(np.sqrt(c@c),1e-12))
    centers=np.array(centers);parents=np.array([int(rows.loc[rows.disease.eq(name),'label'].iloc[0]) for name in diseases])
    matrix=np.zeros((len(diseases),len(diseases)))
    for i in range(len(diseases)):
        siblings=[j for j in range(len(diseases)) if parents[j]==parents[i] and j!=i]
        if not siblings:matrix[i,i]=1
        elif mode=='uniform':matrix[i,siblings]=1/len(siblings)
        else:
            logits=np.array([np.dot(centers[i],centers[j])/.2 for j in siblings]);matrix[i,siblings]=np.exp(logits-logsumexp(logits))
    return matrix,centers


def independent_witness(proof):
    logits=proof['logits'].astype('float64');lp=logits-logsumexp(logits,axis=1,keepdims=True)
    d=proof['disease'];valid=d>=0;target=proof['transition'][d.clip(0)]
    rows=-(target*lp).sum(1)+lp[np.arange(len(d)),d.clip(0)]
    penalty=float(np.mean(rows*proof['weight']*valid)) if float(proof['alpha']) else 0.
    total=float(proof['base_loss'])+float(proof['aux'])*float(proof['alpha'])*penalty
    assert np.isclose(penalty,proof['penalty'],atol=3e-6,rtol=3e-6)
    assert np.isclose(total,proof['total_loss'],atol=3e-6,rtol=3e-6)
    return {'penalty':penalty,'total_loss':total,'images':len(d)}


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[fit]
    diseases=sorted(tf.disease.unique());proofs=[]
    for mode in ['uniform','geometry']:
        bank=build_transition(tf,features['global'][fit],diseases,mode)
        expected,centers=independent_transition(tf,features['global'][fit],diseases,mode)
        assert np.allclose(expected,bank['transition'],atol=5e-8,rtol=5e-8)
        assert np.allclose(centers,bank['centers'],atol=1e-12,rtol=1e-12)
        matrix=bank['transition'];parents=bank['parents'];mapping=np.eye(3)[parents]
        assert (matrix>=0).all() and np.allclose(matrix.sum(1),1)
        assert np.allclose(matrix@mapping,mapping) and np.all(matrix[parents[:,None]!=parents[None,:]]==0)
        # Repeating all images of a patient leaves its normalized centroid and total contribution fixed.
        chosen=tf.patient_group.iloc[0];repeat=np.flatnonzero(tf.patient_group.eq(chosen))
        duplicated=tf.iloc[np.r_[np.arange(len(tf)),repeat]]
        again=build_transition(duplicated,features['global'][fit][np.r_[np.arange(len(tf)),repeat]],diseases,mode)
        assert np.allclose(again['transition'],matrix,atol=5e-8,rtol=5e-8)
        torch.manual_seed(71);logits=torch.randn(9,len(diseases),dtype=torch.float64,requires_grad=True)
        disease=torch.tensor([0,1,2,3,4,5,6,7,-1]);weight=torch.linspace(.2,2,9,dtype=torch.float64)
        transition=torch.tensor(matrix,dtype=torch.float64)
        value=transport_penalty(logits,disease,weight,transition);value.backward()
        arrays={'logits':logits.detach().numpy(),'disease':disease.numpy(),'weight':weight.numpy(),
            'transition':matrix,'alpha':.3,'aux':.8,'penalty':float(value.detach()),'base_loss':4.,'total_loss':4.+.24*float(value.detach())}
        independent_witness(arrays)
        for i,j in [(0,0),(4,3),(8,1)]:
            plus=logits.detach().clone();minus=plus.clone();plus[i,j]+=1e-5;minus[i,j]-=1e-5
            numerical=(transport_penalty(plus,disease,weight,transition)-transport_penalty(minus,disease,weight,transition))/2e-5
            assert torch.isclose(logits.grad[i,j],numerical,atol=1e-9,rtol=1e-5)
        proofs.append({'mode':mode,'diseases':len(diseases),'matrix_entropy_mean':float((-(matrix*np.log(matrix.clip(1e-12)))).sum(1).mean())})
    dump(ROOT/'verification_taxonomy_smoothing.json',{'passed':True,'manifest_sha256':digest,'diagnostics':proofs,
        'checks':['Independent patient-normalized DINO centroids and within-parent transitions',
            'Nonnegative stochastic rows, no crossparent mass and exact original coarse target preservation',
            'Repeating onepatientsimages leaves transition unchanged',
            'Independent doubleprecision weighted softtarget formula, ignored unknown labels and finite difference gradients']})
    print('Patient geometry, parent-preserving target transport and objective gradients verified.',flush=True)


if __name__=='__main__':main()

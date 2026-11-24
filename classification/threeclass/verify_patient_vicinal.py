"""Independent vicinal geometry/targets and finite-difference gradients on real input."""
import json
import tempfile
from pathlib import Path
import numpy as np
from scipy.special import ndtr,logsumexp
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump,sample_weights
from patient_vicinal import VicinalModel,vicinal_loss,patient_partners,synthetic_logits


def independent_witness(proof):
    w={k:proof[k] for k in proof};source=w['source'].astype('float64')
    partner=w['partner'];coefficient=w['coefficient'].astype('float64');valid=w['valid']
    assert np.array_equal(valid,partner!=np.arange(len(partner)))
    assert np.all(w['patients'][valid]!=w['patients'][partner[valid]])
    mode=str(w['pair_mode'])
    if mode=='coarse':assert np.all(w['y'][valid]==w['y'][partner[valid]])
    if mode=='disease':assert np.all(w['disease'][valid]==w['disease'][partner[valid]])
    assert np.all(coefficient[~valid]==1) and np.all((coefficient>=0)&(coefficient<=1))
    mixed=coefficient[:,None]*source+(1-coefficient[:,None])*source[partner]
    pre=mixed@w['encoder_weight'].astype('float64').T+w['encoder_bias']
    hidden=pre*ndtr(pre)*w['mask']
    logits=hidden@w['flat_weight'].astype('float64').T+w['flat_bias']
    q=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));p=q@w['mapping']
    errors={}
    for name,expected in [('mixed',mixed),('hidden',hidden),('logits',logits),('p',p)]:
        errors[name]=float(np.max(np.abs(expected-w[name])))
        assert np.allclose(expected,w[name],atol=3e-5,rtol=3e-5),name
    def rows(p,logits,y,disease):
        logp=np.log(np.maximum(p.astype('float64'),1e-7));lp=logits.astype('float64')-logsumexp(logits.astype('float64'),axis=1,keepdims=True)
        smooth=float(w['label_smoothing']);fine=np.where(disease>=0,-lp[np.arange(len(disease)),np.maximum(disease,0)],0)
        return -(1-smooth)*logp[np.arange(len(y)),y]-smooth*logp.mean(1)+float(w['aux'])*fine
    original=np.mean(rows(w['original_p'],w['original_logits'],w['y'],w['disease'])*w['weights'])
    a=rows(p,logits,w['y'],w['disease']);b=rows(p,logits,w['y'][partner],w['disease'][partner])
    synthetic=np.mean(coefficient*w['weights']*a+(1-coefficient)*w['weights'][partner]*b)
    total=float(w['base_loss'])+float(w['strength'])*(synthetic-original)
    assert np.isclose(original,w['original_semantic_loss'],atol=3e-6,rtol=3e-6)
    assert np.isclose(synthetic,w['synthetic_semantic_loss'],atol=3e-6,rtol=3e-6)
    assert np.isclose(total,w['total_loss'],atol=3e-6,rtol=3e-6)
    return {'max_formula_errors':errors,'valid_fraction':float(valid.mean()),
        'cross_class_fraction':float((w['y']!=w['y'][partner]).mean()),
        'cross_disease_fraction':float((w['disease']!=w['disease'][partner]).mean()),
        'mean_coefficient':float(coefficient.mean()),'total_loss':total}


def main():
    torch.set_num_threads(4)
    selected=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(selected['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit')
    ids=np.random.default_rng(20271101).choice(np.flatnonzero(frame.split.eq('train')),256,replace=False)
    data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);x={k:torch.from_numpy(v) for k,v in data.items()}
    tf=frame.iloc[ids];y=torch.tensor(tf.label.to_numpy().copy());disease=torch.tensor([c['diseases'].index(d) for d in tf.disease])
    patients=torch.tensor(__import__('pandas').factorize(tf.patient_group)[0]);weights={'leaf':torch.tensor(sample_weights(tf,tf.label)),'patient_ids':patients}
    diagnostics=[]
    for mode in ['any','coarse','disease']:
        cfg={**c['cfg'],'vicinal_pair_mode':mode,'vicinal_alpha':.4,'vicinal_strength':.5,'seed':0}
        with tempfile.TemporaryDirectory() as temporary:
            model=VicinalModel(cfg,c['dims'],len(c['diseases']),Path(temporary)/'witness.npz');model.load_state_dict(c['state_dict'],strict=True)
            base=Model(cfg,c['dims'],len(c['diseases']));base.load_state_dict(c['state_dict'],strict=True)
            model.train();base.train();torch.manual_seed(42);expected=base(x);rng=torch.get_rng_state().clone()
            torch.manual_seed(42);out=model(x)
            assert torch.equal(expected['p'],out['p']) and torch.equal(rng,torch.get_rng_state())
            loss=vicinal_loss(out,y,disease,weights,cfg);assert torch.equal(rng,torch.get_rng_state())
            loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.region_proj.weight.grad.abs().sum()>0 and model.encoder[0].weight.grad.abs().sum()>0
            detail=independent_witness(np.load(Path(temporary)/'witness.npz'));diagnostics.append({'mode':mode,**detail})
            if mode=='any':assert detail['cross_class_fraction']>.1 and detail['cross_disease_fraction']>.2
            if mode=='coarse':assert detail['cross_class_fraction']==0 and detail['cross_disease_fraction']>.05
            if mode=='disease':assert detail['cross_class_fraction']==detail['cross_disease_fraction']==0
            # Mixed representation gradients reach both endpoints; compare a double-precision finite difference.
            source=torch.randn(4,640,dtype=torch.float64,requires_grad=True)
            small=VicinalModel(cfg,c['dims'],len(c['diseases'])).double()
            partner=torch.tensor([1,0,3,2]);coefficient=torch.tensor([.2,.7,.3,.9],dtype=torch.float64);mask=torch.ones(4,256,dtype=torch.float64)
            value=synthetic_logits(small,source,mask,partner,coefficient)[2].square().mean();value.backward()
            for i,j in [(0,0),(1,13),(3,17)]:
                plus=source.detach().clone();minus=plus.clone();plus[i,j]+=1e-5;minus[i,j]-=1e-5
                a=synthetic_logits(small,plus,mask,partner,coefficient)[2].square().mean()
                b=synthetic_logits(small,minus,mask,partner,coefficient)[2].square().mean()
                assert torch.isclose(source.grad[i,j],(a-b)/2e-5,atol=1e-9,rtol=1e-4)
            model.eval();base.eval();assert torch.equal(model(x)['p'],base(x)['p'])
            # Inference state schema is exactly the original model, including strict restoration after an update.
            torch.optim.AdamW(model.parameters(),lr=.002).step()
            base.load_state_dict(model.state_dict(),strict=True);assert torch.equal(model(x)['p'],base(x)['p'])
    # No eligible patient: identity partner and coefficient 1; pairing uses no global NumPy stream.
    state=np.random.get_state();partner,valid=patient_partners([0,0],[0,0],[1,1],'any',np.random.default_rng(3))
    assert not valid.any() and np.array_equal(partner,[0,1])
    after=np.random.get_state();assert all(np.array_equal(a,b) for a,b in zip(state,after))
    cfg={**c['cfg'],'vicinal_strength':0.,'seed':0}
    base=Model(cfg,c['dims'],len(c['diseases']));model=VicinalModel(cfg,c['dims'],len(c['diseases']))
    base.load_state_dict(c['state_dict']);model.load_state_dict(c['state_dict']);base.train();model.train()
    torch.manual_seed(43);a=base(x);torch.manual_seed(43);b=model(x)
    assert torch.equal(loss_fn(a,y,disease,weights,cfg),vicinal_loss(b,y,disease,weights,cfg))
    dump(ROOT/'verification_patient_vicinal.json',{'passed':True,'manifest_sha256':digest,'real_input_images':len(ids),'diagnostics':diagnostics,
        'checks':['Exact original training/evaluation probabilities and RNG stream; same dropout mask reused for synthetic view',
            'Independent NumPy/SciPy latent interpolation, encoder GELU, disease/coarse targets and endpoint patient weights',
            'Patient exclusion, hierarchy constraints, no eligible partner fallback and local random stream',
            'Finite difference gradients and finite real task backpropagation through both endpoints and original pooling',
            'Zero-strength task exactly original; inference has original parameter schema and strict restoration after update']})
    print('Patient-excluded latent geometry, target weighting, gradients, RNG and restoration verified.',flush=True)


if __name__=='__main__':main()

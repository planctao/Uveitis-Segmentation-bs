"""Independent evidence-gain formula and meaningful real-input/gradient checks."""
import json
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump,sample_weights
from evidence_gating import EvidenceGains,EvidenceGatingModel,LENGTHS


def independent_gains(module,fused):
    x=fused.astype('float64') if module.conditioned else np.zeros_like(fused,dtype='float64')
    state={k:v.detach().cpu().numpy().astype('float64') for k,v in module.state_dict().items()}
    norm=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+module.norm.eps)
    norm=norm*state['norm.weight']+state['norm.bias']
    pre=norm@state['down.weight'].T+state['down.bias']
    raw=(pre*ndtr(pre))@state['readout.weight'].T+state['readout.bias']
    gains=1+.5*np.tanh(raw)
    return fused*np.repeat(gains,LENGTHS,axis=1),gains


def main():
    torch.set_num_threads(4)
    selected=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(selected['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit')
    ids=np.random.default_rng(20271114).choice(np.flatnonzero(frame.split.eq('train')),96,replace=False)
    data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);x={k:torch.from_numpy(v) for k,v in data.items()}
    tf=frame.iloc[ids];y=torch.tensor(tf.label.to_numpy().copy());disease=torch.tensor([c['diseases'].index(d) for d in tf.disease])
    weights={'leaf':torch.tensor(sample_weights(tf,tf.label)),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0])}
    diagnostics=[]
    for rank in [8,32]:
        parameter_counts=[]
        for conditioned in [False,True]:
            cfg={**c['cfg'],'evidence_gain_rank':rank,'evidence_gain_conditioned':conditioned}
            torch.manual_seed(82);base=Model(cfg,c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(82);model=EvidenceGatingModel(cfg,c['dims'],len(c['diseases']))
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            base.eval();model.eval();assert torch.equal(base(x)['p'],model(x)['p'])
            base.train();model.train();torch.manual_seed(83);a=base(x);rng=torch.get_rng_state().clone()
            torch.manual_seed(83);b=model(x);assert torch.equal(a['p'],b['p']) and torch.equal(rng,torch.get_rng_state())
            loss=loss_fn(b,y,disease,weights,cfg);loss.backward()
            assert model.evidence_gains.readout.weight.grad.abs().sum()>0
            torch.optim.AdamW(model.parameters(),lr=.002).step();model.eval();out=model(x)
            expected,gains=independent_gains(model.evidence_gains,out['evidence_gain_input'].detach().numpy())
            assert np.allclose(expected,out['evidence_gain_output'].detach().numpy(),atol=3e-5,rtol=3e-5)
            assert np.allclose(gains,out['evidence_gains'].detach().numpy(),atol=3e-6,rtol=3e-6)
            assert np.all((gains>=.5)&(gains<=1.5))
            if not conditioned:assert np.all(gains==gains[:1])
            else:assert np.max(gains.std(0))>1e-6
            restored=EvidenceGatingModel(cfg,c['dims'],len(c['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
            assert torch.equal(restored(x)['p'],out['p'])
            # Check gain-input derivatives away from zero initialization.
            module=EvidenceGains(rank,conditioned).double();module.load_state_dict(model.evidence_gains.state_dict())
            fused=torch.randn(3,640,dtype=torch.float64,requires_grad=True);value=module(fused)[0].square().mean();value.backward()
            for i,j in [(0,0),(1,270),(2,630)]:
                plus=fused.detach().clone();minus=plus.clone();plus[i,j]+=1e-5;minus[i,j]-=1e-5
                numerical=(module(plus)[0].square().mean()-module(minus)[0].square().mean())/2e-5
                assert torch.isclose(fused.grad[i,j],numerical,atol=1e-8,rtol=1e-5)
            count=sum(p.numel() for p in model.parameters());parameter_counts.append(count)
            diagnostics.append({'rank':rank,'conditioned':conditioned,'parameters':count,'mean_gains':gains.mean(0).tolist()})
        assert parameter_counts[0]==parameter_counts[1]
    dump(ROOT/'verification_evidence_gating.json',{'passed':True,'manifest_sha256':digest,'real_input_images':len(ids),'diagnostics':diagnostics,
        'checks':['Originalinitialcommonweights, RNG and train/evalprobabilities exactly identical',
            'IndependentNumPy/SciPyLayerNorm,GELU,fourboundedgains andgroupmultiplication',
            'Sameparameterconstantcontrol, constantinvariance andnonconstantconditionedgainsafteractualtaskupdate',
            'Finite differencederivatives forinputthroughgainandmultiplication; finitenonzeroreal taskgradients',
            'Strictrestorationafteroptimization, fullM5evidencepreserved withgainsin[0.5,1.5]']})
    print('Evidence gains: independent formula, exact initialization, matched controls and real gradients verified.',flush=True)


if __name__=='__main__':main()

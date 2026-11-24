"""Check symmetry, recoverable scale differences and stable scale interactions."""
import copy
import torch
from experiment import Model,configuration,dump,ROOT


def main():
    torch.set_num_threads(4);torch.manual_seed(7)
    dims={'global':1536,'regions':1536,'mid':1536,'dispersion':1536,'disease_to_class':[0,0,1,1,1,2]}
    x={k:torch.randn(5,9,v) if k=='regions' else torch.randn(5,v) for k,v in dims.items() if k!='disease_to_class'}
    swapped={k:torch.cat(t.chunk(2,dim=-1)[::-1],dim=-1) for k,t in x.items()}
    checks=[]
    for kind in ['difference','product','both']:
        cfg=configuration('scale_interaction','flat',64,taxonomy_only=True,aux=1.,interaction_kind=kind)
        model=Model(cfg,dims,6).eval();out=model(x);other=model(swapped)
        assert torch.allclose(out['p'],other['p'],atol=1e-6)
        assert torch.allclose(out['scale_probability'],other['scale_probability'].flip(1),atol=1e-6)
        assert torch.allclose(out['p'].sum(1),torch.ones(5),atol=1e-6)
        restored=Model(cfg,dims,6).eval();restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        assert torch.equal(restored(x)['p'],out['p'])
        center=torch.randn(5,model.scale_pool.base_dimension)
        delta=torch.randn_like(center)
        same=torch.stack([center,center],1);different=torch.stack([center-delta,center+delta],1)
        assert torch.allclose(same.mean(1),different.mean(1),atol=1e-6)
        assert not torch.allclose(model.scale_pool.pair_features(same),model.scale_pool.pair_features(different))
        for drop in [0.,1.]:
            model.train();model.cfg['interaction_dropout']=drop;model.cfg['scale_dropout']=drop;model.zero_grad()
            train_out=model(x);loss=-train_out['p'][:,0].clamp(min=1e-7).log().mean();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        checks.append(kind)
    dump(ROOT/'verification_interaction_head.json',{'passed':True,'interaction_kinds':checks,'checks':['Scale permutation invariance','Same-mean scale pairs retain distinguishable interaction evidence','Finite backward with no dropout and complete single-view/interaction dropout','Normalized probabilities and strict checkpoint reconstruction']})
    print('Interaction head checks passed',flush=True)


if __name__=='__main__':main()

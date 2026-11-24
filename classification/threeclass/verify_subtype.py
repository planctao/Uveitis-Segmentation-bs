"""Check subtype marginalization and fine supervision without cross-patient positives."""
import json
import torch
from experiment import ROOT,Model,configuration,loss_fn,dump

def main():
    torch.set_num_threads(4);dims={'global':768,'regions':768,'mid':768,'dispersion':768,'disease_to_class':[0,0,1,1,2,2]}
    cfg=configuration('attention','dual',64,subtype_fusion=True,subtype_consistency=.1,fine_contrastive=.025,contrastive=.05)
    model=Model(cfg,dims,6)
    x={'global':torch.randn(12,768),'regions':torch.randn(12,9,768),'mid':torch.randn(12,768),'dispersion':torch.randn(12,768)}
    y=torch.tensor([0,0,1,1,2,2]*2);disease=torch.arange(12)%6
    out=model(x);assert torch.allclose(out['p'].sum(1),torch.ones(12),atol=1e-6)
    fine=out['subtype'].softmax(-1)
    expected=torch.stack([fine[:,:2].sum(1),fine[:,2:4].sum(1),fine[:,4:].sum(1)],1)
    assert torch.allclose(out['subtype_probability'],expected,atol=1e-6)
    assert out['mixture_weights'].shape==(12,4) and torch.allclose(out['mixture_weights'].sum(1),torch.ones(12))
    for patient_ids in [torch.arange(12),torch.zeros(12,dtype=torch.long)]:
        model.zero_grad();o=model(x);loss=loss_fn(o,y,disease,{'leaf':torch.ones(12),'patient_ids':patient_ids},cfg)
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();restored=Model(cfg,dims,6);restored.load_state_dict(model.state_dict(),strict=True);restored.eval()
    with torch.no_grad():assert torch.allclose(model(x)['p'],restored(x)['p'])
    for aux in [0.,.6]:
        compact_cfg=configuration('attention','flat',64,taxonomy_only=True,aux=aux,fine_contrastive=.025)
        compact=Model(compact_cfg,dims,6);o=compact(x)
        expected=o['subtype'].softmax(-1)@compact.subtype_to_class
        assert torch.allclose(o['p'],expected,atol=1e-6)
        assert not hasattr(compact,'binary') and not hasattr(compact,'mixture')
        l=loss_fn(o,y,disease,{'leaf':torch.ones(12),'patient_ids':torch.arange(12)},compact_cfg);l.backward()
        assert torch.isfinite(l) and all(torch.isfinite(p.grad).all() for p in compact.parameters() if p.grad is not None)
    result={'passed':True,'checks':['Subtype probabilities marginalize exactly to user-defined three classes','Four normalized mixture branches','Finite gradients with and without valid cross-patient subtype positives','Mapping buffer survives strict checkpoint restoration','Compact marginal head has no routing branches and finite gradients with zero and positive subtype supervision']}
    dump(ROOT/'verification_subtype.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

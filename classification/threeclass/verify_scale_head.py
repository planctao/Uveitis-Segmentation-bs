"""Check symmetry, scale gating and finite gradients in shared scale fusion."""
import json
import torch
from experiment import ROOT,configuration,Model,loss_fn,dump

def main():
    torch.set_num_threads(4);mapping=[0,0,1,1,1,2]
    cfg=configuration('scale_consensus','flat',64,taxonomy_only=True,aux=1.,scale_aux=.3,scale_consistency=.1)
    dims={'global':1536,'regions':1536,'mid':1536,'dispersion':1536,'disease_to_class':mapping}
    m=Model(cfg,dims,6);x={k:torch.randn(12,1536) for k in ['global','mid','dispersion']};x['regions']=torch.randn(12,9,1536)
    m.eval();swapped={k:torch.cat(v.chunk(2,-1)[::-1],dim=-1) for k,v in x.items()}
    with torch.no_grad():
        a=m(x);b=m(swapped)
        assert torch.allclose(a['p'],b['p'],atol=1e-6)
        assert torch.allclose(a['scale_probability'],b['scale_probability'].flip(1),atol=1e-6)
        assert torch.allclose(a['scale_weights'].sum(1),torch.ones(12))
        assert torch.allclose(a['p'].sum(1),torch.ones(12),atol=1e-6)
    disease=torch.arange(12)%6;y=torch.tensor(mapping)[disease]
    for drop in [0.,1.]:
        m.cfg['scale_dropout']=drop;m.train();m.zero_grad();o=m(x)
        loss=loss_fn(o,y,disease,{'leaf':torch.ones(12),'patient_ids':torch.arange(12)},m.cfg)
        loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if drop:assert torch.allclose(o['scale_weights'].amax(1),torch.ones(12))
    m.eval();restored=Model(cfg,dims,6);restored.load_state_dict(m.state_dict(),strict=True);restored.eval()
    with torch.no_grad():assert torch.allclose(m(x)['p'],restored(x)['p'])
    result={'passed':True,'checks':['Swapping scale order preserves fused predictions and swaps individual outputs','Both scale weights and all class distributions normalize','Finite gradients with no scale drop and complete single-scale dropout','Strict checkpoint round trip']}
    dump(ROOT/'verification_scale_head.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

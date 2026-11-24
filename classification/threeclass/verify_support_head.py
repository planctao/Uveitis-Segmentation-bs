"""Check neutral family priors, shrinkage and strict checkpoint reconstruction."""
import json
import torch
from taxonomy_head import SupportResidualHead
from experiment import ROOT,Model,configuration,loss_fn,dump

def main():
    torch.set_num_threads(4)
    mapping=[0,0,1,1,1,2];counts=[1,40,2,10,50,111]
    head=SupportResidualHead(16,mapping,counts,{'support_shrink':10.,'support_prior':'support'})
    x=torch.randn(12,16);matrix=torch.nn.functional.one_hot(torch.tensor(mapping),3).float()
    assert torch.allclose(head(x).softmax(-1)@matrix,head.coarse(x).softmax(-1),atol=1e-6)
    assert head.shrinkage[0]<head.shrinkage[1]
    for c in range(3):assert torch.allclose(head.prior[head.mapping==c].sum(),torch.tensor(1.))
    with torch.no_grad():head.residual.weight.normal_(0,.1);head.residual.bias.normal_(0,.1)
    residual=head.centered_residual(x)
    for c in range(3):assert torch.allclose((residual[:,head.mapping==c]*head.prior[head.mapping==c]).sum(1),torch.zeros(12),atol=1e-6)
    cfg=configuration('attention','flat',64,taxonomy_only=True,taxonomy_residual=True,aux=1.,support_shrink=10.)
    dims={'global':768,'regions':768,'mid':768,'dispersion':768,'disease_to_class':mapping,'disease_patient_counts':counts}
    model=Model(cfg,dims,6);features={k:torch.randn(12,768) for k in ['global','mid','dispersion']};features['regions']=torch.randn(12,9,768)
    disease=torch.arange(12)%6;y=torch.tensor(mapping)[disease]
    output=model(features);loss=loss_fn(output,y,disease,{'leaf':torch.ones(12),'patient_ids':torch.arange(12)},cfg)
    loss.backward();assert torch.isfinite(loss) and torch.allclose(output['p'].sum(1),torch.ones(12),atol=1e-6)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();replica=Model(cfg,dims,6);replica.load_state_dict(model.state_dict(),strict=True);replica.eval()
    with torch.no_grad():assert torch.allclose(model(features)['p'],replica(features)['p'])
    result={'passed':True,'checks':['Zero residual yields exactly the coarse classifier probabilities independent of subtype counts','Patient-support prior sums to one within every family','Rare subtype receives stronger shrinkage','Centered residual has zero prior-weighted family mean','Integrated marginal probabilities and gradients remain finite','Support buffers survive strict checkpoint reconstruction']}
    dump(ROOT/'verification_support_head.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

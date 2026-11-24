"""Check label binding, masking and a neutral path into a disease-query model."""
import copy
import torch
import torch.nn.functional as F
from experiment import ROOT,Model,configuration,loss_fn,dump
from subtype_query_head import SubtypeEvidenceQueries


def main():
    torch.set_num_threads(4)
    dims={'global':1536,'regions':1536,'mid':1536,'dispersion':1536,'disease_to_class':[0,0,1,1,1,2]}
    cfg=configuration('attention','flat',256,taxonomy_only=True,aux=.8,dropout=.6,wd=.1,lr=.002,label_smoothing=.03,fine_contrastive=.05)
    torch.manual_seed(23);base=Model(cfg,dims,6).eval()
    torch.manual_seed(23);new=Model({**cfg,'arch':'subtype_attention','query_aux':.2},dims,6).eval()
    for key,value in base.state_dict().items():assert torch.equal(value,new.state_dict()[key]),key
    x={k:torch.randn(5,9,v) if k=='regions' else torch.randn(5,v) for k,v in dims.items() if k!='disease_to_class'}
    x['region_mask']=torch.tensor([[1,0,0,0,1,0,0,0,1],[0]*9,[1]*9,[1]*9,[1]*9],dtype=torch.float)
    assert torch.equal(base(x)['p'],new(x)['p'])
    new.train();optimizer=torch.optim.AdamW(new.parameters(),lr=.002)
    y=torch.tensor([0,0,1,1,2]);disease=torch.tensor([0,1,2,3,5]);weights={'leaf':torch.ones(5),'patient_ids':torch.arange(5)}
    for step in range(2):
        optimizer.zero_grad();output=new(x);loss=loss_fn(output,y,disease,weights,new.cfg);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
        assert new.subtype_evidence.readout.grad.abs().sum()>0
        if step:assert new.subtype_evidence.queries.grad.abs().sum()>0
        optimizer.step()
    new.eval();out=new(x);assert torch.allclose(out['p'].sum(1),torch.ones(5),atol=1e-6)
    restored=Model(new.cfg,dims,6).eval();restored.load_state_dict(copy.deepcopy(new.state_dict()),strict=True)
    assert torch.equal(restored(x)['p'],out['p'])
    head=SubtypeEvidenceQueries(6,128,{'query_heads':2,'query_dropout':0.}).eval()
    with torch.no_grad():head.readout.normal_();head.bias.normal_()
    tokens=torch.randn(5,9,128);mask=x['region_mask'];evidence,attention=head(tokens,mask)
    assert (attention[0,:,:,mask[0]==0]==0).all() and torch.isfinite(attention[1]).all()
    assert torch.allclose(attention.sum(-1),torch.ones_like(attention.sum(-1)),atol=1e-6)
    order=torch.randperm(9);other,_=head(tokens[:,order],mask[:,order]);assert torch.allclose(evidence,other,atol=1e-5)
    disease_order=torch.randperm(6);permuted=copy.deepcopy(head)
    with torch.no_grad():
        for name in ['queries','readout','bias']:getattr(permuted,name).copy_(getattr(head,name)[disease_order])
    permuted_evidence,_=permuted(tokens,mask)
    assert torch.allclose(permuted_evidence,evidence[:,disease_order],atol=1e-6)
    mapping=F.one_hot(torch.tensor(dims['disease_to_class']),3).float()
    assert torch.allclose(evidence.softmax(-1)@mapping,permuted_evidence.softmax(-1)@mapping[disease_order],atol=1e-6)
    dump(ROOT/'verification_subtype_queries.json',{'passed':True,'checks':['Identical initialization of all existing base parameters','Zero residual exactly recovers the existing compact model','Readout and query parameters receive finite training gradients','Invalid regions excluded, empty masks handled, token order invariant','Disease query permutation preserves mapped class probabilities','Normalized outputs and strict checkpoint round trip']})
    print('Subtype query verification passed',flush=True)


if __name__=='__main__':main()

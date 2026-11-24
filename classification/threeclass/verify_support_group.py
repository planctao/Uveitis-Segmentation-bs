"""Check fit-support grouping without changing the original output architecture."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from support_group import SupportGroups


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    mapping=[0,0,0,1,1,2];counts=[2,3,10,2,20,40]
    groups=SupportGroups(mapping,counts,5)
    assert groups.disease_to_group.tolist()==[0,0,1,2,3,4]
    logits=torch.randn(9,6,requires_grad=True)
    grouped=groups(logits);fine=logits.softmax(1)
    expected=torch.stack([fine[:,groups.disease_to_group==g].sum(1) for g in range(groups.group_count)],1)
    assert torch.allclose(grouped.softmax(1),expected,atol=1e-7)
    targets=torch.tensor([0,1,2,3,4,5,-1,0,3]);gt=groups.targets(targets)
    assert gt[-3]==-1
    actual=F.cross_entropy(grouped,gt,ignore_index=-1,reduction='none')
    explicit=torch.where(targets>=0,-expected[torch.arange(9),gt.clamp(min=0)].log(),torch.zeros(9))
    assert torch.allclose(actual,explicit,atol=1e-6)
    singleton=SupportGroups(mapping,counts,1)
    assert torch.allclose(singleton(logits),logits,atol=1e-7)
    for group in range(groups.group_count):
        parent={mapping[i] for i in torch.nonzero(groups.disease_to_group==group).flatten().tolist()}
        assert len(parent)==1
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit')
    fitting=frame.split.eq('train');tf=frame.loc[fitting]
    support=tf.groupby('disease').patient_group.nunique()
    dims={**c['dims'],'disease_patient_counts':[int(support[name]) for name in c['diseases']]}
    selected=np.flatnonzero(frame.split.eq('val'))[:9]
    normalized,_=normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
    x={key:torch.from_numpy(value) for key,value in normalized.items()}
    for threshold in [5,10,20]:
        cfg={**c['cfg'],'support_group_threshold':threshold}
        torch.manual_seed(79);base=Model(c['cfg'],dims,len(c['diseases'])).eval()
        torch.manual_seed(79);model=Model(cfg,dims,len(c['diseases'])).eval()
        for key,value in base.state_dict().items():assert torch.equal(model.state_dict()[key],value)
        assert torch.equal(base(x)['p'],model(x)['p'])
        model.load_state_dict({**c['state_dict'],**{k:v for k,v in model.state_dict().items() if k.startswith('support_groups.')}},strict=True)
        base.load_state_dict(c['state_dict'],strict=True)
        assert torch.equal(model(x)['p'],base(x)['p'])
        for mode in [False,True]:
            model.cfg={**cfg,'support_group_contrastive':mode}
            y=torch.tensor(frame.iloc[selected].label.tolist())
            disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[selected].disease])
            model.train();out=model(x);loss=loss_fn(out,y,disease,{'leaf':torch.ones(9),'patient_ids':torch.arange(9)},model.cfg)
            model.zero_grad();loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval();restored=Model(model.cfg,dims,len(c['diseases'])).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        assert torch.equal(model(x)['p'],restored(x)['p'])
    dump(ROOT/'verification_support_group.json',{'passed':True,'manifest_sha256':digest,
        'checks':['Only low-support sibling diseases merge; every group belongs to exactly one coarse family',
            'Grouped logsumexp logits equal sums of original fine probabilities',
            'Singleton groups recover original auxiliary crossentropy',
            'Unknown disease targets ignored consistently',
            'Every original parameter, initial coarse prediction and restored M5prediction exactly unchanged',
            'Both original-disease and grouped-positive contrastive losses have finite full backward',
            'Strict serialized restore exactly reproduces grouped-model probabilities']})
    print('Support grouped taxonomy probability/initialization/loss checks passed',flush=True)


if __name__=='__main__':main()

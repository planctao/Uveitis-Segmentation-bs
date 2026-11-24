"""Regression checks for grouping, routing, numerical stability and saved outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,Model,configuration,loss_fn,leaf_probs,routed,sample_weights,normalize,dump
from evaluate import stages

def main():
    torch.set_num_threads(4);checks=[]
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    splits=sorted(frame.split.unique())
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            for key in ['patient_group','split_group','pixel_sha256']:
                assert not set(frame.loc[frame.split.eq(a),key]) & set(frame.loc[frame.split.eq(b),key])
    checks.append('patient/group/pixel disjointness')
    for path in (ROOT/'features').glob('*.npz'):
        if not path.with_suffix('.json').exists():continue
        arrays=np.load(path);assert str(arrays['manifest_sha256'])==digest
        assert np.array_equal(arrays['path'],frame.path)
    checks.append('feature row identities and manifest fingerprints')
    train=frame[frame.split.eq('train')];w=sample_weights(train,train.label)
    assert np.ptp([w[train.label.eq(i)].sum() for i in [0,1,2]])<.01
    checks.append('equal contributions per patient and class')
    y=np.array([0,1,2]);g=np.array([[.9,.1],[.1,.1],[.1,.1]])
    m=stages(y,g,'cascade_i',[.5,.5]);assert m['stage1_accuracy']==1 and m['stage2_oracle_accuracy']==.5 and m['end_to_end_correct_n']==2
    assert not np.isclose(m['stage1_accuracy']*m['stage2_oracle_accuracy'],2/3)
    g=np.array([[.1,.9],[.1,.9],[.1,.1]])
    m=stages(y,g,'cascade_i',[.5,.5]);assert m['stage2_oracle_accuracy']==1 and m['end_to_end_correct_n']==2
    for mode in ['cascade_i','cascade_n']:assert np.allclose(leaf_probs(g,mode).sum(1),1)
    checks.append('routing errors propagate; end-to-end accuracy is not the stage product')
    for arch in ['global','context','attention','class_attention','residual','cross_attention','evidence_mil']:
        cfg=configuration(arch,'dual',64,contrastive=.025)
        dims={'global':1536,'regions':1536,'mid':1152,'dispersion':1536}
        model=Model(cfg,dims,14);x={k:torch.randn(9,9,d) if k=='regions' else torch.randn(9,d) for k,d in dims.items()}
        out=model(x);y=torch.tensor([0,1,2]*3)
        assert torch.allclose(out['p'].sum(1),torch.ones(9),atol=1e-5)
        loss=loss_fn(out,y,torch.arange(9),{'leaf':torch.ones(9),'patient_ids':torch.arange(9)},cfg)
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    checks.append('seven model architectures: normalized probabilities and finite gradients')
    dims={'global':768,'regions':768,'mid':768,'dispersion':768}
    x={k:torch.randn(3,49,d) if k=='regions' else torch.randn(3,d) for k,d in dims.items()}
    x['region_mask']=torch.zeros(3,49);x['region_mask'][:,:5]=1
    model=Model(configuration('evidence_mil','dual',64,evidence_aux=.1),dims,14).eval()
    with torch.no_grad():
        before=model(x)['p'];x['regions'][:,5:]=1000
        after=model(x)['p'];assert torch.allclose(before,after,atol=1e-6)
        x['region_mask'].zero_();assert torch.isfinite(model(x)['p']).all()
        model.evidence.score_scale.fill_(1000);assert torch.isfinite(model(x)['p']).all()
    checks.append('local evidence ignores masked regions and handles empty masks and extreme temperature')
    features={k:v.numpy() for k,v in x.items()};features['region_mask'][:,:5]=1
    train_mask=np.array([True,True,False]);normalized,stats=normalize(features,train_mask)
    expected=features['regions'][train_mask][features['region_mask'][train_mask]>.5].mean(0)
    assert np.allclose(stats['regions']['mean'],expected)
    assert np.array_equal(normalized['region_mask'],features['region_mask'])
    checks.append('foreground feature statistics fit only valid training regions; binary masks preserved')
    for gamma in [.5,1.,2.]:
        logits=torch.tensor([[1000.,-1000.,-1000.]],requires_grad=True)
        loss=loss_fn({'p':logits.softmax(1)},torch.tensor([0]),torch.tensor([0]),{'leaf':torch.ones(1)},configuration('global','flat',64,focal_gamma=gamma))
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    checks.append('fractional focal gradients at saturated softmax probabilities')
    n=0
    for feature in ['convnext_320','vitb_320']:
        docs=json.loads((ROOT/'results'/feature/'baseline/comparison.json').read_text())
        for mode,document in docs.items():
            for split in ['val','test_independent','test_ningbo']:
                path=ROOT/'results'/feature/'baseline'/mode/f'{split}_images.csv';df=pd.read_csv(path)
                p=df[['p_infectious','p_noninfectious','p_normal']].to_numpy()
                assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6)
                assert np.array_equal(df.path,frame.loc[frame.split.eq(split),'path'])
                assert np.isclose((df.label==df.prediction).mean(),document['results'][split]['image']['accuracy'])
                if mode.startswith('cascade'):
                    g=df[['stage1_probability','stage2_probability']].to_numpy(dtype='float32')
                    assert np.array_equal(routed(g,mode,document['selection']['thresholds']),df.prediction)
                n+=1
    checks.append(f'{n} saved prediction tables: probabilities, row identities, accuracy and hard routing reproduce')
    result={'passed':True,'manifest_sha256':digest,'checks':checks}
    dump(ROOT/'verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

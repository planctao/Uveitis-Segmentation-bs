"""Guard controller composition of taxonomy residuals and shared scale models."""
import json
import numpy as np
import torch
from continuous import canonical_cfg,signature,proposals,fresh_proposals,starting_pool
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump

def main():
    torch.set_num_threads(4)
    baseline={'arch':'attention','mode':'flat','hidden':64,'dropout':.3,'lr':.001,'wd':.01,'epochs':120,'patience':18,'taxonomy_only':True,'aux':1.}
    assert signature('multiscale_vit',baseline)=='4726000e84fe'
    frame,features,digest=load_data('multiscale_vit')
    fit=frame.split.eq('train').to_numpy();data,stats=normalize(features,fit)
    diseases=sorted(frame.loc[fit,'disease'].unique());mapping=frame.loc[fit].groupby('disease').label.first();support=frame.loc[fit].groupby('disease').patient_group.nunique()
    dims={k:v.shape[-1] for k,v in features.items()}
    dims.update(disease_to_class=[int(mapping[d]) for d in diseases],disease_patient_counts=[int(support[d]) for d in diseases])
    x={k:torch.tensor(v[:6]) for k,v in data.items()};y=torch.tensor(frame.label.to_numpy()[:6]);sub=torch.tensor([diseases.index(d) for d in frame.disease[:6]])
    checked=[]
    for family in ['support_taxonomy','scale_consensus','scale_interaction','subtype_queries']:
        if not (ROOT/'runs/multiscale_vit'/family/'leaderboard.json').exists():continue
        records=json.loads((ROOT/'runs/multiscale_vit'/family/'leaderboard.json').read_text())
        for record in records:
            cfg=record['config'];clean=canonical_cfg(cfg)
            assert clean.get('taxonomy_only')
            if family=='support_taxonomy':assert clean.get('taxonomy_residual')
            elif family=='subtype_queries':assert clean['arch']=='subtype_attention' and not clean.get('taxonomy_residual')
            else:assert clean['arch'] in ['scale_consensus','scale_interaction'] and not clean.get('taxonomy_residual')
            ckpt=torch.load(record['checkpoints'][0],map_location='cpu',weights_only=False)
            model=Model(clean,ckpt['dims'],len(ckpt['diseases']));model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
            out=model(x);assert torch.isfinite(out['p']).all() and torch.allclose(out['p'].sum(-1),torch.ones(6),atol=1e-5)
            weights={'leaf':torch.ones(6),'patient_ids':torch.tensor(frame.patient_group.astype('category').cat.codes.to_numpy()[:6],dtype=torch.long)}
            loss=loss_fn(out,y,sub,weights,clean);loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            for arch in ['global','attention','subtype_attention','residual','class_attention','memory_attention','cross_attention','scale_consensus','scale_interaction']:
                transitioned=canonical_cfg({**cfg,'arch':arch,'prototype':True})
                if arch not in ['scale_consensus','scale_interaction']:assert not any(k.startswith('scale_') for k in transitioned)
                if not transitioned.get('taxonomy_only'):assert not transitioned.get('taxonomy_residual')
                if arch in ['scale_consensus','scale_interaction','subtype_attention']:assert transitioned['mode']=='flat' and transitioned.get('taxonomy_only') and not transitioned.get('taxonomy_residual') and not transitioned.get('prototype')
                if arch!='scale_interaction':assert not any(k.startswith('interaction_') for k in transitioned)
                if arch!='subtype_attention':assert not any(k.startswith('query_') for k in transitioned)
            checked.append({'family':family,'candidate_id':record['candidate_id'],'parameters':sum(p.numel() for p in model.parameters())})
    pool=starting_pool();seen=set()
    for iteration in range(200):
        batch=fresh_proposals(iteration,pool,seen)
        assert len(batch)==6
        for feature,cfg,_ in batch:
            key=signature(feature,cfg);assert key not in seen;seen.add(key)
            if cfg['arch'] in ['scale_consensus','scale_interaction']:assert feature=='multiscale_vit' and cfg['mode']=='flat' and cfg.get('taxonomy_only') and not cfg.get('taxonomy_residual')
            if cfg['arch']=='subtype_attention':assert cfg['mode']=='flat' and cfg.get('taxonomy_only') and not cfg.get('taxonomy_residual')
            if cfg.get('taxonomy_residual'):assert cfg.get('taxonomy_only') and cfg['arch'] not in ['scale_consensus','scale_interaction','subtype_attention']
    dump(ROOT/'verification_search_extensions.json',{'passed':True,'checks':['Existing compact champion identity preserved','All available new candidates reconstruct strictly after controller canonicalization','Probabilities and backward passes finite','Architecture changes remove incompatible extras','1200 fresh proposals are unique and respect model constraints'],'candidates':checked})
    print('Search extension verification passed',len(checked),'models; fresh proposals',len(seen),flush=True)

if __name__=='__main__':main()

"""Persistent validation-only exploration. STOP file requests a clean pause.

Each round: propose ablations -> three-seed experiments -> validation ranking ->
use the leading configurations to propose the next round. External outcomes are
never read. Keep all summaries and prune only unpromising search checkpoints.
"""
import argparse
import hashlib
import json
import os
import random
import math
import traceback
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from experiment import ROOT,load_data,train_one,summarize_group,configuration,dump

OUT=ROOT/'runs/continuous_search'

def canonical_cfg(config):
    cfg=config.copy();arch=cfg['arch'];mode=cfg['mode']
    if arch in ['scale_consensus','scale_interaction','subtype_attention']:
        cfg.update(mode='flat',taxonomy_only=True);mode='flat'
        cfg.pop('taxonomy_residual',None)
    if arch.startswith('memory_') and mode not in ['flat','dual']:cfg['mode']=mode='dual'
    for key in ['contrastive','fine_contrastive','label_smoothing','focal_gamma','prototype','subtype_fusion','subtype_consistency','taxonomy_only']:
        if key in cfg and not cfg[key]:cfg.pop(key)
    if cfg['hidden']==0:cfg['dropout']=0.
    if cfg.get('taxonomy_only',False):
        if mode!='flat' or arch not in ['global','attention','class_attention','subtype_attention','context','residual','scale_consensus','scale_interaction']:cfg.pop('taxonomy_only')
        else:cfg.pop('prototype',None)
    if not cfg.get('taxonomy_only',False):cfg.pop('taxonomy_residual',None)
    if not cfg.get('taxonomy_residual',False):
        for key in ['support_prior','support_shrink','residual_penalty','taxonomy_residual']:cfg.pop(key,None)
    if arch not in ['scale_consensus','scale_interaction']:
        for key in ['learn_scale_gate','scale_aux','scale_consistency','scale_dropout','scale_probability_blend']:cfg.pop(key,None)
    if arch!='scale_interaction':
        for key in ['interaction_kind','interaction_rank','interaction_dropout']:cfg.pop(key,None)
    if arch!='subtype_attention':
        for key in ['query_heads','query_temperature','query_dropout','query_aux']:cfg.pop(key,None)
    if mode!='dual':
        for key in ['consistency','subtype_fusion','subtype_consistency']:cfg.pop(key,None)
        if not cfg.get('taxonomy_only',False):cfg.pop('aux',None)
    if mode.startswith('cascade'):
        for key in ['contrastive','fine_contrastive','prototype','label_smoothing','focal_gamma']:cfg.pop(key,None)
    if not cfg.get('subtype_fusion',False):cfg.pop('subtype_consistency',None)
    attention=arch in ['attention','class_attention','subtype_attention','memory_attention','scale_consensus','scale_interaction']
    if not attention or cfg.get('region_heads')==(3 if arch=='class_attention' else 2):cfg.pop('region_heads',None)
    if arch!='evidence_mil':
        for key in ['evidence_aux','evidence_residual','topk','global_conditioning','region_dropout']:cfg.pop(key,None)
    if not arch.startswith('memory_'):
        for key in list(cfg):
            if key.startswith('memory_'):cfg.pop(key)
    if not attention and arch!='evidence_mil':cfg.pop('use_region_mask',None)
    if cfg.get('use_region_mask') is True:cfg.pop('use_region_mask')
    return cfg

def signature(feature,cfg):return hashlib.sha256(json.dumps([feature,canonical_cfg(cfg)],sort_keys=True).encode()).hexdigest()[:12]

def unique_pool(pool):
    unique={}
    for item in pool:
        key=signature(item['feature'],item['cfg'])
        if key not in unique or item['macro_f1']>unique[key]['macro_f1']:unique[key]=item
    return list(unique.values())

def known_cv():
    registry={}
    for feature_dir in (ROOT/'runs').iterdir():
        if not (ROOT/'features'/f'{feature_dir.name}.npz').exists():continue
        for namespace in ['crossvalidation','candidate_crossvalidation']:
            for path in (feature_dir/namespace).glob('*/summary.json'):
                report=json.loads(path.read_text());cfg=report['config'];key=signature(feature_dir.name,cfg)
                registry[key]={'feature':feature_dir.name,'config':cfg,'mean_macro_f1':report['mean_macro_f1'],'sd_macro_f1':report['sd_macro_f1'],'source':str(path)}
    return registry

def audit_leader(records,registry,iteration,experiment_round=None):
    unverified=[r for r in records if signature(r['feature'],r['config']) not in registry]
    if not unverified:return
    candidate=max(unverified,key=lambda x:x['val']['macro_f1']);key=signature(candidate['feature'],candidate['config'])
    request=OUT/'cv_requests'/f'{candidate["candidate_id"]}.json';dump(request,candidate)
    experiment_round=iteration if experiment_round is None else experiment_round
    state={'status':'cross_validation','pid':os.getpid(),'round':experiment_round,'completed_round':experiment_round,'completed_experiment_rounds':experiment_round,'proposal_iteration':iteration,'candidate_id':candidate['candidate_id'],'feature':candidate['feature'],'time':now(),'external_test_policy':'Training/validation evidence only; external files never read.'}
    dump(OUT/'state.json',state)
    with (OUT/'crossvalidation.log').open('a') as log:
        for args in [['crossvalidate.py','--features',candidate['feature'],'--candidate-json',str(request)],['audit_candidate_cv.py','--candidate-json',str(request)]]:
            subprocess.run([sys.executable,'-u',str(ROOT/args[0]),*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    registry.update(known_cv());dump(OUT/'cv_registry.json',registry)
    ranked=sorted(registry.values(),key=lambda x:x['mean_macro_f1'],reverse=True)
    dump(OUT/'validated_leaderboard.json',ranked)

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def starting_pool():
    pool=[]
    available=[p.stem for p in (ROOT/'features').glob('*.npz') if (ROOT/'runs'/p.stem).is_dir()]
    for feature in sorted(available):
        for phase in ['baseline','explore','memory_attention','subtype_hierarchy','taxonomy']:
            path=ROOT/'runs'/feature/phase/'selection.json'
            if path.exists():
                best=json.loads(path.read_text())['best']
                pool.append({'feature':feature,'cfg':best['config'],'macro_f1':best['val']['macro_f1']})
        for phase in ['support_taxonomy','scale_consensus','scale_interaction','regularization_ablation','subtype_queries']:
            path=ROOT/'runs'/feature/phase/'leaderboard.json'
            if path.exists():
                for record in json.loads(path.read_text()):
                    pool.append({'feature':feature,'cfg':record['config'],'macro_f1':record['val']['macro_f1']})
    return pool

def proposals(round_number,pool):
    rng=random.Random(20261124+round_number);items=[]
    if round_number==0:
        base=configuration('attention','dual',128,aux=.4,consistency=.3)
        items=[('vitb_320',{**base,'arch':'class_attention'},'Use three regional attention queries'),
               ('vitb_320',{**base,'arch':'residual'},'Test a residual feature adapter'),
               ('vitb_320',{**base,'contrastive':.025},'Align same-class representations across different patients'),
               ('fusion_320',{**base,'arch':'cross_attention'},'Let each DINO backbone query regions from the other'),
               ('fusion_320',{**base,'arch':'class_attention'},'Test class-conditioned regions with both backbones'),
               ('vitb_320',{**base,'label_smoothing':.03},'Test calibration through mild label smoothing')]
    else:
        pool=unique_pool(pool)
        leaders=sorted(pool,key=lambda x:x['macro_f1'],reverse=True)[:8]
        families={}
        for item in pool:families.setdefault((item['feature'],item['cfg']['arch'],item['cfg']['mode'],bool(item['cfg'].get('taxonomy_only')),bool(item['cfg'].get('subtype_fusion')),bool(item['cfg'].get('taxonomy_residual'))),[]).append(item)
        def ucb(arm):return max(x['macro_f1'] for x in arm)+.02*math.sqrt(math.log(len(pool)+1)/(len(arm)+1))
        diverse=sorted(families.values(),key=ucb,reverse=True)
        for i in range(6):
            leader=rng.choice(leaders[:3]) if i<4 else max(rng.choice(diverse[:3]),key=lambda x:x['macro_f1'])
            feature=leader['feature'];cfg=leader['cfg'].copy()
            changes={'hidden':[64,128,256],'dropout':[.15,.3,.45,.6],'lr':[.0002,.0005,.001,.002],'wd':[.003,.01,.05,.1],'contrastive':[0.,.025,.05,.1],'label_smoothing':[0.,.03,.06],'focal_gamma':[0.,.5,1.]}
            if cfg['mode']=='dual':changes.update(aux=[0.,.2,.4,.6],consistency=[0.,.1,.3,.6])
            if cfg['mode']=='dual':changes.update(subtype_fusion=[False,True],subtype_consistency=[0.,.05,.1,.2])
            if cfg.get('taxonomy_only',False):changes['aux']=[0.,.2,.6,1.]
            if cfg.get('taxonomy_only',False) and cfg['arch'] not in ['scale_consensus','scale_interaction','subtype_attention']:changes['taxonomy_residual']=[False,True]
            if cfg.get('taxonomy_residual',False):changes.update(support_prior=['uniform','support'],support_shrink=[0.,5.,10.,30.],residual_penalty=[0.,.001,.01])
            if cfg['arch'] in ['scale_consensus','scale_interaction']:changes.update(learn_scale_gate=[False,True],scale_aux=[0.,.1,.3,.6],scale_consistency=[0.,.05,.1,.2],scale_dropout=[0.,.1,.25],scale_probability_blend=[0.,.25,.5])
            if cfg['arch']=='scale_interaction':changes.update(interaction_kind=['difference','product','both'],interaction_rank=[32,64,128],interaction_dropout=[0.,.1,.25])
            if cfg['arch']=='subtype_attention':changes.update(query_heads=[1,2,4],query_temperature=[.1,.2,.5,1.],query_dropout=[0.,.1,.3],query_aux=[0.,.1,.2,.4])
            if cfg['mode'] in ['flat','dual']:changes['fine_contrastive']=[0.,.01,.025,.05]
            if cfg['arch'] in ['attention','class_attention','subtype_attention','memory_attention','scale_consensus','scale_interaction']:changes['region_heads']=[1,2,3,4]
            if cfg['arch'].startswith('memory_'):changes.update(memory_topk=[3,5,15],memory_cap=[.25,.5,.75],memory_gate_evidence=[False,True])
            if cfg['arch']=='evidence_mil':changes.update(topk=[3,5,8],evidence_aux=[0.,.05,.1],global_conditioning=[False,True])
            if i==5:
                architectures=['attention','class_attention','subtype_attention','residual','global','memory_attention']
                if feature in ['fusion_320','multiscale_vit']:architectures.append('cross_attention')
                if feature=='multiscale_vit':architectures.extend(['scale_consensus','scale_interaction'])
                if feature.startswith('vitb_patch'):architectures.append('evidence_mil')
                cfg['arch']=rng.choice(architectures)
                if cfg['arch'] in ['global','attention','class_attention','residual'] and rng.random()<.25:cfg.update(mode='flat',taxonomy_only=True)
                key='arch'
            else:
                key=rng.choice(list(changes));cfg[key]=rng.choice(changes[key])
            items.append((feature,canonical_cfg(cfg),f'Change {key} around a leading validation configuration' if i<4 else f'Explore a less-tested family and change {key}'))
    return items

def fresh_proposals(iteration,pool,seen,target=6):
    """Create real new trials when a tested local neighbourhood is exhausted."""
    fresh=[];reserved=set(seen)
    for feature,cfg,hypothesis in proposals(iteration,pool):
        cfg=canonical_cfg(cfg);key=signature(feature,cfg)
        if key not in reserved:fresh.append((feature,cfg,hypothesis));reserved.add(key)
    rng=random.Random(20261124+iteration*100003)
    leaders=sorted(unique_pool(pool),key=lambda x:x['macro_f1'],reverse=True)
    for attempt in range(200):
        if len(fresh)>=target:break
        leader=rng.choice(leaders[:min(12,len(leaders))]);feature=leader['feature'];cfg=leader['cfg'].copy()
        cfg['lr']=float(f'{math.exp(rng.uniform(math.log(1e-4),math.log(.003))):.8g}')
        cfg['wd']=float(f'{math.exp(rng.uniform(math.log(1e-5),math.log(.1))):.8g}')
        cfg['dropout']=round(rng.uniform(.1,.6),2)
        if rng.random()<.5:cfg['hidden']=rng.choice([64,96,128,192,256])
        if cfg['mode']=='dual' or cfg.get('taxonomy_only',False):cfg['aux']=rng.choice([0.,.2,.4,.6,.8,1.])
        if cfg['mode'] in ['flat','dual'] and rng.random()<.5:cfg['contrastive']=rng.choice([0.,.025,.05])
        cfg=canonical_cfg(cfg);key=signature(feature,cfg)
        if key in reserved:continue
        fresh.append((feature,cfg,'Joint optimizer, regularization and capacity exploration after tested local proposals were exhausted'));reserved.add(key)
    if len(fresh)<target:raise RuntimeError('No new experiment proposals; refuse to spin without training')
    return fresh[:target]

def prune(records):
    # At most 8 new candidates plus each feature's leader retain checkpoint files.
    keep={r['group_path'] for r in sorted(records,key=lambda x:x['val']['macro_f1'],reverse=True)[:8]}
    for feature in {r['feature'] for r in records}:
        own=[r for r in records if r['feature']==feature]
        if own:keep.add(max(own,key=lambda x:x['val']['macro_f1'])['group_path'])
    for record in records:
        if record['group_path'] in keep:continue
        directory=__import__('pathlib').Path(record['group_path'])
        if not directory.resolve().is_relative_to(OUT.resolve()):raise ValueError('Unsafe retention path')
        for path in directory.glob('seed_*/model.pt'):path.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',type=int,default=0,help='0 means continuous');parser.add_argument('--cv-every',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    pid_file=OUT/'runner.pid'
    if pid_file.exists():
        old_pid=int(pid_file.read_text());proc=__import__('pathlib').Path('/proc',str(old_pid),'cmdline')
        if old_pid!=os.getpid() and proc.exists() and b'continuous.py' in proc.read_bytes():
            raise RuntimeError(f'Another continuous exploration process is already running: {old_pid}')
    pid_file.write_text(str(os.getpid()))
    records=json.loads((OUT/'leaderboard.json').read_text()) if (OUT/'leaderboard.json').exists() else []
    state=json.loads((OUT/'state.json').read_text()) if (OUT/'state.json').exists() else {}
    start_round=state.get('proposal_iteration',state.get('completed_round',-1))+1
    experiment_round=state.get('completed_experiment_rounds',len({r.get('round') for r in records if r.get('round',-1)>=0}))
    legacy_proposals=state.get('legacy_proposal_iterations',start_round)
    seen={signature(r['feature'],r['config']) for r in records};cache={};pool=starting_pool()
    pool += [{'feature':r['feature'],'cfg':r['config'],'macro_f1':r['val']['macro_f1']} for r in records]
    registry=known_cv();dump(OUT/'cv_registry.json',registry)
    dump(OUT/'validated_leaderboard.json',sorted(registry.values(),key=lambda r:r['mean_macro_f1'],reverse=True))
    iteration=start_round
    while args.rounds==0 or iteration<start_round+args.rounds:
        # Newly completed resolution/feature experiments can join subsequent rounds.
        for starter in starting_pool():
            seen.add(signature(starter['feature'],starter['cfg']))
            if not any(p['feature']==starter['feature'] and p['cfg']==starter['cfg'] for p in pool):pool.append(starter)
        registry.update(known_cv())
        best_cv=max([r['mean_macro_f1'] for r in registry.values()],default=0.)
        for item in pool:
            cv=registry.get(signature(item['feature'],item['cfg']))
            item['cv_rejected']=bool(cv and cv['mean_macro_f1']<best_cv-.005)
        if (ROOT/'STOP').exists():dump(OUT/'state.json',{'status':'stopped_by_STOP_file','pid':os.getpid(),'completed_round':experiment_round,'completed_experiment_rounds':experiment_round,'proposal_iteration':iteration-1,'time':now()});return
        eligible=[item for item in pool if not item.get('cv_rejected')]
        for feature,cfg,hypothesis in fresh_proposals(iteration,eligible or pool,seen):
            cfg=canonical_cfg(cfg);candidate_id=signature(feature,cfg)
            if candidate_id in seen:continue
            if (ROOT/'STOP').exists():
                dump(OUT/'state.json',{'status':'stopped_by_STOP_file','pid':os.getpid(),'completed_round':experiment_round,'completed_experiment_rounds':experiment_round,'proposal_iteration':iteration-1,'time':now()});return
            directory=OUT/f'{feature}_{candidate_id}'
            dump(OUT/'state.json',{'status':'running','pid':os.getpid(),'round':experiment_round+1,'completed_round':experiment_round,'completed_experiment_rounds':experiment_round,'proposal_iteration':iteration,'legacy_proposal_iterations':legacy_proposals,'feature':feature,'candidate_id':candidate_id,'hypothesis':hypothesis,'policy':'Six unseen configurations per real experiment round; joint-parameter fallback; periodic development CV; external results never read.','time':now(),'external_test_policy':'Never read or evaluate external results inside this loop.'})
            try:
                if feature not in cache:cache[feature]=load_data(feature)
                frame,features,digest=cache[feature]
                for seed in range(3):train_one(frame,features,digest,{**cfg,'seed':seed},directory/f'seed_{seed}')
                record=summarize_group(directory,frame)
                record.update(feature=feature,candidate_id=candidate_id,round=experiment_round+1,proposal_iteration=iteration,round_policy_version=2,hypothesis=hypothesis,group_path=str(directory),time=now())
                records.append(record);seen.add(candidate_id)
                pool.append({'feature':feature,'cfg':cfg,'macro_f1':record['val']['macro_f1']})
                records.sort(key=lambda x:x['val']['macro_f1'],reverse=True)
                dump(OUT/'leaderboard.json',records);prune(records)
                print(json.dumps({'event':'exploration_complete','result':record}),flush=True)
            except Exception:
                directory.mkdir(parents=True,exist_ok=True);(directory/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True);torch.cuda.empty_cache()
        experiment_round+=1
        if args.cv_every>0 and experiment_round%args.cv_every==0:
            try:audit_leader(records,registry,iteration,experiment_round)
            except Exception:
                with (OUT/'cv_errors.log').open('a') as log:log.write(traceback.format_exc())
        dump(OUT/'state.json',{'status':'round_complete','pid':os.getpid(),'round':experiment_round,'completed_round':experiment_round,'completed_experiment_rounds':experiment_round,'proposal_iteration':iteration,'legacy_proposal_iterations':legacy_proposals,'candidates_completed':len(records),'best_validation':records[0] if records else None,'best_development_cv':max([r['mean_macro_f1'] for r in registry.values()],default=None),'time':now()})
        iteration+=1

if __name__=='__main__':main()

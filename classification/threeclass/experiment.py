"""Validation-only exploration of frozen-DINO MLPs and hierarchical region heads.

External labels never enter fitting, early stopping, threshold choice or ranking.
Run evaluate.py explicitly after a selection milestone is frozen.
"""
import os
os.environ.setdefault('NUMEXPR_NUM_THREADS','6')
import argparse
import hashlib
import json
import random
import time
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,log_loss,roc_auc_score

ROOT=Path(__file__).resolve().parent
KEYS=['global','regions','mid','dispersion']
CLASSES=['infectious','noninfectious','normal']
MEMORY_ARCHES=('memory_hybrid','memory_attention')

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        stream.write(serialized);temporary=Path(stream.name)
    temporary.replace(path)

def metrics(y,p,pred=None):
    y=np.asarray(y,dtype=int);p=np.asarray(p);pred=p.argmax(1) if pred is None else np.asarray(pred)
    cm=confusion_matrix(y,pred,labels=[0,1,2]);n=cm.sum(1)
    log_prob=np.clip(p.astype('float64'),1e-7,1)
    log_prob=log_prob/log_prob.sum(1,keepdims=True)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(recall_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),log_loss=float(log_loss(y,log_prob,labels=[0,1,2])),recall={CLASSES[i]:float(cm[i,i]/n[i]) if n[i] else None for i in range(3)},confusion_matrix=cm.tolist())

def leaf_probs(g,mode):
    a,b=g[:,0],g[:,1]
    if mode=='cascade_i': return np.stack([a,(1-a)*b,(1-a)*(1-b)],1)
    return np.stack([a*b,a*(1-b),1-a],1)

def routed(g,mode,thresholds=(.5,.5)):
    a,b=g[:,0]>=thresholds[0],g[:,1]>=thresholds[1]
    return np.where(a,0,np.where(b,1,2)) if mode=='cascade_i' else np.where(a,np.where(b,0,1),2)

def tune_thresholds(y,g,mode):
    best=None
    for a in np.linspace(.2,.8,13):
        for b in np.linspace(.2,.8,13):
            pred=routed(g,mode,(a,b));score=float(f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0))
            key=(score,float(accuracy_score(y,pred)),-abs(a-.5)-abs(b-.5))
            if best is None or key>best[0]: best=(key,[float(a),float(b)])
    return best[1]

def sample_weights(df,labels,mask=None):
    labels=np.asarray(labels);mask=np.ones(len(df),dtype=bool) if mask is None else np.asarray(mask)
    x=pd.DataFrame({'patient':df.patient_group.to_numpy()[mask],'label':labels[mask]})
    n=x.groupby(['patient','label']).size();nc=x.drop_duplicates().groupby('label').size()
    out=np.zeros(len(df),dtype='float32')
    out[mask]=[1/(nc[r.label]*n[r.patient,r.label]) for r in x.itertuples()]
    out[mask]*=mask.sum()/out.sum()
    return out

def load_data(feature='convnext_320'):
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    if feature in {'multiscale_mirror_mean','multiscale_mirror_bag','multiscale_mirror_pairs'}:
        from mirror_data import load_derived
        return frame,load_derived(ROOT,frame,digest,feature),digest
    arrays=np.load(ROOT/'features'/f'{feature}.npz')
    assert str(arrays['manifest_sha256'])==digest,'Feature/manifest fingerprint mismatch'
    assert np.array_equal(arrays['path'],frame.path),'Feature rows must match manifest exactly'
    keys=KEYS+(['region_mask'] if 'region_mask' in arrays.files else [])
    return frame,{k:arrays[k].astype('float32') for k in keys},digest

def normalize(features,train_mask,stats=None):
    if stats is None:
        stats={}
        for k,x in features.items():
            if k=='region_mask':
                stats[k]={'mean':np.zeros(x.shape[-1],dtype='float32'),'std':np.ones(x.shape[-1],dtype='float32')}
                continue
            z=x[train_mask]
            if k=='regions' and 'region_mask' in features:
                valid=features['region_mask'][train_mask]>.5
                if valid.any():z=z[valid]
            axes=tuple(range(z.ndim-1))
            stats[k]={'mean':z.mean(axes).astype('float32'),'std':np.maximum(z.std(axes),.05).astype('float32')}
    normalized={k:np.clip((v-stats[k]['mean'])/stats[k]['std'],-10,10).astype('float32') for k,v in features.items()}
    return normalized,stats

class Head(nn.Module):
    def __init__(self,d,h,c,drop):
        super().__init__()
        self.net=nn.Linear(d,c) if h==0 else nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(drop),nn.Linear(h,c))
    def forward(self,x): return self.net(x)

class Model(nn.Module):
    def __init__(self,cfg,dims,n_diseases):
        super().__init__();self.cfg=cfg;self.mode=cfg['mode'];arch=cfg['arch'];h=cfg['hidden'];drop=cfg['dropout']
        d=dims['global']
        self.arch=arch
        if arch not in ['global','context','attention','class_attention','subtype_attention','cross_attention','residual','evidence_mil','scale_consensus','scale_interaction','mirror_consensus',*MEMORY_ARCHES]:
            raise ValueError(f'Unknown model architecture: {arch}')
        if arch=='context': d+=dims['mid']+dims['dispersion']
        if arch in ['attention','class_attention','subtype_attention','memory_attention']:
            region_heads=cfg.get('region_heads',3 if arch=='class_attention' else 2)
            self.region_proj=nn.Linear(dims['regions'],128)
            self.region_gate=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,region_heads))
            self.global_proj=nn.Linear(dims['global'],256)
            self.mid_proj=nn.Linear(dims['mid'],64)
            self.std_proj=nn.Linear(dims['dispersion'],64)
            d=384+128*region_heads
        if arch=='cross_attention':
            assert dims['global']%2==0 and dims['regions']==dims['global']
            self.view_proj=nn.ModuleList([nn.Linear(dims['global']//2,128) for _ in range(2)])
            self.cross_attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
            self.mid_proj=nn.Linear(dims['mid'],64)
            self.std_proj=nn.Linear(dims['dispersion'],64)
            d=640
        if arch=='residual':
            self.adapter=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,d))
            self.adapter_scale=nn.Parameter(torch.tensor(.1))
        if arch=='evidence_mil':
            from patch_head import EvidencePool
            self.evidence=EvidencePool(dims,cfg)
            d=768
        if arch=='mirror_consensus':
            if self.mode!='flat' or not cfg.get('taxonomy_only') or cfg.get('taxonomy_residual'):
                raise ValueError('Mirror pooling requires ordinary flat compact subtype marginalization')
            from mirror_head import MirrorConsensusPool
            self.mirror_pool=MirrorConsensusPool(dims,cfg)
            d=self.mirror_pool.dimension
        if arch in ['scale_consensus','scale_interaction']:
            if self.mode!='flat' or not cfg.get('taxonomy_only') or cfg.get('taxonomy_residual'):
                raise ValueError('Shared scale pooling currently uses a compact ordinary subtype head')
            from scale_head import ScaleConsensusPool
            if arch=='scale_interaction':
                from interaction_head import ScaleInteractionPool
                self.scale_pool=ScaleInteractionPool(dims,cfg)
            else:self.scale_pool=ScaleConsensusPool(dims,cfg)
            d=self.scale_pool.dimension
        if arch in MEMORY_ARCHES:
            if self.mode not in ['flat','dual']:raise ValueError('Patient memory is supported for flat and joint hierarchical heads')
            from memory_head import PatientMemory
            self.memory=PatientMemory(dims['global'],dims['memory_count'],cfg)
        if self.mode in ['cascade_i','cascade_n']:
            # Independent MLPs implement exactly the two requested conditional tasks.
            self.stage1=Head(d,h,2,drop);self.stage2=Head(d,h,2,drop)
        else:
            self.encoder=nn.Identity() if h==0 else nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(drop))
            hd=d if h==0 else h
            taxonomy=cfg.get('taxonomy_only',False)
            if cfg.get('taxonomy_angular',False) and (not taxonomy or cfg.get('taxonomy_residual',False) or arch!='attention'):
                raise ValueError('Angular taxonomy currently requires ordinary attention with compact disease marginalization')
            if taxonomy and (self.mode!='flat' or arch not in ['global','attention','class_attention','subtype_attention','context','residual','scale_consensus','scale_interaction','mirror_consensus'] or cfg.get('prototype',False)):
                raise ValueError('Compact taxonomy uses a flat parametric head without memory or prototype extras')
            if cfg.get('taxonomy_residual',False) and not taxonomy:
                raise ValueError('Support residuals require compact subtype marginalization')
            if taxonomy and cfg.get('taxonomy_residual',False):
                from taxonomy_head import SupportResidualHead
                self.flat=SupportResidualHead(hd,dims['disease_to_class'],dims['disease_patient_counts'],cfg)
            elif taxonomy and cfg.get('taxonomy_angular',False):
                from angular_taxonomy import AngularTaxonomy
                self.flat=AngularTaxonomy(hd,n_diseases,cfg)
            else:self.flat=nn.Linear(hd,n_diseases if taxonomy else 3)
            if taxonomy:
                mapping=dims['disease_to_class'];assert len(mapping)==n_diseases and set(mapping)=={0,1,2}
                self.register_buffer('subtype_to_class',F.one_hot(torch.tensor(mapping),3).float())
            if self.mode=='dual':
                self.binary=nn.Linear(hd,4)
                branches=4 if cfg.get('subtype_fusion',False) else 3
                self.mixture=nn.Linear(hd,branches)
                self.subtype=nn.Linear(hd,n_diseases)
                if cfg.get('subtype_fusion',False):
                    mapping=dims['disease_to_class']
                    assert len(mapping)==n_diseases and set(mapping)=={0,1,2}
                    self.register_buffer('subtype_to_class',F.one_hot(torch.tensor(mapping),3).float())
            if cfg.get('fine_contrastive',0.):
                self.fine_projection=nn.Sequential(nn.Linear(hd,64),nn.GELU(),nn.Linear(64,64))
            if cfg.get('prototype',False):
                self.prototype=nn.Parameter(torch.randn(3,3,hd)*.05)
                self.prototype_scale=nn.Parameter(torch.tensor(2.))
            if arch in MEMORY_ARCHES:
                self.memory_gate=nn.Linear(hd+(4 if cfg.get('memory_gate_evidence',False) else 0),1)
                nn.init.constant_(self.memory_gate.bias,-2.)
        if arch=='subtype_attention':
            if self.mode!='flat' or not cfg.get('taxonomy_only') or cfg.get('taxonomy_residual'):
                raise ValueError('Disease-specific regional evidence uses ordinary compact subtype marginalization')
            from subtype_query_head import SubtypeEvidenceQueries
            self.subtype_evidence=SubtypeEvidenceQueries(n_diseases,128,cfg)
            self.evidence_gate=nn.Linear(hd,n_diseases)
            nn.init.zeros_(self.evidence_gate.weight);nn.init.constant_(self.evidence_gate.bias,-2.)
        if cfg.get('taxonomy_second_order',False):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or cfg.get('taxonomy_angular') or cfg.get('taxonomy_residual'):
                raise ValueError('Second-order evidence requires ordinary attention and compact subtype marginalization')
            from second_order_head import RegionalSecondOrder
            self.second_order=RegionalSecondOrder(dims['regions'],n_diseases,cfg)
            self.second_order_gate=nn.Linear(hd,1)
            nn.init.zeros_(self.second_order_gate.weight);nn.init.constant_(self.second_order_gate.bias,-2.)
        if cfg.get('taxonomy_factorized',False):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order']):
                raise ValueError('Factorized taxonomy requires ordinary compact attention')
            from factorized_head import FactorizedCoarse
            self.factorized_coarse=FactorizedCoarse(hd,cfg.get('factorization_order','flat'))
        if cfg.get('region_graph_steps',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized']):
                raise ValueError('Spatial graph requires ordinary compact attention')
            from spatial_graph import SpatialGraph
            self.region_graph=SpatialGraph(128,32,cfg['region_graph_steps'])
        if cfg.get('support_group_threshold',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps']):
                raise ValueError('Support grouping requires ordinary compact attention')
            from support_group import SupportGroups
            self.support_groups=SupportGroups(dims['disease_to_class'],dims['disease_patient_counts'],cfg['support_group_threshold'])
        if cfg.get('context_query',False):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps','support_group_threshold']):
                raise ValueError('Context queries require ordinary compact attention')
            from context_attention import ContextQueries
            context_kind=cfg.get('query_context','global')
            if context_kind not in ['global','all']:
                raise ValueError('Query context must be global or all')
            self.context_queries=ContextQueries(256 if context_kind=='global' else 384,128,
                cfg.get('region_heads',2),cfg.get('context_query_rank',16),cfg.get('context_query_strength',1.))
        if cfg.get('attention_diversity_weight',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps','support_group_threshold','context_query']):
                raise ValueError('Attention diversity control requires ordinary compact attention')
            if cfg['attention_diversity_weight']<0 or not 0<=cfg.get('attention_diversity_margin',.9)<1:
                raise ValueError('Invalid attention diversity weight or margin')
        if cfg.get('taxonomy_subclusters',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps','support_group_threshold','context_query','attention_diversity_weight']):
                raise ValueError('Within-disease mixtures require ordinary compact attention')
            from subcluster_head import SubclusterEvidence
            self.disease_clusters=SubclusterEvidence(hd,dims['disease_patient_counts'],cfg['taxonomy_subclusters'],
                cfg.get('subcluster_shrink',10.),cfg.get('subcluster_init_std',.005))
        if cfg.get('taxonomy_shared_morphology',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps','support_group_threshold','context_query','attention_diversity_weight','taxonomy_subclusters']):
                raise ValueError('Shared morphology requires ordinary compact attention')
            from shared_morphology import SharedMorphology
            self.shared_morphology=SharedMorphology(hd,dims['disease_to_class'],dims['disease_patient_counts'],
                cfg['taxonomy_shared_morphology'],cfg.get('shared_prior_shrink',10.),cfg.get('shared_morphology_init_std',.005))
        if cfg.get('embedding_refinement_rank',0):
            if arch!='attention' or self.mode!='flat' or not cfg.get('taxonomy_only') or any(cfg.get(key) for key in ['taxonomy_angular','taxonomy_residual','taxonomy_second_order','taxonomy_factorized','region_graph_steps','support_group_threshold','context_query','attention_diversity_weight','taxonomy_subclusters','taxonomy_shared_morphology','patient_contrastive_mode']):
                raise ValueError('Embedding refinement requires ordinary compact attention')
            from embedding_refinement import EmbeddingRefinement
            # New deterministic branch does not shift the original dropout/permutation random stream.
            with torch.random.fork_rng(devices=[]):
                self.embedding_refinement=EmbeddingRefinement(hd,cfg['embedding_refinement_rank'],cfg.get('embedding_refinement_steps',1),.1)

    def forward(self,x):
        z=x['global']
        local_logits=None
        scale_views=None
        subtype_local=None
        mirror_views=None
        if self.arch=='mirror_consensus':
            z,mirror_views,mirror_attention,mirror_disagreement=self.mirror_pool(x)
        if self.arch in ['scale_consensus','scale_interaction']:z,scale_views,scale_weights=self.scale_pool(x)
        if self.arch=='context': z=torch.cat([z,x['mid'],x['dispersion']],-1)
        if self.arch=='residual':z=z+self.adapter_scale.tanh()*self.adapter(z)
        if self.arch in ['attention','class_attention','subtype_attention','memory_attention']:
            tokens=F.gelu(self.region_proj(x['regions']))
            if self.cfg.get('region_graph_steps',0):
                tokens=self.region_graph(tokens,x.get('region_mask') if self.cfg.get('use_region_mask',True) else None)
            scores=self.region_gate(tokens)
            if self.cfg.get('context_query',False):
                query_context=F.gelu(self.global_proj(z))
                if self.cfg.get('query_context','global')=='all':
                    query_context=torch.cat([query_context,F.gelu(self.mid_proj(x['mid'])),F.gelu(self.std_proj(x['dispersion']))],-1)
                context_score_delta=self.context_queries(tokens,query_context)
                scores=scores+context_score_delta
            if 'region_mask' in x and self.cfg.get('use_region_mask',True):
                valid=x['region_mask']>.5
                valid=torch.where(valid.any(1,keepdim=True),valid,torch.ones_like(valid))
                scores=scores.masked_fill(~valid[:,:,None],-1e4)
            attention=scores.softmax(1)
            pooled=torch.einsum('btk,btd->bkd',attention,tokens).flatten(1)
            z=torch.cat([F.gelu(self.global_proj(z)),pooled,F.gelu(self.mid_proj(x['mid'])),F.gelu(self.std_proj(x['dispersion']))],-1)
            if self.arch=='subtype_attention':
                subtype_local,subtype_attention=self.subtype_evidence(tokens,x.get('region_mask') if self.cfg.get('use_region_mask',True) else None)
        if self.arch=='cross_attention':
            global_views=z.chunk(2,-1);region_views=x['regions'].chunk(2,-1)
            queries=[F.gelu(self.view_proj[i](global_views[i]))[:,None] for i in range(2)]
            patches=[F.gelu(self.view_proj[i](region_views[i])) for i in range(2)]
            cross=[self.cross_attention(queries[i],patches[1-i],patches[1-i],need_weights=False)[0][:,0] for i in range(2)]
            z=torch.cat([queries[0][:,0],queries[1][:,0],*cross,F.gelu(self.mid_proj(x['mid'])),F.gelu(self.std_proj(x['dispersion']))],-1)
        if self.arch=='evidence_mil':z,local_logits=self.evidence(x)
        if self.mode in ['cascade_i','cascade_n']:
            l1,l2=self.stage1(z),self.stage2(z)
            a,b=l1.softmax(-1)[:,1],l2.softmax(-1)[:,1]
            p=torch.stack([a,(1-a)*b,(1-a)*(1-b)],1) if self.mode=='cascade_i' else torch.stack([a*b,a*(1-b),1-a],1)
            return {'p':p,'gates':torch.stack([a,b],1),'stage1':l1,'stage2':l2}
        z=self.encoder(z)
        if self.cfg.get('embedding_refinement_rank',0):
            original_embedding=z
            z=self.embedding_refinement(z)
        logits=self.flat(z)
        if self.cfg.get('taxonomy_only',False):
            if self.cfg.get('taxonomy_subclusters',0):
                subcluster_evidence,subcluster_logits=self.disease_clusters(z)
                logits=logits+subcluster_evidence
            if self.cfg.get('taxonomy_shared_morphology',0):
                morphology_evidence,morphology_components,morphology_priors=self.shared_morphology(z)
                logits=logits+morphology_evidence
            if subtype_local is not None:
                base_subtype_logits=logits;evidence_gate=self.evidence_gate(z).sigmoid()
                logits=logits+evidence_gate*subtype_local
            if self.cfg.get('taxonomy_second_order',False):
                base_second_order_logits=logits
                second_order_logits,second_order_embedding=self.second_order(x['regions'],x.get('region_mask'))
                second_order_gate=self.second_order_gate(z).sigmoid()
                logits=logits+second_order_gate*second_order_logits
            p=logits.softmax(-1)@self.subtype_to_class
            out={'p':p,'flat':p.clamp(min=1e-7).log(),'subtype':logits,'subtype_probability':p,'embedding':z}
            if self.cfg.get('embedding_refinement_rank',0):
                out.update(embedding_before_refinement=original_embedding,refinement_delta=z-original_embedding)
            if self.cfg.get('taxonomy_subclusters',0):
                out.update(subcluster_evidence=subcluster_evidence,subcluster_logits=subcluster_logits,
                    subcluster_regularizer=self.disease_clusters.regularization())
            if self.cfg.get('taxonomy_shared_morphology',0):
                out.update(morphology_evidence=morphology_evidence,morphology_components=morphology_components,
                    morphology_log_priors=morphology_priors,morphology_regularizer=self.shared_morphology.regularization())
            if self.cfg.get('context_query',False):
                out.update(context_score_delta=context_score_delta,region_attention=attention)
            if self.cfg.get('attention_diversity_weight',0):
                out['region_attention']=attention
            if self.cfg.get('taxonomy_factorized',False):
                from factorized_head import conditional_disease_logp
                conditional_logp=conditional_disease_logp(logits,self.subtype_to_class)
                p,coarse_gates=self.factorized_coarse(z)
                disease_probability=conditional_logp.exp()*(p@self.subtype_to_class.T)
                out.update(p=p,flat=p.clamp(min=1e-7).log(),conditional_disease_logp=conditional_logp,
                    factorized_disease_probability=disease_probability,subtype_probability=disease_probability@self.subtype_to_class,
                    factorized_gates=coarse_gates)
            if self.cfg.get('support_group_threshold',0):
                out.update(grouped_subtype_logits=self.support_groups(logits),
                    disease_to_support_group=self.support_groups.disease_to_group)
            if self.cfg.get('taxonomy_angular',False):
                out.update(subtype_cosine=self.flat.similarities(z),subtype_scale=self.flat.scale())
            if self.cfg.get('taxonomy_second_order',False):
                out.update(second_order_logits=second_order_logits,second_order_embedding=second_order_embedding,
                    second_order_gate=second_order_gate,base_second_order_logits=base_second_order_logits)
            if mirror_views is not None:
                mirror_logits=self.flat(self.encoder(mirror_views))
                out.update(mirror_view_logits=mirror_logits,
                    mirror_view_probability=mirror_logits.softmax(-1)@self.subtype_to_class,
                    mirror_attention=mirror_attention,mirror_disagreement=mirror_disagreement)
            if subtype_local is not None:out.update(local_subtype_logits=subtype_local,base_subtype_logits=base_subtype_logits,query_attention=subtype_attention,query_gate=evidence_gate)
            if self.cfg.get('taxonomy_residual',False):out['taxonomy_regularizer']=self.flat.regularization()
            if scale_views is not None:
                scale_logits=self.flat(self.encoder(scale_views))
                scale_prob=scale_logits.softmax(-1)@self.subtype_to_class
                blend=self.cfg.get('scale_probability_blend',0.)
                if not 0<=blend<=1:raise ValueError('Scale probability blend must be in [0,1]')
                out.update(p=(1-blend)*p+blend*scale_prob.mean(1),scale_logits=scale_logits,scale_probability=scale_prob,scale_weights=scale_weights)
            if self.cfg.get('fine_contrastive',0.):out['fine_embedding']=self.fine_projection(z)
            return out
        if local_logits is not None:logits=logits+self.cfg.get('evidence_residual',.2)*local_logits
        if self.cfg.get('prototype',False):
            sim=torch.einsum('bd,ckd->bck',F.normalize(z,dim=-1),F.normalize(self.prototype,dim=-1))
            logits=logits+.5*torch.logsumexp(sim*self.prototype_scale.exp().clamp(max=30),dim=-1)
        p=logits.softmax(-1)
        out={'p':p,'flat':logits,'embedding':z}
        if self.cfg.get('fine_contrastive',0.):out['fine_embedding']=self.fine_projection(z)
        if local_logits is not None:out['local_logits']=local_logits
        if self.mode=='dual':
            binary=self.binary(z);q=binary.sigmoid();a,b,c,d=q.unbind(1)
            pi=torch.stack([a,(1-a)*b,(1-a)*(1-b)],1)
            pn=torch.stack([c*d,c*(1-d),1-c],1)
            mix=self.mixture(z).softmax(-1)
            subtype=self.subtype(z);views=[p,pi,pn]
            if self.cfg.get('subtype_fusion',False):
                subtype_probability=subtype.softmax(-1)@self.subtype_to_class
                views.append(subtype_probability);out['subtype_probability']=subtype_probability
            out.update(p=(torch.stack(views,1)*mix[:,:,None]).sum(1),binary=binary,pi=pi,pn=pn,subtype=subtype,mixture_weights=mix)
        if self.arch in MEMORY_ARCHES:
            memory_p,memory_scores=self.memory(x['global'],x.get('_patient_ids'))
            gate_input=z
            if self.cfg.get('memory_gate_evidence',False):
                base_entropy=-(out['p']*out['p'].clamp(min=1e-7).log()).sum(1,keepdim=True)
                memory_entropy=-(memory_p*memory_p.clamp(min=1e-7).log()).sum(1,keepdim=True)
                agreement=(out['p']*memory_p).sum(1,keepdim=True)
                top=memory_p.topk(2,dim=1).values;margin=top[:,:1]-top[:,1:]
                gate_input=torch.cat([z,base_entropy,memory_entropy,agreement,margin],dim=1)
            gate=self.memory_gate(gate_input).sigmoid()*self.cfg.get('memory_cap',.75)
            out.update(parametric_probability=out['p'],p=(1-gate)*out['p']+gate*memory_p,memory_probability=memory_p,memory_gate=gate,memory_scores=memory_scores)
        return out

def loss_fn(out,y,disease,weights,cfg):
    mode=cfg['mode'];w=weights['leaf']
    if mode in ['cascade_i','cascade_n']:
        if mode=='cascade_i': t1=(y==0).long();t2=(y==1).long();mask=y!=0
        else: t1=(y!=2).long();t2=(y==0).long();mask=y!=2
        l1=(F.cross_entropy(out['stage1'],t1,reduction='none')*weights['stage1']).mean()
        l2=(F.cross_entropy(out['stage2'],t2,reduction='none')*weights['stage2']).sum()/mask.sum().clamp(min=1)
        return l1+l2
    logp=out['p'].clamp(min=1e-7).log()
    nll=F.nll_loss(logp,y,reduction='none')
    smooth=cfg.get('label_smoothing',0.)
    if smooth:nll=(1-smooth)*nll-smooth*logp.mean(1)
    gamma=cfg.get('focal_gamma',0.)
    if gamma:
        # Fractional powers need a positive floor when a confident softmax rounds to 1.
        focal_base=(1-out['p'].gather(1,y[:,None]).squeeze(1)).clamp(min=1e-6)
        nll=nll*focal_base.pow(gamma)
    loss=(nll*w).mean()
    if cfg.get('attention_diversity_weight',0):
        from attention_diversity import attention_overlap_penalty
        diversity=attention_overlap_penalty(out['region_attention'],cfg.get('attention_diversity_margin',.9))
        loss=loss+cfg['attention_diversity_weight']*(diversity*w).mean()
    if cfg.get('taxonomy_subclusters',0):
        loss=loss+cfg.get('subcluster_penalty',.01)*out['subcluster_regularizer']
    if cfg.get('taxonomy_shared_morphology',0):
        loss=loss+cfg.get('shared_morphology_penalty',.01)*out['morphology_regularizer']
    if cfg.get('taxonomy_only',False):
        if cfg.get('taxonomy_angular',False):
            from angular_taxonomy import margin_logits
            fine_logits=margin_logits(out['subtype_cosine'],out['subtype_scale'],disease,
                cfg.get('angular_margin_kind','none'),cfg.get('angular_margin',0.))
        else:fine_logits=out['subtype']
        if cfg.get('support_group_threshold',0):
            fine_target=torch.where(disease>=0,out['disease_to_support_group'][disease.clamp(min=0)],torch.full_like(disease,-1))
            fine_aux=F.cross_entropy(out['grouped_subtype_logits'],fine_target,reduction='none',ignore_index=-1)
        elif cfg.get('taxonomy_factorized',False) and cfg.get('factorization_conditional_aux',True):
            fine_aux=F.nll_loss(out['conditional_disease_logp'],disease,reduction='none',ignore_index=-1)
        else:fine_aux=F.cross_entropy(fine_logits,disease,reduction='none',ignore_index=-1)
        loss=loss+cfg.get('aux',.6)*(fine_aux*w).mean()
        if 'second_order_logits' in out and cfg.get('covariance_aux',0.):
            covariance_loss=F.cross_entropy(out['second_order_logits'],disease,reduction='none',ignore_index=-1)
            loss=loss+cfg['covariance_aux']*(covariance_loss*w).mean()
        if 'local_subtype_logits' in out and cfg.get('query_aux',0.):
            local_fine=F.cross_entropy(out['local_subtype_logits'],disease,reduction='none',ignore_index=-1)
            loss=loss+cfg['query_aux']*(local_fine*w).mean()
        if 'taxonomy_regularizer' in out:loss=loss+cfg.get('residual_penalty',.001)*out['taxonomy_regularizer']
        if 'scale_logits' in out:
            fine=out['scale_logits'];n,views,classes=fine.shape
            targets=disease[:,None].expand(-1,views).reshape(-1)
            view_aux=F.cross_entropy(fine.reshape(-1,classes),targets,reduction='none',ignore_index=-1).reshape(n,views).mean(1)
            loss=loss+cfg.get('scale_aux',.3)*(view_aux*w).mean()
            q=out['scale_probability'];center=q.mean(1,keepdim=True)
            discrepancy=(q*(q.clamp(min=1e-7).log()-center.clamp(min=1e-7).log())).sum(-1).mean(1)
            loss=loss+cfg.get('scale_consistency',.05)*(discrepancy*w).mean()
        if 'mirror_view_logits' in out:
            fine=out['mirror_view_logits'];n,views,classes=fine.shape
            targets=disease[:,None].expand(-1,views).reshape(-1)
            auxiliary=F.cross_entropy(fine.reshape(-1,classes),targets,reduction='none',ignore_index=-1).reshape(n,views).mean(1)
            loss=loss+cfg.get('mirror_aux',0.)*(auxiliary*w).mean()
            q=out['mirror_view_probability'];center=q.mean(1,keepdim=True)
            discrepancy=(q*(q.clamp(min=1e-7).log()-center.clamp(min=1e-7).log())).sum(-1).mean(1)
            loss=loss+cfg.get('mirror_consistency',0.)*(discrepancy*w).mean()
    if mode=='dual':
        targets=torch.stack([(y==0).float(),(y==1).float(),(y!=2).float(),(y==0).float()],1)
        mask=torch.stack([torch.ones_like(y),y!=0,torch.ones_like(y),y!=2],1).float()
        binary=F.binary_cross_entropy_with_logits(out['binary'],targets,reduction='none')
        loss=loss+.25*(binary*mask*w[:,None]).sum()/mask.sum()
        m=(out['pi']+out['pn'])/2
        js=.5*((out['pi']*(out['pi'].clamp(min=1e-7).log()-m.clamp(min=1e-7).log())).sum(1)+(out['pn']*(out['pn'].clamp(min=1e-7).log()-m.clamp(min=1e-7).log())).sum(1))
        loss=loss+cfg.get('consistency',.1)*(js*w).mean()
        aux=F.cross_entropy(out['subtype'],disease,reduction='none',ignore_index=-1)
        loss=loss+cfg.get('aux',.2)*(aux*w).mean()
        if 'subtype_probability' in out and cfg.get('subtype_consistency',0.):
            a,b=out['p'],out['subtype_probability'];mid=(a+b)/2
            discrepancy=.5*((a*(a.clamp(min=1e-7).log()-mid.clamp(min=1e-7).log())).sum(1)+(b*(b.clamp(min=1e-7).log()-mid.clamp(min=1e-7).log())).sum(1))
            loss=loss+cfg['subtype_consistency']*(discrepancy*w).mean()
    if 'local_logits' in out and cfg.get('evidence_aux',0.):
        local_loss=F.cross_entropy(out['local_logits'],y,reduction='none')
        loss=loss+cfg['evidence_aux']*(local_loss*w).mean()
    if cfg.get('contrastive',0.):
        emb=F.normalize(out['embedding'],dim=-1);sim=emb@emb.T/.15
        identity=torch.eye(len(y),device=y.device,dtype=torch.bool)
        sim=sim.masked_fill(identity,-1e4)
        positives=(y[:,None]==y[None,:]) & (weights['patient_ids'][:,None]!=weights['patient_ids'][None,:])
        count=positives.sum(1);valid=count>0
        if valid.any():
            contrast=-(F.log_softmax(sim,dim=1)*positives).sum(1)/count.clamp(min=1)
            loss=loss+cfg['contrastive']*(contrast[valid]*w[valid]).mean()
    if cfg.get('fine_contrastive',0.):
        fine_target=disease
        if cfg.get('support_group_threshold',0) and cfg.get('support_group_contrastive',False):
            fine_target=torch.where(disease>=0,out['disease_to_support_group'][disease.clamp(min=0)],torch.full_like(disease,-1))
        if cfg.get('patient_contrastive_mode'):
            from patient_contrastive import patient_contrastive_loss
            loss=loss+cfg['fine_contrastive']*patient_contrastive_loss(out['fine_embedding'],fine_target,
                weights['patient_ids'],w,cfg['patient_contrastive_mode'])
        else:
            emb=F.normalize(out['fine_embedding'],dim=-1);sim=emb@emb.T/.15
            identity=torch.eye(len(y),device=y.device,dtype=torch.bool);sim=sim.masked_fill(identity,-1e4)
            positives=(fine_target[:,None]==fine_target[None,:]) & (fine_target[:,None]>=0) & (weights['patient_ids'][:,None]!=weights['patient_ids'][None,:])
            count=positives.sum(1);valid=count>0
            if valid.any():
                fine_loss=-(F.log_softmax(sim,dim=1)*positives).sum(1)/count.clamp(min=1)
                loss=loss+cfg['fine_contrastive']*(fine_loss[valid]*w[valid]).mean()
    return loss

@torch.inference_mode()
def predict(model,x,batch=256):
    model.eval();ps=[];gs=[]
    n=len(x['global'])
    for start in range(0,n,batch):
        o=model({k:v[start:start+batch] for k,v in x.items()});ps.append(o['p'].cpu().numpy())
        if 'gates' in o: gs.append(o['gates'].cpu().numpy())
    return np.concatenate(ps),np.concatenate(gs) if gs else np.zeros((n,0),dtype='float32')

def train_one(frame,features,digest,cfg,run_dir,train_mask=None,val_mask=None,model_class=None,training_loss=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'summary.json').exists(): return json.loads((run_dir/'summary.json').read_text())
    seed=20261124+cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    train_mask=frame.split.eq('train').to_numpy() if train_mask is None else train_mask
    val_mask=frame.split.eq('val').to_numpy() if val_mask is None else val_mask
    normalized,stats=normalize(features,train_mask)
    tf,vf=frame.loc[train_mask].reset_index(drop=True),frame.loc[val_mask].reset_index(drop=True)
    tx={k:torch.from_numpy(v[train_mask]).cuda() for k,v in normalized.items()}
    vx={k:torch.from_numpy(v[val_mask]).cuda() for k,v in normalized.items()}
    y=torch.tensor(tf.label.to_numpy(),device='cuda',dtype=torch.long)
    diseases=sorted(tf.disease.unique());dm={d:i for i,d in enumerate(diseases)}
    disease=torch.tensor(tf.disease.map(dm).to_numpy(),device='cuda',dtype=torch.long)
    weights={'leaf':sample_weights(tf,tf.label),'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')}
    if cfg['mode']=='cascade_i':
        weights.update(stage1=sample_weights(tf,tf.label.eq(0).astype(int)),stage2=sample_weights(tf,tf.label.eq(1).astype(int),tf.label.ne(0)))
    elif cfg['mode']=='cascade_n':
        weights.update(stage1=sample_weights(tf,tf.label.ne(2).astype(int)),stage2=sample_weights(tf,tf.label.eq(0).astype(int),tf.label.ne(2)))
    wt={k:torch.from_numpy(v).cuda() for k,v in weights.items()}
    dims={k:v.shape[-1] for k,v in features.items()}
    memory_audit=None
    if cfg.get('subtype_fusion',False) or cfg.get('taxonomy_only',False):
        disease_class=tf.groupby('disease').label.agg(['nunique','first'])
        assert disease_class['nunique'].eq(1).all(),'Every subtype must map to exactly one requested class'
        dims['disease_to_class']=[int(disease_class.loc[name,'first']) for name in diseases]
        if cfg.get('taxonomy_residual',False) or cfg.get('support_group_threshold',0) or cfg.get('taxonomy_subclusters',0) or cfg.get('taxonomy_shared_morphology',0):
            patient_counts=tf.groupby('disease').patient_group.nunique()
            dims['disease_patient_counts']=[int(patient_counts[name]) for name in diseases]
    if cfg['arch'] in MEMORY_ARCHES:dims['memory_count']=len(tf[['patient_group','label']].drop_duplicates())
    model=(Model if model_class is None else model_class)(cfg,dims,len(diseases)).cuda()
    if cfg['arch'] in MEMORY_ARCHES:
        info=model.memory.initialize(normalized['global'][train_mask],tf.label.to_numpy(),weights['patient_ids'],stats['global']['mean'],stats['global']['std'])
        patient_names=dict(enumerate(pd.unique(tf.patient_group)))
        for row in info:row['patient_group']=patient_names[row['patient_token']]
        pd.DataFrame(info).to_csv(run_dir/'memory_support.csv',index=False)
        memory_audit={'source':'Only this fit mask; memory rebuilt independently for every outer CV fold','support_patient_groups':sorted(tf.patient_group.unique().tolist()),'prototypes':len(info),'prototypes_per_class':pd.Series([r['label'] for r in info]).value_counts().sort_index().to_dict(),'training_query_own_patient_excluded':True}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    criterion=loss_fn if training_loss is None else training_loss
    history=[];best=None;wait=0;start=time.time()
    batch=cfg.get('batch',256)
    for epoch in range(cfg.get('epochs',120)):
        model.train();permutation=torch.randperm(len(y),device='cuda');loss_sum=0
        for begin in range(0,len(y),batch):
            ix=permutation[begin:begin+batch];optimizer.zero_grad(set_to_none=True)
            inputs={k:v[ix] for k,v in tx.items()}
            if cfg['arch'] in MEMORY_ARCHES:inputs['_patient_ids']=wt['patient_ids'][ix]
            out=model(inputs)
            loss=criterion(out,y[ix],disease[ix],{k:v[ix] for k,v in wt.items()},cfg)
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss: {cfg}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();loss_sum+=loss.item()*len(ix)
        p,g=predict(model,vx)
        pred=routed(g,cfg['mode']) if g.shape[1] else p.argmax(1)
        m=metrics(vf.label,p,pred);key=(m['macro_f1'],-m['log_loss'])
        history.append({'epoch':epoch+1,'train_loss':loss_sum/len(y),'val':m})
        if best is None or key>best:
            best=key;wait=0
            checkpoint={'cfg':cfg,'dims':dims,'diseases':diseases,'state_dict':{k:v.cpu().clone() for k,v in model.state_dict().items()},'stats':stats,'manifest_sha256':digest,'epoch':epoch+1,'val':m}
            if memory_audit is not None:checkpoint['memory_audit']=memory_audit
            np.savez_compressed(run_dir/'val_predictions.npz',p=p,g=g)
        else: wait+=1
        if wait>=cfg.get('patience',18): break
    torch.save(checkpoint,run_dir/'model.pt');dump(run_dir/'history.json',history)
    summary={'cfg':cfg,'epoch':checkpoint['epoch'],'val':checkpoint['val'],'seconds':time.time()-start,'parameters':sum(p.numel() for p in model.parameters()),'manifest_sha256':digest}
    if memory_audit is not None:summary['memory_audit']=memory_audit
    dump(run_dir/'summary.json',summary)
    console_summary={k:v for k,v in summary.items() if k!='memory_audit'}
    if memory_audit is not None:console_summary['memory_prototypes']=memory_audit['prototypes']
    print(json.dumps({'event':'trial_complete','path':str(run_dir),'summary':console_summary}),flush=True)
    del model,tx,vx;torch.cuda.empty_cache()
    return summary

def summarize_group(group_dir,frame):
    summaries=sorted(group_dir.glob('seed_*/summary.json'))
    if not summaries: return None
    preds=[np.load(p.parent/'val_predictions.npz') for p in summaries]
    cfg=json.loads(summaries[0].read_text())['cfg'];mode=cfg['mode']
    gates=np.mean([p['g'] for p in preds],0)
    probabilities=leaf_probs(gates,mode) if gates.shape[1] else np.mean([p['p'] for p in preds],0)
    y=frame.loc[frame.split.eq('val'),'label'].to_numpy()
    thresholds=tune_thresholds(y,gates,mode) if gates.shape[1] else None
    pred=routed(gates,mode,thresholds) if gates.shape[1] else probabilities.argmax(1)
    summary={'group':group_dir.name,'mode':mode,'config':{k:v for k,v in cfg.items() if k!='seed'},'seeds':len(preds),'thresholds':thresholds,'val':metrics(y,probabilities,pred),'seed_macro_f1':[json.loads(s.read_text())['val']['macro_f1'] for s in summaries],'checkpoints':[str(p.parent/'model.pt') for p in summaries]}
    dump(group_dir/'ensemble.json',summary)
    np.savez_compressed(group_dir/'val_ensemble.npz',p=probabilities,g=gates,pred=pred)
    return summary

def configuration(arch,mode,hidden=256,dropout=.3,lr=.001,wd=.01,**extra):
    return dict(arch=arch,mode=mode,hidden=hidden,dropout=dropout,lr=lr,wd=wd,epochs=120,patience=18,**extra)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['baseline','explore'],default='baseline');parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--features',default='convnext_320');args=parser.parse_args()
    frame,features,digest=load_data(args.features)
    configs=[]
    if args.phase=='baseline':
        configs=[configuration('global',mode,h) for mode in ['flat','cascade_i','cascade_n'] for h in [128,256]]
        configs += [configuration('global','flat',0,0)]
    else:
        configs=[configuration('global','dual',h,aux=aux) for h,aux in [(128,.2),(256,.2),(256,0)]]
        configs += [configuration('context',mode,256,dropout=.4) for mode in ['flat','dual']]
        configs += [configuration('attention',mode,h,dropout=.4) for mode,h in [('flat',128),('dual',128),('dual',256)]]
        configs += [configuration('global','flat',256,prototype=True),configuration('attention','dual',128,aux=.4,consistency=.3)]
    out=ROOT/'runs'/args.features/args.phase;out.mkdir(parents=True,exist_ok=True)
    all_groups=[]
    for i,cfg in enumerate(configs):
        group=out/f'{i:02d}_{cfg["arch"]}_{cfg["mode"]}_h{cfg["hidden"]}'
        for seed in range(args.seeds): train_one(frame,features,digest,{**cfg,'seed':seed},group/f'seed_{seed}')
        summary=summarize_group(group,frame);all_groups.append(summary)
        dump(out/'leaderboard.json',sorted(all_groups,key=lambda x:x['val']['macro_f1'],reverse=True))
        print(json.dumps({'event':'ensemble_complete','result':summary}),flush=True)
    selected={}
    for group in all_groups:
        mode=group['mode']
        if mode not in selected or group['val']['macro_f1']>selected[mode]['val']['macro_f1']: selected[mode]=group
    dump(out/'selection.json',{'phase':args.phase,'manifest_sha256':digest,'selection_rule':'Maximum validation image macro-F1; equal-probability three-seed ensemble; cascade thresholds selected only on validation; external labels not read for selection.','selected':selected,'best':max(all_groups,key=lambda x:x['val']['macro_f1'])})

if __name__=='__main__': main()

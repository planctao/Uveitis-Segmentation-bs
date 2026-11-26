"""Fixed image-only photometric counterfactuals and unsupervised calibration."""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.extmath import randomized_svd

from bs.severity_features import retinal_fov

SYNTHETIC_METHODS=['synthetic_direction','meanmatched_direction',
                   'gamma_orthogonal_direction','synthetic_oriented_pc1']


def synthetic_views(rgb,uid):
    rgb=np.asarray(rgb,dtype=np.float64)
    gray=rgb@np.array([.299,.587,.114])
    fov=retinal_fov(gray)
    seed=int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    centers=np.argwhere(fov)[rng.integers(0,fov.sum(),size=3)]
    yy,xx=np.indices(gray.shape);field=np.zeros(gray.shape)
    for cy,cx in centers:
        sigma=rng.uniform(8,24)
        field+=np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma**2))
    field=field/field.max()*fov
    low=rgb+.08*field[...,None]*(1-rgb)
    high=rgb+.16*field[...,None]*(1-rgb)
    c=(high-rgb)[fov].sum()/max((1-rgb)[fov].sum(),1e-15)
    control=rgb+c*fov[...,None]*(1-rgb)
    quantized=[np.rint(np.clip(a,0,1)*255).astype(np.uint8) for a in [low,high,control]]
    diagnostics=dict(meanmatch_error_before_quantization=float(np.mean(high[fov])-np.mean(control[fov])),
                     meanmatch_error_after_quantization=float(np.mean(quantized[1][fov])/255-np.mean(quantized[2][fov])/255),
                     uniform_control_c=float(c),field_fov_mean=float(field[fov].mean()))
    return quantized,diagnostics


class SyntheticDirectionOrderer:
    """Known generated intensity increases provide direction; no real labels."""
    def __init__(self,method,seed=42):
        if method not in SYNTHETIC_METHODS:raise ValueError(method)
        self.method,self.seed=method,seed

    def transform_features(self,z):
        return self.scaler.transform(normalize(np.asarray(z,dtype=np.float64)))

    def fit(self,z,low,high,control,gamma08,gamma12):
        self.scaler=StandardScaler().fit(normalize(np.asarray(z,dtype=np.float64)))
        original=self.transform_features(z);higher=self.transform_features(high)
        controlled=self.transform_features(control)
        self.diagnostics=dict(method=self.method,seed=self.seed,n_fit=len(z),manual_grade_labels_used=False,
                              masks_used=False,synthetic_supervision=True)
        if self.method=='synthetic_oriented_pc1':
            self.pca=PCA(n_components=1,random_state=self.seed).fit(original)
            mean_change=float((self.pca.transform(higher)-self.pca.transform(original)).mean())
            self.orientation=1 if mean_change>=0 else -1
            self.diagnostics['synthetic_mean_pc1_change']=mean_change
        else:
            direction=(higher-original).mean(0) if self.method=='synthetic_direction' else (higher-controlled).mean(0)
            before=float(np.linalg.norm(direction))
            if self.method=='gamma_orthogonal_direction':
                gamma_changes=np.concatenate([self.transform_features(gamma08)-original,self.transform_features(gamma12)-original])
                _,singular,vectors=randomized_svd(gamma_changes,n_components=8,random_state=self.seed,n_iter=7)
                direction-=vectors.T@(vectors@direction)
                self.diagnostics.update(nuisance_singular_values=singular.tolist(),nuisance_dimensions=8,
                                        retained_direction_norm_fraction=float(np.linalg.norm(direction)/max(before,1e-15)))
            self.diagnostics['direction_norm']=float(np.linalg.norm(direction))
            if np.linalg.norm(direction)<1e-10:raise ValueError('Synthetic direction degenerates')
            self.direction=direction/np.linalg.norm(direction)
        scores=self.raw_score(z)
        self.sorted_scores=np.sort(scores);self.cutpoints=np.quantile(scores,[.25,.5,.75])
        low_score,high_score=self.raw_score(low),self.raw_score(high)
        self.diagnostics.update(train_low_gt_original_fraction=float((low_score>scores).mean()),
                                train_high_gt_low_fraction=float((high_score>low_score).mean()),
                                train_monotonic_fraction=float(((scores<low_score)&(low_score<high_score)).mean()))
        return self

    def raw_score(self,z):
        features=self.transform_features(z)
        if self.method=='synthetic_oriented_pc1':return self.pca.transform(features)[:,0]*self.orientation
        return features@self.direction

    def predict(self,z,f=None):
        scores=self.raw_score(z)
        rank=(np.searchsorted(self.sorted_scores,scores,'left')+np.searchsorted(self.sorted_scores,scores,'right'))/(2*len(self.sorted_scores))
        levels=np.searchsorted(self.cutpoints,scores,'right')
        return rank,levels

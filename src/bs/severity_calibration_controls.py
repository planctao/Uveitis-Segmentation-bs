"""Follow-up factorial and rank/strength matched projection controls."""
import numpy as np
from sklearn.utils.extmath import randomized_svd

from bs.severity_synthetic import SyntheticDirectionOrderer

CONTROL_METHODS=['synthetic_gamma_orthogonal_direction',
                 'meanmatched_random_rank_projection','meanmatched_random_norm_projection']


class CalibrationControlOrderer(SyntheticDirectionOrderer):
    def __init__(self,method,seed=42):
        if method not in CONTROL_METHODS:raise ValueError(method)
        self.method,self.seed=method,seed

    def fit(self,z,low,high,control,gamma08,gamma12):
        name=self.method
        self.method='synthetic_direction' if name=='synthetic_gamma_orthogonal_direction' else 'meanmatched_direction'
        super().fit(z,low,high,control,gamma08,gamma12)
        self.method=name
        original=self.transform_features(z);higher=self.transform_features(high);controlled=self.transform_features(control)
        direction=(higher-original).mean(0) if name=='synthetic_gamma_orthogonal_direction' else (higher-controlled).mean(0)
        norm=float(np.linalg.norm(direction))
        gamma=np.concatenate([self.transform_features(gamma08)-original,self.transform_features(gamma12)-original])
        _,singular,vectors=randomized_svd(gamma,n_components=8,random_state=self.seed,n_iter=7)
        actual=direction-vectors.T@(vectors@direction)
        reference_fraction=float(np.linalg.norm(actual)/norm)
        rng=np.random.default_rng(self.seed)
        if name=='synthetic_gamma_orthogonal_direction':
            projected=actual
        elif name=='meanmatched_random_rank_projection':
            basis=np.linalg.qr(rng.normal(size=(len(direction),8)))[0]
            projected=direction-basis@(basis.T@direction)
        else:
            # Randomize orientation while matching rank and removed norm.
            u=direction/norm
            v=rng.normal(size=len(direction));v-=u*(u@v);v/=np.linalg.norm(v)
            fraction=np.clip(reference_fraction,0,1)
            first=np.sqrt(max(1-fraction**2,0))*u+fraction*v
            remaining=rng.normal(size=(len(direction),7))
            remaining-=u[:,None]*(u@remaining)[None,:]+v[:,None]*(v@remaining)[None,:]
            others=np.linalg.qr(remaining)[0]
            basis=np.column_stack([first,others])
            projected=direction-basis@(basis.T@direction)
            assert np.isclose(np.linalg.norm(projected)/norm,reference_fraction,atol=1e-10)
        if np.linalg.norm(projected)<1e-10:raise ValueError('Control direction degenerates')
        self.direction=projected/np.linalg.norm(projected)
        scores=self.raw_score(z);self.sorted_scores=np.sort(scores);self.cutpoints=np.quantile(scores,[.25,.5,.75])
        lows=self.raw_score(low);highs=self.raw_score(high)
        self.diagnostics.update(method=name,projection_rank=8,
            reference_gamma_retained_fraction=reference_fraction,
            actual_retained_fraction=float(np.linalg.norm(projected)/norm),
            nuisance_singular_values=singular.tolist(),
            train_low_gt_original_fraction=float((lows>scores).mean()),
            train_high_gt_low_fraction=float((highs>lows).mean()),
            train_monotonic_fraction=float(((scores<lows)&(lows<highs)).mean()))
        return self

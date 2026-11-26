"""Fixed image-only ordinal baselines. Clinical labels/masks are not inputs."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize

from bs.severity_features import ANCHOR_COLUMN, MORPHOLOGY_COLUMNS, NUISANCE_COLUMNS

METHODS = ['mean', 'absolute_bright', 'local_anchor', 'morphology_pc1',
           'dino_pc1', 'dino_kmeans', 'dino_diffusion', 'dino_residual_diffusion']


def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.
    return float(spearmanr(a, b).statistic)


class DiffusionCoordinate:
    """alpha=1 kNN diffusion map with a fixed, label-free anchor selection."""
    def fit(self, x, anchor, seed=42):
        n=len(x); k=min(30, n-1)
        self.neighbors=NearestNeighbors(n_neighbors=k+1).fit(x)
        distances, indices=self.neighbors.kneighbors(x)
        self.scales=np.maximum(distances[:, -1], 1e-8)
        distances,indices=distances[:,1:],indices[:,1:]
        weights=np.exp(-distances**2/(self.scales[:,None]*self.scales[indices]))
        graph=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(n),k),indices.ravel())),shape=(n,n))
        graph=graph.maximum(graph.T)+sparse.eye(n)
        n_components,_=connected_components(graph,directed=False)
        if n_components != 1:
            raise ValueError(f'Diffusion graph has {n_components} components; ordinal axis is not identifiable.')
        self.density=np.asarray(graph.sum(1)).ravel()
        kernel=sparse.diags(1/self.density)@graph@sparse.diags(1/self.density)
        degree=np.asarray(kernel.sum(1)).ravel()
        sym=sparse.diags(1/np.sqrt(degree))@kernel@sparse.diags(1/np.sqrt(degree))
        eigenvalues,eigenvectors=eigsh(sym,k=min(7,n-2),which='LA',v0=np.random.default_rng(seed).normal(size=n),tol=1e-8)
        order=np.argsort(eigenvalues)[::-1]
        eigenvalues,eigenvectors=eigenvalues[order][1:],eigenvectors[:,order][:,1:]
        functions=eigenvectors/np.sqrt(degree[:,None])
        coordinates=functions*eigenvalues
        correlations=np.array([rho(coordinates[:,j],anchor) for j in range(coordinates.shape[1])])
        self.axis=int(np.argmax(np.abs(correlations)))
        self.orientation=1 if correlations[self.axis]>=0 else -1
        self.function=functions[:,self.axis]*self.orientation
        self.training_scores=coordinates[:,self.axis]*self.orientation
        self.diagnostics=dict(components=n_components,neighbors=k,selected_axis=self.axis+1,
                              eigenvalues=eigenvalues.tolist(),anchor_correlations=correlations.tolist())
        return self

    def transform(self,x):
        distances,indices=self.neighbors.kneighbors(x,n_neighbors=self.diagnostics['neighbors'])
        scales=np.maximum(distances[:,-1],1e-8)
        weights=np.exp(-distances**2/(scales[:,None]*self.scales[indices]))
        # Query density cancels when normalizing this row-stochastic kernel.
        kernel=weights/self.density[indices]
        probabilities=kernel/np.maximum(kernel.sum(1,keepdims=True),1e-15)
        return np.sum(probabilities*self.function[indices],axis=1)


class ImageOnlyOrderer:
    def __init__(self,method,seed=42):
        if method not in METHODS:raise ValueError(method)
        self.method,self.seed=method,seed

    def _dino(self,z,f,fit=False):
        z=normalize(np.asarray(z,dtype=np.float64))
        if fit:self.scaler=StandardScaler().fit(z)
        z=self.scaler.transform(z)
        if self.method=='dino_residual_diffusion':
            nuisance=f[:,NUISANCE_COLUMNS]
            if fit:
                self.nuisance_scaler=StandardScaler().fit(nuisance)
                self.polynomial=PolynomialFeatures(2,include_bias=False)
            nuisance=self.nuisance_scaler.transform(nuisance)
            design=self.polynomial.fit_transform(nuisance) if fit else self.polynomial.transform(nuisance)
            if fit:self.regression=Ridge(alpha=1.).fit(design,z)
            z=z-self.regression.predict(design)
        if fit:
            self.pca=PCA(n_components=min(32,z.shape[1],len(z)-1),random_state=self.seed).fit(z)
        return self.pca.transform(z)

    def fit(self,z,f):
        f=np.asarray(f);anchor=f[:,ANCHOR_COLUMN]
        self.diagnostics=dict(method=self.method,seed=self.seed,n_fit=len(f),uses_masks=False,uses_clinical_labels=False)
        if self.method=='mean':scores=f[:,0]
        elif self.method=='absolute_bright':scores=f[:,8]
        elif self.method=='local_anchor':scores=anchor
        elif self.method=='morphology_pc1':
            self.scaler=StandardScaler().fit(f[:,MORPHOLOGY_COLUMNS])
            self.pca=PCA(n_components=1,random_state=self.seed).fit(self.scaler.transform(f[:,MORPHOLOGY_COLUMNS]))
            scores=self.pca.transform(self.scaler.transform(f[:,MORPHOLOGY_COLUMNS]))[:,0]
            self.orientation=1 if rho(scores,anchor)>=0 else -1
            scores=scores*self.orientation
        else:
            projected=self._dino(z,f,fit=True)
            self.diagnostics['pca_variance_fraction']=float(self.pca.explained_variance_ratio_.sum())
            if self.method=='dino_pc1':
                self.orientation=1 if rho(projected[:,0],anchor)>=0 else -1
                scores=projected[:,0]*self.orientation
            elif self.method=='dino_kmeans':
                self.kmeans=KMeans(n_clusters=4,n_init=10,random_state=self.seed).fit(projected)
                means=[anchor[self.kmeans.labels_==j].mean() for j in range(4)]
                self.cluster_order=np.argsort(np.argsort(means))
                scores=self.cluster_order[self.kmeans.labels_].astype(float)
                self.diagnostics.update(cluster_anchor_means=list(map(float,means)),cluster_counts=np.bincount(self.kmeans.labels_).tolist())
            else:
                self.diffusion=DiffusionCoordinate().fit(projected,anchor,self.seed)
                scores=self.diffusion.training_scores
                self.diagnostics['diffusion']=self.diffusion.diagnostics
        self.training_scores=np.asarray(scores)
        self.sorted_scores=np.sort(scores)
        self.cutpoints=np.quantile(scores,[.25,.5,.75])
        self.diagnostics['anchor_rho_fit']=rho(scores,anchor)
        return self

    def raw_score(self,z,f):
        if self.method=='mean':return f[:,0]
        if self.method=='absolute_bright':return f[:,8]
        if self.method=='local_anchor':return f[:,ANCHOR_COLUMN]
        if self.method=='morphology_pc1':
            return self.pca.transform(self.scaler.transform(f[:,MORPHOLOGY_COLUMNS]))[:,0]*self.orientation
        projected=self._dino(z,f)
        if self.method=='dino_pc1':return projected[:,0]*self.orientation
        if self.method=='dino_kmeans':return self.cluster_order[self.kmeans.predict(projected)].astype(float)
        return self.diffusion.transform(projected)

    def predict(self,z,f):
        scores=self.raw_score(z,f)
        left=np.searchsorted(self.sorted_scores,scores,side='left')
        right=np.searchsorted(self.sorted_scores,scores,side='right')
        rank=(left+right)/(2*len(self.sorted_scores))
        levels=scores.astype(int) if self.method=='dino_kmeans' else np.searchsorted(self.cutpoints,scores,side='right')
        return rank,levels

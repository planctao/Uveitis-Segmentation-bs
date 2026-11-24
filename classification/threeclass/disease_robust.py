"""Within-parent Group-DRO-style disease reweighting, fitted patients only."""
import numpy as np
from experiment import sample_weights


class DiseaseRobustWeights:
    def __init__(self, frame, eta, support_shrink=0.):
        self.eta = float(eta); self.support_shrink = float(support_shrink)
        self.names = sorted(frame.disease.unique())
        self.targets = frame.disease.map({name:i for i,name in enumerate(self.names)}).to_numpy()
        mapping = frame.groupby('disease').label.agg(['first','nunique']); assert mapping['nunique'].eq(1).all()
        self.parents = np.array([int(mapping.loc[name,'first']) for name in self.names])
        self.image_weight = sample_weights(frame, frame.label).astype('float64')
        self.mass = np.bincount(self.targets, weights=self.image_weight, minlength=len(self.names))
        self.prior = self.mass.copy()
        for parent in range(3):
            selected = self.parents == parent; self.prior[selected] /= self.mass[selected].sum()
        self.support = frame.groupby('disease').patient_group.nunique().reindex(self.names).to_numpy()
        self.log_factor = np.zeros(len(self.names))

    def factors(self):
        unnormalized = np.exp(self.log_factor); result = unnormalized.copy()
        for parent in range(3):
            selected = self.parents == parent
            result[selected] /= (unnormalized[selected] * self.prior[selected]).sum()
        return result.astype('float32')

    def group_losses(self, probability, labels, smoothing=.03):
        logp = np.log(np.maximum(np.asarray(probability,dtype='float64'), 1e-7))
        loss = -(1-smoothing)*logp[np.arange(len(labels)),np.asarray(labels)] - smoothing*logp.mean(1)
        return np.bincount(self.targets,weights=loss*self.image_weight,minlength=len(self.names))/self.mass

    def update(self, losses):
        losses=np.asarray(losses,dtype='float64'); centered=losses.copy()
        for parent in range(3):
            selected=self.parents==parent;centered[selected]-=(losses[selected]*self.prior[selected]).sum()
        support=self.support/(self.support+self.support_shrink)
        self.log_factor=np.clip(self.log_factor+self.eta*support*centered,-np.log(2),np.log(2))
        return self.factors()

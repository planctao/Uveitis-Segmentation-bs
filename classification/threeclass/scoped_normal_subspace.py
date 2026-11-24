"""Apply the frozen normal-patient projection to selected DINO evidence streams."""
from experiment import KEYS
from normal_subspace import NormalSuppressedModel


SCOPES={'global_regions':['global','regions'],'global':['global'],'regions':['regions'],
        'mid_dispersion':['mid','dispersion']}


class ScopedNormalModel(NormalSuppressedModel):
    def transform(self,x):
        scope=SCOPES[self.cfg['normal_scope']]
        transformed=super().transform(x)
        return {key:(transformed[key] if key in scope else value) for key,value in x.items()}

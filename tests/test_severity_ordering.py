"""Numerical semantics and fit/holdout separation on known synthetic geometry."""
import numpy as np
from scipy.stats import spearmanr

from bs.severity_features import describe_image
from bs.severity_ordering import DiffusionCoordinate, ImageOnlyOrderer


def test_descriptor_handles_constant_and_black_images():
    for value in [0.,.5,1.]:
        f,fov=describe_image(np.full((256,256),value))
        assert f.shape==(20,) and np.isfinite(f).all()
        assert fov.any()


def test_fit_cutpoints_are_not_changed_by_holdout_extremes():
    rng=np.random.default_rng(1)
    f=rng.uniform(size=(80,20));z=rng.normal(size=(80,32))
    model=ImageOnlyOrderer('mean').fit(z,f)
    before=model.cutpoints.copy()
    query=np.full((5,20),100.)
    scores,levels=model.predict(np.zeros((5,32)),query)
    assert np.array_equal(model.cutpoints,before)
    assert np.all(scores==1) and np.all(levels==3)


def test_diffusion_nystrom_orders_unseen_points_on_a_curve():
    t=np.linspace(0,1,150)
    x=np.column_stack([t,.1*np.sin(t*3)])
    model=DiffusionCoordinate().fit(x,t)
    q=np.linspace(.03,.97,73)
    scores=model.transform(np.column_stack([q,.1*np.sin(q*3)]))
    assert spearmanr(scores,q).statistic>.99
    assert model.diagnostics['components']==1
    assert len(model.diagnostics['eigenvalues'])==6


def test_residual_model_predicts_finite_with_fixed_training_state():
    rng=np.random.default_rng(5)
    f=rng.uniform(size=(120,20));z=rng.normal(size=(120,48))
    model=ImageOnlyOrderer('dino_residual_diffusion').fit(z,f)
    coefficients=model.regression.coef_.copy()
    scores,levels=model.predict(z[:10],f[:10])
    assert np.isfinite(scores).all() and set(levels)<={0,1,2,3}
    assert np.array_equal(model.regression.coef_,coefficients)

import numpy as np
from bs.severity_synthetic import SyntheticDirectionOrderer, synthetic_views


def test_generated_views_match_mean_and_preserve_outside_fov():
    yy,xx=np.indices((256,256))
    fov=(xx-128)**2+(yy-128)**2<100**2
    image=np.zeros((256,256,3));image[fov]=.4
    views,diag=synthetic_views(image,'known-example')
    repeated,_=synthetic_views(image,'known-example')
    for a,b in zip(views,repeated):assert np.array_equal(a,b)
    assert abs(diag['meanmatch_error_before_quantization'])<1e-12
    assert abs(diag['meanmatch_error_after_quantization'])<1/255
    assert np.all(views[1]>=views[0])
    assert np.all(views[1][~fov]==0)


def test_counterfactual_direction_handles_known_independent_nuisance():
    rng=np.random.default_rng(3)
    z=rng.normal(size=(150,32));d=np.zeros(32);d[0]=1
    nuisance=np.zeros(32);nuisance[1]=1
    model=SyntheticDirectionOrderer('synthetic_direction').fit(z,z+.1*d,z+.2*d,z+.2*nuisance,z+.1*nuisance,z-.1*nuisance)
    raw=model.raw_score(z)
    assert np.mean(model.raw_score(z+.2*d)>raw)>.99
    assert model.diagnostics['synthetic_supervision']
    assert not model.diagnostics['manual_grade_labels_used']

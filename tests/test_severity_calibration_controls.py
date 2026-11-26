import numpy as np
from bs.severity_calibration_controls import CalibrationControlOrderer


def test_random_projection_matches_reference_rank_and_strength():
    rng=np.random.default_rng(4)
    z=rng.normal(size=(120,48));d=rng.normal(size=48)*.03;g=rng.normal(size=48)*.02
    m=CalibrationControlOrderer('meanmatched_random_norm_projection',42).fit(z,z+d,z+2*d,z+g,z+g,z-g)
    assert m.diagnostics['projection_rank']==8
    assert np.isclose(m.diagnostics['actual_retained_fraction'],m.diagnostics['reference_gamma_retained_fraction'],atol=1e-10)
    score,level=m.predict(z[:9]);assert np.isfinite(score).all()
    assert set(level)<={0,1,2,3}

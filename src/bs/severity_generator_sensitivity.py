"""Predeclared generator variants; canonical synthetic generator stays frozen."""
import hashlib
import numpy as np
from bs.severity_features import retinal_fov

GENERATOR_SPECS = {
    'baseline': (3, 8., 24., .08, .16),
    'narrow': (3, 4., 12., .08, .16),
    'broad': (3, 16., 48., .08, .16),
    'single': (1, 8., 24., .08, .16),
    'many': (6, 8., 24., .08, .16),
    'half_dose': (3, 8., 24., .04, .08),
    'double_dose': (3, 8., 24., .16, .32),
}


def sensitivity_views(rgb, uid, variant):
    count, sigma_min, sigma_max, dose_low, dose_high = GENERATOR_SPECS[variant]
    rgb = np.asarray(rgb, dtype=np.float64)
    fov = retinal_fov(rgb @ np.array([.299, .587, .114]))
    seed = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    centers = np.argwhere(fov)[rng.integers(0, fov.sum(), size=3)]
    sigmas = [rng.uniform(sigma_min, sigma_max) for _ in range(3)]
    if count > 3:
        extra_seed = int.from_bytes(hashlib.sha256((uid + ':extra').encode()).digest()[:8], 'little')
        extra = np.random.default_rng(extra_seed)
        centers = np.concatenate([centers, np.argwhere(fov)[extra.integers(0, fov.sum(), size=count-3)]])
        sigmas.extend(extra.uniform(sigma_min, sigma_max, size=count-3))
    yy, xx = np.indices(fov.shape)
    field = np.zeros(fov.shape)
    for (cy, cx), sigma in zip(centers[:count], sigmas[:count]):
        field += np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
    field = field / field.max() * fov
    low = rgb + dose_low * field[..., None] * (1-rgb)
    high = rgb + dose_high * field[..., None] * (1-rgb)
    c = (high-rgb)[fov].sum() / max((1-rgb)[fov].sum(), 1e-15)
    control = rgb + c * fov[..., None] * (1-rgb)
    views = [np.rint(np.clip(a, 0, 1)*255).astype(np.uint8) for a in [low, high, control]]
    diagnostics = dict(field_fov_mean=float(field[fov].mean()),
        meanmatch_error_before_quantization=float(high[fov].mean()-control[fov].mean()),
        meanmatch_error_after_quantization=float(views[1][fov].mean()/255-views[2][fov].mean()/255),
        uniform_control_c=float(c))
    return views, diagnostics

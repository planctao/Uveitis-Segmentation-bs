"""Fit-patient spectra and nonlinear neutral residuals on original features."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model, sample_weights


def fit_geometry(frame, normalized, rank=64):
    result = {}
    image_weights = sample_weights(frame, frame.label).astype('float64')
    image_weights /= image_weights.sum()
    for key in ['global', 'regions']:
        x = normalized[key].astype('float64')
        if x.ndim == 3:
            weights = np.repeat(image_weights / x.shape[1], x.shape[1]); x = x.reshape(-1, x.shape[-1])
        else: weights = image_weights
        center = np.sum(x * weights[:, None], 0)
        centered = x - center
        covariance = centered.T @ (centered * weights[:, None])
        eigenvalues, basis = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]; eigenvalues = np.maximum(eigenvalues[order], 0)
        basis = basis[:, order[:rank]]
        signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)]); basis *= np.where(signs == 0, 1, signs)
        result[key + '_center'] = center.astype('float32')
        result[key + '_basis'] = basis.astype('float32')
        result[key + '_eigenvalues'] = eigenvalues.astype('float32')
        result[key + '_ridge'] = np.array(.05 * np.trace(covariance) / len(center), dtype='float32')
    return result


class SpectralBranch(nn.Module):
    def __init__(self, dimension, output, bottleneck, alpha, rank=64):
        super().__init__(); self.alpha = alpha; self.rank = rank
        assert alpha in [0., .5]
        self.register_buffer('center', torch.zeros(dimension))
        self.register_buffer('basis', torch.zeros(dimension, rank))
        self.register_buffer('gain', torch.ones(rank))
        self.down = nn.Linear(rank, bottleneck)
        self.up = nn.Linear(bottleneck, output)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def fit_from(self, state, key):
        eigenvalues = state[key + '_eigenvalues'][:self.rank]
        gain = np.maximum(eigenvalues + float(state[key + '_ridge']), 1e-8) ** (-self.alpha / 2)
        self.center.copy_(torch.as_tensor(state[key + '_center']))
        self.basis.copy_(torch.as_tensor(state[key + '_basis']))
        self.gain.copy_(torch.as_tensor(gain))

    def forward(self, features):
        spectral = ((features - self.center) @ self.basis) * self.gain
        return .1 * self.up(F.gelu(self.down(spectral)))


class SpectralModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        assert cfg['spectral_scope'] in ['global', 'both']
        with torch.random.fork_rng(devices=[]):
            self.spectral_global = SpectralBranch(dims['global'], 256, 32, cfg['spectral_alpha'])
            if cfg['spectral_scope'] == 'both':
                self.spectral_regions = SpectralBranch(dims['regions'], 128, 16, cfg['spectral_alpha'])
        self.global_proj.register_forward_hook(self._global)
        if cfg['spectral_scope'] == 'both': self.region_proj.register_forward_hook(self._regions)

    def set_geometry(self, geometry):
        self.spectral_global.fit_from(geometry, 'global')
        if hasattr(self, 'spectral_regions'): self.spectral_regions.fit_from(geometry, 'regions')

    def _global(self, module, args, output):
        self._spectral_global_delta = self.spectral_global(args[0]); self._original_global_projection = output
        return output if self.cfg.get('spectral_disabled', False) else output + self._spectral_global_delta

    def _regions(self, module, args, output):
        self._spectral_region_delta = self.spectral_regions(args[0]); self._original_region_projection = output
        return output if self.cfg.get('spectral_disabled', False) else output + self._spectral_region_delta

    def forward(self, x):
        out = super().forward(x)
        out.update(spectral_global_delta=self._spectral_global_delta, original_global_projection=self._original_global_projection)
        if hasattr(self, 'spectral_regions'):
            out.update(spectral_region_delta=self._spectral_region_delta, original_region_projection=self._original_region_projection)
        return out

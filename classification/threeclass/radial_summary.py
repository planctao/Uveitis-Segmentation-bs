"""Foreground-weighted concentric summaries complement original M5 evidence."""
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from experiment import ROOT, KEYS, Model, load_data


def radial_bins():
    xy = np.stack(np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7), indexing='ij'), -1).reshape(49, 2)
    radius = np.linalg.norm(xy, axis=-1)
    return np.where(radius <= .5, 0, np.where(radius <= 1.000001, 1, 2))


def summarize_regions(regions, occupancy, mode):
    assert mode in ['global', 'radial', 'contrast']
    assert regions.ndim == 3 and regions.shape[1] == 49 and occupancy.shape == regions.shape[:2]
    assert np.isfinite(regions).all() and np.isfinite(occupancy).all() and occupancy.min() >= 0
    weight = occupancy.astype('float64'); values = regions.astype('float64')
    denominator = weight.sum(1, keepdims=True)
    total = (values * weight[:, :, None]).sum(1) / np.maximum(denominator, 1e-12)
    total = np.where(denominator > 1e-12, total, values.mean(1))
    if mode == 'global': return np.repeat(total[:, None], 3, axis=1).astype('float32')
    bins = radial_bins(); summaries = []
    for ring in range(3):
        selected = bins == ring; w = weight[:, selected]; v = values[:, selected]
        mass = w.sum(1, keepdims=True)
        summary = (v * w[:, :, None]).sum(1) / np.maximum(mass, 1e-12)
        summary = np.where(mass > 1e-12, summary, total)
        summaries.append(summary - total if mode == 'contrast' else summary)
    return np.stack(summaries, 1).astype('float32')


def load_radial(mode):
    frame, values, digest = load_data('multiscale_vit')
    protocol = json.loads((ROOT / 'runs/foreground_multiscale/protocol.json').read_text())
    assert protocol['manifest_sha256'] == digest
    with np.load(protocol['feature_storage']) as bank:
        indices = np.flatnonzero(frame.split.isin(['train', 'val']))
        assert np.array_equal(indices, np.arange(2091)) and np.array_equal(bank['indices'], indices)
        assert str(bank['manifest_sha256']) == digest and np.array_equal(bank['path'], frame.iloc[indices].path)
        summary = summarize_regions(bank['weighted_7'], bank['occupancy_7'], mode)
    values['spatial_summary'] = np.full((len(frame), *summary.shape[1:]), np.nan, dtype='float32')
    values['spatial_summary'][indices] = summary
    for v in values.values(): v[2091:] = np.nan
    assert all(np.isfinite(v[:2091]).all() and np.isnan(v[2091:]).all() for v in values.values())
    return frame, values, digest


class RadialSummary(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        assert rank in [16, 32]
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(3 * rank, 256)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, summaries):
        return .1 * self.up(F.gelu(self.down(self.norm(summaries))).flatten(1))


class RadialModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only'] and cfg['hidden'] == 256
        with torch.random.fork_rng(devices=[]): self.radial = RadialSummary(dims['spatial_summary'], cfg['radial_rank'])
        self.encoder[2].register_forward_pre_hook(self._add_summary)

    def _add_summary(self, module, args):
        self._before_radial = args[0]; self._radial_delta = self.radial(self._summary)
        return (args[0] if self.cfg.get('radial_disabled', False) else args[0] + self._radial_delta,)

    def forward(self, x):
        self._summary = x['spatial_summary']; out = super().forward(x)
        out.update(radial_delta=self._radial_delta, hidden_before_radial=self._before_radial)
        return out

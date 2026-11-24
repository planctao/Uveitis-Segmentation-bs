"""Spatial evidence from intermediate DINO blocks before final-region attention."""
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


class EarlyRegionAdapter(nn.Module):
    def __init__(self, input_dimension, rank, output_dimension=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dimension)
        self.down = nn.Linear(input_dimension, rank)
        self.up = nn.Linear(rank, output_dimension)
        self.gate_norm = nn.LayerNorm(output_dimension)
        self.gate = nn.Linear(output_dimension, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.)

    def forward(self, early, final_projection):
        gate = self.gate(self.gate_norm(final_projection)).sigmoid()
        delta = .1 * gate * self.up(F.gelu(self.down(self.norm(early))))
        return delta, gate


class EarlyRegionModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        with torch.random.fork_rng(devices=[]):
            self.early_adapter = EarlyRegionAdapter(dims['early_regions'], cfg['early_region_rank'])
        self.region_proj.register_forward_hook(self._supplement_regions)

    def _supplement_regions(self, module, args, original):
        self._original_region_projection = original
        self._early_delta, self._early_gate = self.early_adapter(self._early_input, original)
        if self.cfg.get('early_region_disabled', False):
            return original
        return original + self._early_delta

    def forward(self, x):
        self._early_input = x['early_regions']
        assert self._early_input.shape[:2] == x['regions'].shape[:2]
        out = super().forward(x)
        out.update(early_region_delta=self._early_delta, early_region_gate=self._early_gate,
                   original_region_projection=self._original_region_projection)
        return out

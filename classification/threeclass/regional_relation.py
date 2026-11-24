"""Content-dependent nonlocal regional messages with a neutral readout."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


class RegionalRelation(nn.Module):
    def __init__(self, rank=16, kind='content', position=False):
        super().__init__()
        assert rank in [16, 32] and kind in ['uniform', 'content']
        assert not position or kind == 'content'
        self.rank = rank; self.kind = kind
        self.norm = nn.LayerNorm(128)
        self.value = nn.Linear(128, rank)
        self.readout = nn.Linear(rank, 128, bias=False)
        nn.init.zeros_(self.readout.weight)
        if kind == 'content':
            self.query = nn.Linear(128, rank, bias=False)
            self.key = nn.Linear(128, rank, bias=False)
        if position:
            # One bias per relative displacement on the original 3x3 grid.
            self.relative_bias = nn.Parameter(torch.zeros(25))
            xy = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij'), -1).reshape(9, 2)
            relative = xy[:, None] - xy[None, :] + 2
            self.register_buffer('relative_index', relative[:, :, 0] * 5 + relative[:, :, 1])

    def forward(self, tokens, mask=None):
        n, count, _ = tokens.shape
        z = self.norm(tokens)
        values = F.gelu(self.value(z))
        scores = tokens.new_zeros((n, count, count))
        if self.kind == 'content':
            scores = self.query(z) @ self.key(z).transpose(1, 2) / math.sqrt(self.rank)
        if hasattr(self, 'relative_bias'):
            assert count == 9
            scores = scores + self.relative_bias[self.relative_index]
        valid = torch.ones((n, count), device=tokens.device, dtype=torch.bool) if mask is None else mask > .5
        valid = torch.where(valid.any(1, keepdim=True), valid, torch.ones_like(valid))
        identity = torch.eye(count, device=tokens.device, dtype=torch.bool)[None]
        allowed = valid[:, None, :] & ~identity
        allowed = torch.where(allowed.any(-1, keepdim=True), allowed, identity)
        attention = scores.masked_fill(~allowed, -1e4).softmax(-1)
        message = attention @ values - values
        delta = .1 * self.readout(message) * valid[:, :, None]
        return delta, attention


class RelationModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        with torch.random.fork_rng(devices=[]):
            self.relation = RegionalRelation(cfg['relation_rank'], cfg['relation_kind'], cfg['relation_position'])
        self.region_proj.register_forward_hook(self._relations)

    def _relations(self, module, args, output):
        self._original_projection = output
        self._relation_tokens = F.gelu(output)
        self._relation_delta, self._relation_attention = self.relation(self._relation_tokens, self._relation_mask)
        return output if self.cfg.get('relation_disabled', False) else output + self._relation_delta

    def forward(self, x):
        self._relation_mask = x.get('region_mask') if self.cfg.get('use_region_mask', True) else None
        out = super().forward(x)
        out.update(relation_input=self._relation_tokens, relation_delta=self._relation_delta,
                   relation_attention=self._relation_attention, original_region_projection=self._original_projection)
        return out

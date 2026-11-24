"""Balanced regional slots and matched softmax summaries on the original M5."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


def slot_weights(scores, mask=None, balanced=True, iterations=30):
    """Column-normalized slot weights; balanced mode also constrains region mass."""
    batch, count, slots = scores.shape
    valid = torch.ones((batch, count), device=scores.device, dtype=torch.bool) if mask is None else mask > .5
    valid = torch.where(valid.any(1, keepdim=True), valid, torch.ones_like(valid))
    log_kernel = scores.masked_fill(~valid[:, :, None], -torch.inf)
    if not balanced:
        return log_kernel.softmax(1), valid
    log_a = -valid.sum(1).to(scores.dtype).log()[:, None]
    log_b = -math.log(slots)
    log_v = torch.zeros((batch, 1, slots), dtype=scores.dtype, device=scores.device)
    for _ in range(iterations):
        # Invalid rows have zero mass, but their dual update stays finite.
        row_lse = torch.logsumexp(scores + log_v, -1)
        log_u = torch.where(valid, log_a - row_lse, torch.zeros_like(row_lse))
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, :, None], 1, keepdim=True)
    return (log_kernel + log_u[:, :, None] + log_v).exp() * slots, valid


class TransportSlots(nn.Module):
    def __init__(self, slots=2, rank=16, balanced=True):
        super().__init__()
        assert slots in [2, 4] and rank == 16
        self.slots = slots; self.rank = rank; self.balanced = balanced
        self.norm = nn.LayerNorm(128)
        self.key = nn.Linear(128, rank)
        self.value = nn.Linear(128, rank)
        self.queries = nn.Parameter(torch.randn(slots, rank) * .1)
        self.readout = nn.Linear(slots * rank, 256, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, tokens, mask=None):
        z = self.norm(tokens)
        raw_scores = self.key(z) @ self.queries.T / math.sqrt(self.rank)
        scores = 5 * torch.tanh(raw_scores / 5)
        weights, valid = slot_weights(scores, mask, self.balanced)
        values = F.gelu(self.value(z))
        summaries = torch.einsum('btk,btr->bkr', weights, values)
        delta = .1 * self.readout(summaries.flatten(1))
        return delta, weights, summaries, scores, valid


class TransportModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only'] and cfg['hidden'] == 256
        with torch.random.fork_rng(devices=[]):
            self.transport_slots = TransportSlots(cfg['transport_slots'], 16, cfg['transport_balanced'])
        self.region_proj.register_forward_hook(self._regions)
        self.encoder[1].register_forward_hook(self._hidden)

    def _regions(self, module, args, output):
        self._slot_tokens = F.gelu(output)

    def _hidden(self, module, args, output):
        self._slot_delta, self._slot_weights, self._slot_summaries, self._slot_scores, self._slot_valid = self.transport_slots(self._slot_tokens, self._slot_mask)
        self._slot_original_hidden = output
        return output if self.cfg.get('transport_disabled', False) else output + self._slot_delta

    def forward(self, x):
        self._slot_mask = x.get('region_mask') if self.cfg.get('use_region_mask', True) else None
        out = super().forward(x)
        out.update(slot_input=self._slot_tokens, slot_delta=self._slot_delta, slot_weights=self._slot_weights,
                   slot_summaries=self._slot_summaries, slot_scores=self._slot_scores,
                   slot_valid=self._slot_valid, slot_original_hidden=self._slot_original_hidden)
        return out

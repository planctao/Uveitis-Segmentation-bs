"""Image-dependent mixtures of small residual experts before disease logits."""
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


class PhenotypeExperts(nn.Module):
    def __init__(self, dimension, rank, experts, scale=.1):
        super().__init__()
        assert dimension > 0 and rank > 0 and experts > 1 and 0 < scale <= 1
        self.scale = float(scale)
        self.norm = nn.LayerNorm(dimension)
        self.route_norm = nn.LayerNorm(dimension)
        self.gate = nn.Linear(dimension, experts)
        self.down = nn.ModuleList([nn.Linear(dimension, rank) for _ in range(experts)])
        self.up = nn.ModuleList([nn.Linear(rank, dimension) for _ in range(experts)])
        for layer in self.up:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, embedding, routing_context):
        # Gate uses the deterministic, pre-dropout hidden context, while expert
        # values refine the same dropped embedding used by the M5 classifier.
        gates = self.gate(self.route_norm(routing_context)).softmax(-1)
        normalized = self.norm(embedding)
        values = torch.stack([up(F.gelu(down(normalized)))
                              for down, up in zip(self.down, self.up)], 1)
        delta = self.scale * (gates[:, :, None] * values).sum(1)
        return embedding + delta, gates, delta


class PhenotypeModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['mode'] == 'flat' and cfg['taxonomy_only']
        assert cfg['hidden'] == 256 and len(self.encoder) == 3
        self.encoder[2].register_forward_pre_hook(self._capture_routing_context)
        with torch.random.fork_rng(devices=[]):
            self.phenotype = PhenotypeExperts(256, cfg['phenotype_rank'], cfg['phenotype_experts'])

    def _capture_routing_context(self, module, args):
        self._routing_context = args[0]

    def forward(self, x):
        out = super().forward(x)
        before = out['embedding']
        refined, gates, delta = self.phenotype(before, self._routing_context)
        if not self.cfg.get('phenotype_disabled', False):
            logits = self.flat(refined)
            p = logits.softmax(-1) @ self.subtype_to_class
            out.update(p=p, flat=p.clamp(min=1e-7).log(), subtype=logits,
                       subtype_probability=p, embedding=refined)
            if self.cfg.get('fine_contrastive', 0.):
                out['fine_embedding'] = self.fine_projection(refined)
        out.update(phenotype_gates=gates, phenotype_delta=delta, embedding_before_experts=before)
        return out

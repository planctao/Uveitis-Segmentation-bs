"""Bounded conditional infection/noninfection correction preserving normal mass."""
import torch
from torch import nn
from register_patch_query import RegisterPatchQueryModel


def pair_correction(probability, score):
    shift = 2*score.double().tanh()
    infectious, noninfectious = probability[:, 0].double(), probability[:, 1].double()
    denominator = infectious*shift.exp()+noninfectious
    transfer = infectious*noninfectious*shift.expm1()/denominator.clamp_min(torch.finfo(torch.float64).tiny)
    corrected = torch.stack([(infectious+transfer).to(probability.dtype),
        (noninfectious-transfer).to(probability.dtype), probability[:, 2]], 1)
    return corrected, shift, transfer


class InflammatoryPairSpecialistModel(RegisterPatchQueryModel):
    def __init__(self, source, tail, config, seed):
        assert config['evidence'] in ['hidden', 'hidden_token'] and config['hidden'] in [0, 64]
        super().__init__(source, tail, {'kind': 'shared', 'rank': 4, 'alpha': 8}, seed)
        self.config = dict(config)
        del self.down, self.up
        width = 256 if config['evidence'] == 'hidden' else 288
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.specialist = nn.Sequential(nn.Linear(width, 1)) if not config['hidden'] else nn.Sequential(
                nn.Linear(width, config['hidden']), nn.GELU(), nn.Linear(config['hidden'], 1))
            nn.init.zeros_(self.specialist[-1].weight)
            nn.init.zeros_(self.specialist[-1].bias)
        self.eval()

    def forward(self, base, bank, raw_registers):
        side = ((raw_registers-self.source_mean_side_tokens)/self.source_std_side_tokens).clamp(-10, 10)
        out = self.head({**base, 'side_tokens': side})
        evidence = out['embedding'] if self.config['evidence'] == 'hidden' else torch.cat([out['embedding'], out['token_pooled']], -1)
        score = self.specialist(evidence).squeeze(1)
        original = out['p']
        probability, shift, transfer = pair_correction(original, score)
        out.update(p=probability, flat=probability.clamp(min=1e-7).log(), pair_base_probability=original,
            pair_score=score, pair_shift=shift, pair_transferred_mass=transfer)
        return out

"""Selected patch residuals with source attention mass or a fit-only constant control."""
import torch
from torch.nn import functional as F
from register_patch_gated_value import RegisterPatchGatedValueModel


class RegisterPatchMassCalibrationModel(RegisterPatchGatedValueModel):
    def __init__(self, source, tail, config, fit_mass_mean, seed):
        assert config['mass_mode'] in ['source_mass', 'constant_mass']
        super().__init__(source, tail, config, seed)
        mean = torch.as_tensor(fit_mass_mean, dtype=torch.float64).detach().clone()
        assert mean.shape == (2,) and torch.isfinite(mean).all()
        assert ((mean >= 0) & (mean <= 1)).all()
        self.register_buffer('fit_mass_mean', mean)

    def correct_registers(self, bank, raw_registers):
        query = bank['query_raw'].double()
        keys = bank['patch_keys'].double()
        values = bank['patch_values'].double()
        attention = ((query@keys.transpose(-2, -1))/8).softmax(-1)
        pooled = attention@values
        update = self.patch_update(values)
        conditional_difference = attention@update
        if self.config['mass_mode'] == 'source_mass':
            mass = bank['full_selected_mass'].double()
            assert mass.shape == query.shape[:-1]
        else:
            mass = self.fit_mass_mean[None, :, None, None].expand(query.shape[:-1])
        difference = conditional_difference*mass[..., None]
        flat = difference.permute(0, 1, 3, 2, 4).reshape(len(query), 2, 4, 1024).float()
        branch = F.linear(flat, self.tail.proj.weight, None)*self.tail.gamma1
        before = bank['postattention'].float()
        correction = self.tail.finish(before+branch)-self.tail.finish(before)
        registers = (raw_registers+torch.cat([correction[:, 0], correction[:, 1]], -1)).half().float()
        return registers, {'selected_attention': attention, 'baseline_pool': pooled,
            'conditional_value_pool_difference': conditional_difference, 'applied_attention_mass': mass,
            'value_pool_difference': difference, 'pool_difference': flat,
            'attention_difference': branch, 'tail_difference': correction}

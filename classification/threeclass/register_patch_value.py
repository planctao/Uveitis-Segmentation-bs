"""Anchored, trainable selected-patch value residuals with frozen DINO/M8 tail."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from register_patch_query import RegisterPatchQueryModel


class RegisterPatchValueModel(RegisterPatchQueryModel):
    def __init__(self, source, tail, config, seed):
        assert config['kind'] in ['shared', 'perhead'] and config['rank'] in [4, 8]
        super().__init__(source, tail, config, seed)
        del self.down, self.up
        heads = 1 if config['kind'] == 'shared' else 16
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.value_down = nn.Parameter(torch.empty(heads, config['rank'], 64))
            self.value_up = nn.Parameter(torch.zeros(heads, 64, config['rank']))
            for head in self.value_down:
                nn.init.kaiming_uniform_(head, a=math.sqrt(5))

    def correct_registers(self, bank, raw_registers):
        query = bank['query_raw'].double()
        keys = bank['patch_keys'].double()
        values = bank['patch_values'].double()
        attention = ((query@keys.transpose(-2, -1))/8).softmax(-1)
        pooled = attention@values
        down = self.value_down.double().expand(16, -1, -1)
        up = self.value_up.double().expand(16, -1, -1)
        code = torch.einsum('bshqd,hrd->bshqr', pooled, down)
        difference = torch.einsum('bshqr,hdr->bshqd', code, up)*(self.config['alpha']/self.config['rank'])
        flat = difference.permute(0, 1, 3, 2, 4).reshape(len(query), 2, 4, 1024).float()
        branch = F.linear(flat, self.tail.proj.weight, None)*self.tail.gamma1
        before = bank['postattention'].float()
        correction = self.tail.finish(before+branch)-self.tail.finish(before)
        corrected = (raw_registers+torch.cat([correction[:, 0], correction[:, 1]], -1)).half().float()
        return corrected, {'selected_attention': attention, 'baseline_pool': pooled,
            'value_pool_difference': difference, 'pool_difference': flat,
            'attention_difference': branch, 'tail_difference': correction}

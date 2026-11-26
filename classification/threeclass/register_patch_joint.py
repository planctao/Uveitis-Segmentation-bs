"""Joint selected-patch query/value residuals anchored to the frozen M8 source."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from register_patch_query import RegisterPatchQueryModel


class RegisterPatchJointModel(RegisterPatchQueryModel):
    def __init__(self, source, tail, config, seed):
        assert config['query_kind'] in ['shared', 'individual']
        assert config['value_kind'] in ['shared', 'perhead']
        query = {'kind': config['query_kind'], 'rank': config['rank'], 'alpha': config['alpha']}
        super().__init__(source, tail, query, seed)
        self.config = dict(config)
        heads = 1 if config['value_kind'] == 'shared' else 16
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1000)
            self.value_down = nn.Parameter(torch.empty(heads, config['rank'], 64))
            self.value_up = nn.Parameter(torch.zeros(heads, 64, config['rank']))
            for value in self.value_down:
                nn.init.kaiming_uniform_(value, a=math.sqrt(5))

    def correct_registers(self, bank, raw_registers):
        context = bank['query_input'].float()
        if self.config['query_kind'] == 'shared':
            context = context.mean(2, keepdim=True).expand_as(context)
        delta = self.up(self.down(context))*(self.config['alpha']/self.config['rank'])
        delta = delta.reshape(len(context), 2, 4, 16, 64).permute(0, 1, 3, 2, 4).double()
        query, keys, values = [bank[key].double() for key in ['query_raw', 'patch_keys', 'patch_values']]
        baseline = ((query@keys.transpose(-2, -1))/8).softmax(-1)
        modified = (((query+delta)@keys.transpose(-2, -1))/8).softmax(-1)
        modified_pool = modified@values
        query_difference = modified_pool-baseline@values
        code = torch.einsum('bshqd,hrd->bshqr', modified_pool, self.value_down.double().expand(16, -1, -1))
        value_difference = torch.einsum('bshqr,hdr->bshqd', code, self.value_up.double().expand(16, -1, -1))
        value_difference = value_difference*(self.config['alpha']/self.config['rank'])
        difference = (query_difference+value_difference).permute(0, 1, 3, 2, 4).reshape(len(context), 2, 4, 1024).float()
        branch = F.linear(difference, self.tail.proj.weight, None)*self.tail.gamma1
        before = bank['postattention'].float()
        correction = self.tail.finish(before+branch)-self.tail.finish(before)
        corrected = (raw_registers+torch.cat([correction[:, 0], correction[:, 1]], -1)).half().float()
        return corrected, {'delta_query': delta, 'selected_attention': modified, 'baseline_attention': baseline,
            'query_pool_difference': query_difference, 'value_pool_difference': value_difference,
            'pool_difference': difference, 'attention_difference': branch, 'tail_difference': correction}

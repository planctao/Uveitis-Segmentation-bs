"""Anchored disease readout refinement with a controlled local-patch pathway."""
import torch
from torch import nn
from register_patch_gated_value import RegisterPatchGatedValueModel


class AnchoredReadout(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert isinstance(original, nn.Linear)
        self.original = original
        for parameter in self.original.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.delta = nn.Linear(original.in_features, original.out_features)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def forward(self, hidden):
        return self.original(hidden)+self.delta(hidden)


class RegisterPatchReadoutAdaptationModel(RegisterPatchGatedValueModel):
    def __init__(self, source, tail, config, seed):
        super().__init__(source, tail, config, seed)
        assert source['cfg']['taxonomy_only'] and source['cfg']['mode'] == 'flat'
        self.head.flat = AnchoredReadout(self.head.flat)
        if not config['adapt_patches']:
            for parameter in [self.value_down, self.value_gate, self.value_up]:
                parameter.requires_grad_(False)
        self.eval()

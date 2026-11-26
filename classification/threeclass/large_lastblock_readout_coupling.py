"""Joint final-block and diagnostic-leaf readout adaptation with two declared controls."""
import numpy as np
import torch
from torch import nn
from experiment import KEYS
from large_lastblock_delta_lora import LastBlockDeltaLoRA

ARMS = {
    'attention_joint': {'kind': 'attention', 'rank': 4, 'alpha': 8},
    'token_joint': {'kind': 'token', 'rank': 8, 'alpha': 16},
    'readout_only': {'kind': 'attention', 'rank': 4, 'alpha': 8},
}
SEED_BASE = 20294600


class LastBlockReadoutCoupling(LastBlockDeltaLoRA):
    def __init__(self, source, templates, arm, seed):
        assert arm in ARMS
        super().__init__(source, templates, ARMS[arm], SEED_BASE + seed)
        self.arm = arm
        assert isinstance(self.head.flat, nn.Linear)
        assert source['cfg']['taxonomy_only'] and source['cfg']['mode'] == 'flat'
        assert self.head.flat.in_features == 256
        for parameter in self.head.flat.parameters():
            parameter.requires_grad_(True)
        if arm == 'readout_only':
            self.adapter_a.requires_grad_(False)
            self.adapter_b.requires_grad_(False)
        expected = {'head.flat.weight', 'head.flat.bias'}
        if arm != 'readout_only':
            expected |= {'adapter_a', 'adapter_b'}
        assert set(self.trainable_parameters()) == expected
        assert self.head.flat.out_features == len(source['diseases'])
        self.eval()

    def trainable_parameters(self):
        return {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}

    def forward(self, cache, features, ids, microbatch=8, use_checkpoint=True,
                independent=False, return_raw=False):
        if self.arm == 'readout_only':
            # The readout-only control has exactly the original cached features.
            raw = {key: torch.from_numpy(np.array(features[key][ids], copy=True)).cuda().float()
                   for key in list(KEYS) + ['side_tokens']}
        else:
            raw = self.anchored_features(cache, features, ids, microbatch, use_checkpoint, independent)
        normalized = {key: ((value-getattr(self, 'source_mean_'+key)) /
                            getattr(self, 'source_std_'+key)).clamp(-10, 10)
                      for key, value in raw.items()}
        out = self.head(normalized)
        if independent:
            # Audit the new trainable readout using a separately expressed affine
            # map and stable exponential marginalization. The inherited frozen
            # feature encoder is shared; this is not a second encoder oracle.
            logits = torch.einsum('bd,kd->bk', out['embedding'], self.head.flat.weight)
            logits = logits + self.head.flat.bias
            probability = torch.exp(logits-logits.max(1, keepdim=True).values)
            probability = probability/probability.sum(1, keepdim=True)
            coarse = probability @ self.head.subtype_to_class
            out.update(subtype=logits, p=coarse, flat=coarse.clamp(min=1e-7).log(),
                       subtype_probability=coarse)
        if return_raw:
            out['_anchored_raw'] = raw
        return out


def restore(checkpoint, source, templates, device='cuda'):
    model = LastBlockReadoutCoupling(source, templates, checkpoint['arm'], checkpoint['seed']).to(device)
    parameters = model.trainable_parameters()
    assert set(parameters) == set(checkpoint['trainable_state_dict'])
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval()

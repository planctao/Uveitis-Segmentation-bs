"""Epoch EMA without changing the source optimizer, buffers or random stream."""
import copy
import torch

ARMS = {'raw': None, 'ema0p8': .8, 'ema0p95': .95}
START_EPOCH = 5


class EpochAverages:
    def __init__(self, model):
        self.models = {name: copy.deepcopy(model).eval() for name in ARMS if name != 'raw'}
        self.count = 0

    @torch.no_grad()
    def update(self, model, epoch):
        if epoch < START_EPOCH:
            return
        source = dict(model.named_parameters())
        for name, averaged in self.models.items():
            for key, value in averaged.named_parameters():
                if self.count == 0:
                    value.copy_(source[key])
                else:
                    value.mul_(ARMS[name]).add_(source[key], alpha=1-ARMS[name])
            for key, value in averaged.named_buffers():
                assert torch.equal(value, dict(model.named_buffers())[key]), key
        self.count += 1


def flat_parameters(model):
    return torch.cat([value.detach().flatten() for value in model.parameters()])


@torch.no_grad()
def assign_flat(model, flat):
    offset = 0
    for value in model.parameters():
        value.copy_(flat[offset:offset+value.numel()].reshape_as(value))
        offset += value.numel()
    assert offset == len(flat)

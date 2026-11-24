"""Tune only selected pretrained DINO LayerNorm affine parameters."""
import torch
from torch import nn
from anchored_lora import AnchoredMultiScaleLoRA
from lora_seed import seed_all


class MultiScaleNorm(AnchoredMultiScaleLoRA):
    def __init__(self, checkpoint, config):
        super().__init__(checkpoint, {'rank': 0, 'blocks': 0, 'alpha': 0})
        self.norm_config = config
        assert config['norm_blocks'] in [0, 4]
        if config['norm_blocks']:
            for block in self.backbone.blocks[-config['norm_blocks']:]:
                for name in ['norm1', 'norm2']:
                    layer = getattr(block, name); assert isinstance(layer, nn.LayerNorm)
                    for p in layer.parameters(): p.requires_grad = True
        if config['final_norm']:
            assert isinstance(self.backbone.norm, nn.LayerNorm)
            for p in self.backbone.norm.parameters(): p.requires_grad = True
        if 'norm_state_dict' in checkpoint: self.load_delta(checkpoint['norm_state_dict'])


def construct_norm(checkpoint, config, seed):
    seed_all(seed); model = MultiScaleNorm(checkpoint, config); seed_all(seed)
    return model

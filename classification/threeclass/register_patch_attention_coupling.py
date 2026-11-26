"""Joint continuation of register evidence attention and selected-patch adaptation."""
import torch
from register_patch_query_selection import RegisterPatchQuerySelectionModel
from patch_adapter_cuda_ready import cuda_ready_restore
from kernel_residual_experiments import sha

PATCH_NAMES = ('value_down', 'value_gate', 'value_up')
ATTENTION_PREFIX = 'head.token_evidence.'


class RegisterPatchAttentionCouplingModel(RegisterPatchQuerySelectionModel):
    def __init__(self, source, tail, config, seed):
        assert isinstance(config['adapt_attention'], bool) and isinstance(config['adapt_patches'], bool)
        super().__init__(source, tail, config, seed)
        for name, parameter in self.head.token_evidence.named_parameters():
            parameter.requires_grad_(config['adapt_attention'] and name != 'key.bias')
        for name in PATCH_NAMES:
            getattr(self, name).requires_grad_(config['adapt_patches'])
        self.eval()

    def parameter_groups(self):
        attention = [p for name, p in self.named_parameters()
            if p.requires_grad and name.startswith(ATTENTION_PREFIX)]
        patches = [p for name, p in self.named_parameters()
            if p.requires_grad and name in PATCH_NAMES]
        groups = []
        if attention:
            groups.append({'params': attention, 'lr': .0001})
        if patches:
            groups.append({'params': patches, 'lr': .001})
        assert sum(len(g['params']) for g in groups) == sum(p.requires_grad for p in self.parameters())
        return groups


def reconstruction_state(model, differentiable=False):
    return {name: p.detach().clone().requires_grad_(differentiable and p.requires_grad)
        for name, p in model.named_parameters()
        if name in PATCH_NAMES or name.startswith(ATTENTION_PREFIX)}


@cuda_ready_restore
def restore(checkpoint, tail):
    source_path = checkpoint['source_checkpoint']
    assert sha(source_path) == checkpoint['source_checkpoint_sha256']
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    model = RegisterPatchAttentionCouplingModel(source, tail, checkpoint['adapter_config'],
        20292900+checkpoint['seed'])
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state_dict']) == expected
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(checkpoint['trainable_state_dict'][name])
    return model.eval(), source

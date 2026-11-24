"""Seeded low-rank adaptation anchored to fit-only frozen-head evidence."""
import numpy as np
import torch
import torch.nn.functional as F
from experiment import Model, normalize
from lora_model import MultiScaleLoRA
from lora_seed import seed_all


class AnchoredMultiScaleLoRA(MultiScaleLoRA):
    def forward(self, x320, x448):
        features = self.raw_features(x320, x448)
        normalized = {key: ((value - getattr(self, 'mean_' + key)) /
                      getattr(self, 'std_' + key)).clamp(-10, 10) for key, value in features.items()}
        out = self.head(normalized)
        out['_adapted_normalized_global'] = normalized['global']
        return out


def construct_anchored(checkpoint, config, seed):
    seed_all(seed)
    model = AnchoredMultiScaleLoRA(checkpoint, config)
    seed_all(seed)
    return model


@torch.inference_mode()
def fit_only_teacher(checkpoint, features, fit):
    indices = np.flatnonzero(fit)
    model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval().cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    probability = []
    global_features = []
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        values, _ = normalize({key: value[selected] for key, value in features.items()}, None, checkpoint['stats'])
        x = {key: torch.from_numpy(value).cuda() for key, value in values.items()}
        probability.append(model(x)['p'].cpu().numpy())
        global_features.append(values['global'])
    del model
    return np.concatenate(probability), np.concatenate(global_features)


def anchor_terms(out, teacher_probability, teacher_global, weights, temperature=2.):
    log_p = F.log_softmax(out['p'].float().clamp(min=1e-7).log() / temperature, dim=-1)
    target = F.softmax(teacher_probability.float().clamp(min=1e-7).log() / temperature, dim=-1)
    divergence = F.kl_div(log_p, target, reduction='none').sum(-1).clamp(min=0) * temperature ** 2
    feature_difference = (out['_adapted_normalized_global'].float() - teacher_global.float()).square().mean(-1)
    return (divergence * weights).mean(), (feature_difference * weights).mean()

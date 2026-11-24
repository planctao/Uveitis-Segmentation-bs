"""Training-only semantic decoders for the unchanged M5 classifier.

The teacher is frozen, self-supervised DINOv3-L: there is no fitted teacher
classifier and no teacher label, stopping-set or outer-set target in training.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from experiment import Model


class DistilledModel(Model):
    def __init__(self, cfg, dims, n_diseases):
        super().__init__(cfg, dims, n_diseases)
        assert cfg['arch'] == 'attention' and cfg['taxonomy_only']
        assert cfg['hidden'] == 256 and len(self.encoder) == 3
        self.encoder[2].register_forward_pre_hook(self._capture_hidden)
        mode = cfg['large_teacher_mode']
        assert mode in ['feature', 'relation']
        # Keep every common initialization and subsequent dropout/permutation RNG
        # identical to M5. The decoder itself contains no stochastic operations.
        with torch.random.fork_rng(devices=[]):
            self.semantic_decoder = (nn.Sequential(nn.Linear(256, 64), nn.GELU(),
                nn.Linear(64, 2048)) if mode == 'feature' else nn.Linear(256, 64))

    def _capture_hidden(self, module, args):
        self._semantic_hidden = args[0]

    def forward(self, x, with_aux=False):
        out = super().forward(x)
        if with_aux:
            out['teacher_prediction'] = self.semantic_decoder(self._semantic_hidden)
        return out

    def inference_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('semantic_decoder.')}


def prepare_teacher(values, fit, stats=None):
    """Fit channel statistics inside the fitting patients, never across splits."""
    values = np.asarray(values, dtype='float32')
    assert values.ndim == 2 and values.shape[1] == 2048
    if stats is None:
        assert fit is not None and len(fit)
        z = values[fit]
        stats = {'mean': z.mean(0).astype('float32'),
                 'std': np.maximum(z.std(0), .05).astype('float32')}
    z = np.clip((values - stats['mean']) / stats['std'], -10, 10)
    assert np.isfinite(z).all()
    z = z.reshape(len(z), 2, 1024)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    return z.astype('float32'), stats


def distillation_loss(prediction, teacher, patient_ids, weights, mode):
    """Patient-weighted cosine reconstruction or interpatient Gram matching."""
    teacher = teacher.detach()
    assert teacher.ndim == 3 and teacher.shape[1:] == (2, 1024)
    if mode == 'feature':
        z = F.normalize(prediction.reshape(-1, 2, 1024), dim=-1)
        per_image = (1 - (z * teacher).sum(-1)).mean(1)
        return (per_image * weights).mean()
    if mode != 'relation':
        raise ValueError('Unknown semantic distillation mode')
    z = F.normalize(prediction, dim=-1)
    student_gram = z @ z.T
    teacher_gram = torch.einsum('ivd,jvd->ij', teacher, teacher) / 2
    mask = patient_ids[:, None] != patient_ids[None, :]
    pair_weights = torch.sqrt(weights[:, None] * weights[None, :]) * mask
    return (((student_gram - teacher_gram).square() * pair_weights).sum()
            / pair_weights.sum().clamp(min=1e-8))

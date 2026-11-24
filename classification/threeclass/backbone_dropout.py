"""Train-only whole-backbone dropout against tied random-channel controls."""
import json
import numpy as np
import torch
from experiment import loss_fn
from normal_cross_geometry import NormalCrossGeometryModel


def draw_gains(count, kind, probability, rng):
    assert kind in ['channel', 'backbone'] and 0 <= probability <= 1
    if probability == 0:
        return np.ones((count, 3584), dtype='float32'), {}
    retain = 1-probability/2
    if kind == 'channel':
        kept = rng.random((count, 3584)) < retain
        return kept.astype('float32')/retain, {'kept': kept}
    chosen = rng.integers(0, 2, size=count)
    dropping = rng.random(count) < probability
    kept = np.ones((count, 2), dtype=bool)
    kept[np.flatnonzero(dropping), chosen[dropping]] = False
    gains = np.repeat(kept, [1536, 2048], axis=1).astype('float32')/retain
    return gains, {'chosen': chosen, 'dropping': dropping}


class BackboneDropoutModel(NormalCrossGeometryModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        assert cfg['cross_geometry_kind'] == 'independent' and cfg['cross_geometry_rank'] == 32
        self.mask_rng = np.random.default_rng(20284500+cfg.get('seed', 0))

    def transform(self, x):
        projected = super().transform(x)
        self._mask_proof = None
        if self.training and self.cfg['backbone_drop_probability']:
            before = json.dumps(self.mask_rng.bit_generator.state, sort_keys=True)
            gains, details = draw_gains(len(x['global']), self.cfg['backbone_drop_kind'],
                self.cfg['backbone_drop_probability'], self.mask_rng)
            out = {}
            for key, value in projected.items():
                gain = torch.from_numpy(gains).to(device=value.device, dtype=value.dtype)
                out[key] = value*(gain[:, None] if value.ndim == 3 else gain)
            if not self.witness_written:
                self._mask_proof = {'mask_gains': gains, 'mask_rng_before': np.array(before),
                    **{'mask_'+k: v for k, v in details.items()}}
            return out
        return projected

    def forward(self, x):
        out = super().forward(x)
        if self._mask_proof is not None:
            out['_backbone_mask_proof'] = self._mask_proof
        return out


def backbone_dropout_loss(out, y, disease, weights, cfg):
    loss = loss_fn(out, y, disease, weights, cfg)
    context = out.get('_large_normal_witness_context')
    if context is not None:
        model = context['model']
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)),
            'kind': np.array(cfg['backbone_drop_kind']), 'probability': np.array(cfg['backbone_drop_probability']),
            'loss': np.array(float(loss.detach())), 'rng_cpu': context['rng_cpu'].numpy(),
            **out['_backbone_mask_proof']}
        tensors = {'raw_global_all': context['raw_global_all'], 'projected_global': context['projected_global'],
            'projected_region_mean': context['projected_region_mean'], 'y': y, 'disease': disease,
            'probability_output': out['p'], 'subtype': out['subtype']}
        tensors.update({'weight_'+k: v for k, v in weights.items()})
        for k, v in tensors.items():
            proof[k] = v.detach().cpu().numpy()
        if context['rng_cuda'] is not None:
            proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for k, v in context['initial_parameter_sha256'].items():
            proof['initial_parameter_sha256_'+k] = np.array(v)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof)
        model.witness_written = True
    return loss

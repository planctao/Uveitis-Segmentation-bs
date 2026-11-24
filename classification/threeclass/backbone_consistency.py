"""Training-only complementary backbone views sharing the original attention MLP.

The full view supplies the supervised task and a detached coarse teacher. Two
complementary masked views are evaluated with the same dropout draws as that
full view, without advancing the source PyTorch random stream.
"""
import hashlib
import json
import numpy as np
import torch
from experiment import Model, loss_fn
from normal_cross_geometry import NormalCrossGeometryModel


def complementary_gains(count, kind, rng):
    if kind == 'backbone':
        first = np.zeros((count, 3584), dtype='float32'); first[:, :1536] = 2.
    elif kind == 'channel':
        first = 2*(rng.random((count, 3584)) < .5).astype('float32')
    else:
        raise ValueError(kind)
    return first, 2-first


class BackboneConsistencyModel(NormalCrossGeometryModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry, witness_path)
        self.view_rng = np.random.default_rng(20284700+cfg.get('seed', 0))

    def forward(self, x):
        active = self.training and self.cfg['backbone_consistency_weight'] > 0
        if active:
            cpu_rng = torch.random.get_rng_state().clone()
            device = x['global'].device
            cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
        out = super().forward(x)
        if active:
            projected = self.transform(x)
            before = json.dumps(self.view_rng.bit_generator.state, sort_keys=True)
            gains = complementary_gains(len(x['global']), self.cfg['backbone_consistency_kind'], self.view_rng)
            views = []
            after_cpu = torch.random.get_rng_state().clone()
            after_cuda = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
            for gain in gains:
                masked = {key: value*torch.from_numpy(gain[:, None] if value.ndim == 3 else gain).to(device)
                    for key, value in projected.items()}
                with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                    torch.random.set_rng_state(cpu_rng)
                    if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, device)
                    views.append(Model.forward(self, masked)['p'])
            assert torch.equal(after_cpu, torch.random.get_rng_state())
            if after_cuda is not None: assert torch.equal(after_cuda, torch.cuda.get_rng_state(device))
            out['complementary_probabilities'] = views
            if '_large_normal_witness_context' in out:
                context = out['_large_normal_witness_context']
                context.update(view_gains=gains[0], view_rng_before=before)
        return out


def backbone_consistency_loss(out, y, disease, weights, cfg):
    task = loss_fn(out, y, disease, weights, cfg)
    if cfg['backbone_consistency_weight'] == 0:
        return task
    teacher = out['p'].detach().clamp(min=1e-7)
    divergence = sum((teacher*(teacher.log()-view.clamp(min=1e-7).log())).sum(1)
        for view in out['complementary_probabilities'])/2
    regularizer = (divergence*weights['leaf']).mean()
    total = task+cfg['backbone_consistency_weight']*regularizer
    context = out.get('_large_normal_witness_context')
    if context is not None:
        model = context['model']
        gradients = torch.autograd.grad(total, tuple(model.parameters()), retain_graph=True)
        proof = {'seed': np.array(cfg['seed']), 'batch_size': np.array(len(y)),
            'kind': np.array(cfg['backbone_consistency_kind']), 'weight': np.array(cfg['backbone_consistency_weight']),
            'task_loss': np.array(float(task.detach())), 'consistency_loss': np.array(float(regularizer.detach())),
            'total_loss': np.array(float(total.detach())), 'rng_cpu': context['rng_cpu'].numpy(),
            'view_gains': context['view_gains'], 'view_rng_before': np.array(context['view_rng_before'])}
        tensors = {'raw_global_all': context['raw_global_all'], 'projected_global': context['projected_global'],
            'projected_region_mean': context['projected_region_mean'], 'y': y, 'disease': disease,
            'probability': out['p'], 'subtype': out['subtype'],
            'view_first_probability': out['complementary_probabilities'][0], 'view_second_probability': out['complementary_probabilities'][1],
            'parameter_gradient': torch.cat([g.flatten() for g in gradients])}
        tensors.update({'weight_'+k: v for k, v in weights.items()})
        for key, value in tensors.items(): proof[key] = value.detach().cpu().numpy()
        if context['rng_cuda'] is not None: proof['rng_cuda'] = context['rng_cuda'].cpu().numpy()
        for key, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_'+key] = np.array(value)
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    return total

"""Matched-dropout normal-coordinate adversarial training on a frozen DINO bank."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiment import KEYS, Model, loss_fn
from normal_subspace import NormalSuppressedModel


def rng_snapshot(device):
    return {'cpu': torch.random.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None}


def replay_forward(model, data, rng):
    device = data['global'].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(rng['cpu'])
        if device.type == 'cuda': torch.cuda.set_rng_state(rng['cuda'], device)
        return Model.forward(model, data)


class NormalTangentModel(NormalSuppressedModel):
    def __init__(self, cfg, dims, n_diseases, geometry=None, witness_path=None):
        super().__init__(cfg, dims, n_diseases, geometry)
        assert cfg['normal_rank'] == 32 and cfg['normal_strength'] == 1.
        assert cfg['normal_tangent_mode'] in ['rademacher', 'fgsm'] and 0 <= cfg['normal_tangent_epsilon'] <= .1
        self.tangent_rng = np.random.default_rng(20282124 + cfg.get('seed', 0))
        self.witness_path = None if witness_path is None else Path(witness_path)
        self.witness_written = False
        for key in KEYS:
            self.register_buffer('tangent_deviation_' + key, torch.zeros(32))
            if geometry is not None:
                getattr(self, 'tangent_deviation_' + key).copy_(torch.as_tensor(geometry[key + '_eigenvalues']).sqrt())

    def perturb(self, base, coordinates):
        result = {}
        for key, value in base.items():
            delta = coordinates[key] @ getattr(self, 'normal_basis_' + key).T
            if value.ndim == 3: delta = delta[:, None, :]
            result[key] = (value + delta).clamp(-10, 10)
        return result

    def forward(self, x):
        base = self.transform(x)
        if not self.training or not torch.is_grad_enabled(): return Model.forward(self, base)
        coordinates = {k: torch.zeros(len(x['global']), 32, device=v.device, dtype=v.dtype, requires_grad=True) for k, v in base.items()}
        rng = rng_snapshot(x['global'].device)
        hashes = None
        if self.witness_path is not None and not self.witness_written:
            hashes = {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in self.named_parameters()}
        out = Model.forward(self, self.perturb(base, coordinates))
        out['_tangent_context'] = {'model': self, 'base': base, 'coordinates': coordinates, 'rng': rng,
                                   'raw': x, 'initial_parameter_sha256': hashes}
        return out


def tangent_loss(out, y, disease, weights, cfg):
    context = out['_tangent_context']; model = context['model']; eps = cfg['normal_tangent_epsilon']
    clean = loss_fn(out, y, disease, weights, cfg)
    gradients = torch.autograd.grad(clean, [context['coordinates'][k] for k in KEYS], retain_graph=True, create_graph=False)
    random_signs = {k: model.tangent_rng.choice(np.array([-1., 1.], dtype='float32'), (len(y), 32)) for k in KEYS}
    coordinates = {}; directions = {}
    for key, gradient in zip(KEYS, gradients):
        direction = gradient.detach().sign() if cfg['normal_tangent_mode'] == 'fgsm' else torch.as_tensor(random_signs[key], device=gradient.device)
        directions[key] = direction
        coordinates[key] = (eps * getattr(model, 'tangent_deviation_' + key) * direction).detach()
        assert not coordinates[key].requires_grad and torch.isfinite(coordinates[key]).all()
    perturbed_inputs = model.perturb(context['base'], coordinates)
    before = rng_snapshot(y.device)
    perturbed = replay_forward(model, perturbed_inputs, context['rng'])
    after = rng_snapshot(y.device)
    assert torch.equal(before['cpu'], after['cpu'])
    if y.device.type == 'cuda': assert torch.equal(before['cuda'], after['cuda'])
    adversarial = loss_fn(perturbed, y, disease, weights, cfg)
    if model.witness_path is not None and not model.witness_written:
        proof = {'batch_size': np.array(len(y)), 'mode': np.array(cfg['normal_tangent_mode']), 'epsilon': np.array(eps),
                 'seed': np.array(cfg['seed']), 'y': y.detach().cpu().numpy(), 'disease': disease.detach().cpu().numpy(),
                 'clean_probability': out['p'].detach().cpu().numpy(), 'perturbed_probability': perturbed['p'].detach().cpu().numpy(),
                 'clean_loss': np.array(float(clean.detach())), 'perturbed_loss': np.array(float(adversarial.detach())),
                 'rng_cpu': context['rng']['cpu'].numpy(), 'raw_global_all': context['raw']['global'].detach().cpu().numpy()}
        if context['rng']['cuda'] is not None: proof['rng_cuda'] = context['rng']['cuda'].cpu().numpy()
        for name, value in weights.items(): proof['weight_' + name] = value.detach().cpu().numpy()
        for name, value in context['initial_parameter_sha256'].items(): proof['initial_parameter_sha256_' + name] = np.array(value)
        for key, gradient in zip(KEYS, gradients):
            proof['input_' + key] = context['raw'][key][:8].detach().cpu().numpy()
            proof['base_' + key] = context['base'][key][:8].detach().cpu().numpy()
            proof['output_' + key] = perturbed_inputs[key][:8].detach().cpu().numpy()
            proof['coordinate_gradient_' + key] = gradient.detach().cpu().numpy()
            proof['direction_' + key] = directions[key].cpu().numpy()
            proof['coordinate_' + key] = coordinates[key].cpu().numpy()
        model.witness_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model.witness_path, **proof); model.witness_written = True
    # Preserve explicit losses for complete-objective and gradient preflight checks.
    out['_tangent_clean_loss'] = clean; out['_tangent_perturbed_loss'] = adversarial
    out['_tangent_perturbed_output'] = perturbed; out['_tangent_coordinate_gradients'] = dict(zip(KEYS, gradients))
    out['_tangent_attack_coordinates'] = coordinates
    return .5 * clean + .5 * adversarial

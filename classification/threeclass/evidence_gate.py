"""A six-feature mixture gate learned only from patient-held-out predictions."""
import numpy as np
import torch
from scipy.special import expit
from experiment import sample_weights


def gate_features(base, evidence):
    p, q = [np.asarray(value, dtype='float64').clip(1e-12, 1) for value in [base, evidence]]
    assert p.shape == q.shape and p.ndim == 2 and p.shape[1] == 3
    p /= p.sum(1, keepdims=True);q /= q.sum(1, keepdims=True)
    center = .5 * (p + q)
    entropy = lambda value: -(value * np.log(value)).sum(1) / np.log(3)
    margin = lambda value: np.sort(value, axis=1)[:, -1] - np.sort(value, axis=1)[:, -2]
    js = .5 * ((p * np.log(p / center)).sum(1) + (q * np.log(q / center)).sum(1)) / np.log(2)
    return np.stack([entropy(p), entropy(q), margin(p), margin(q), js,
        (p.argmax(1) == q.argmax(1)).astype(float)], 1)


def predict_gate(model, base, evidence):
    x = (gate_features(base, evidence) - model['mean']) / model['std']
    cap = float(model['cap'])
    initial = np.log(.1 / (cap - .1))
    weight = cap * expit(initial + float(model['offset']) + x @ model['coefficients'])
    probability = (1 - weight[:, None]) * base + weight[:, None] * evidence
    return probability.astype('float32'), weight


def fit_gate(frame, base, evidence, cap, ridge):
    assert cap > .1 and cap <= 1 and ridge > 0
    features = gate_features(base, evidence)
    mean, std = features.mean(0), np.maximum(features.std(0), .05)
    x = torch.from_numpy((features - mean) / std)
    labels = torch.from_numpy(frame.label.to_numpy().astype('int64'))
    weights = torch.from_numpy(sample_weights(frame, frame.label).astype('float64'))
    p = torch.from_numpy(np.asarray(base, dtype='float64'))
    q = torch.from_numpy(np.asarray(evidence, dtype='float64'))
    coefficients = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64))
    offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.Adam([coefficients, offset], lr=.02)
    initial = float(np.log(.1 / (cap - .1)))
    best, selected = None, None
    history = []
    for step in range(401):
        weight = cap * torch.sigmoid(initial + offset + x @ coefficients)
        mix = (1 - weight[:, None]) * p + weight[:, None] * q
        nll = (-mix[torch.arange(len(labels)), labels].clamp(min=1e-12).log() * weights).mean()
        penalty = ridge * (coefficients.square().mean() + offset.square())
        loss = nll + penalty
        assert torch.isfinite(loss)
        history.append({'step': step, 'objective': float(loss.detach()), 'weighted_nll': float(nll.detach()),
            'penalty': float(penalty.detach()), 'mean_weight': float(weight.detach().mean())})
        if best is None or float(loss.detach()) < best:
            best = float(loss.detach())
            selected = {'mean': mean, 'std': std, 'cap': np.array(cap), 'ridge': np.array(ridge),
                'coefficients': coefficients.detach().numpy().copy(), 'offset': offset.detach().numpy().copy(),
                'selected_step': np.array(step), 'selected_objective': np.array(best)}
        if step == 400:break
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return selected, history

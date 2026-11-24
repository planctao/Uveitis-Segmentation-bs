"""Check mixture bounds, label-free inference, exact restoring and objective fitting."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT, dump, sample_weights
from evidence_gate import gate_features, fit_gate, predict_gate


def main():
    rng = np.random.default_rng(20261124)
    labels = np.tile([0, 1, 2], 15)
    frame = pd.DataFrame({'patient_group': ['p' + str(i // 3) for i in range(45)], 'label': labels})
    p = rng.dirichlet([2, 2, 2], size=45)
    q = .4 * p + .6 * np.eye(3)[labels]
    features = gate_features(p, q)
    assert features.shape == (45, 6) and np.isfinite(features).all()
    permutation = [2, 0, 1]
    assert np.allclose(features, gate_features(p[:, permutation], q[:, permutation]), atol=1e-12)
    for cap in [.25, .5]:
        initial = {'mean': features.mean(0), 'std': np.maximum(features.std(0), .05),
            'cap': np.array(cap), 'coefficients': np.zeros(6), 'offset': np.array(0.)}
        base_mix, weight = predict_gate(initial, p, q)
        assert np.allclose(weight, .1, atol=1e-15)
        assert np.allclose(base_mix, .9 * p + .1 * q, atol=1e-7)
        fitted, history = fit_gate(frame, p, q, cap, .1)
        actual, w = predict_gate(fitted, p, q)
        assert history[-1]['objective'] < history[0]['objective']
        assert (w >= 0).all() and (w <= cap).all()
        assert np.isfinite(actual).all() and np.allclose(actual.sum(1), 1, atol=1e-7)
        assert np.allclose(predict_gate(fitted, p[:, permutation], q[:, permutation])[0], actual[:, permutation], atol=1e-7)
        equal, _ = predict_gate(fitted, p, p)
        assert np.allclose(equal, p, atol=1e-7)
        repeated, _ = fit_gate(frame, p, q, cap, .1)
        assert np.array_equal(predict_gate(repeated, p, q)[0], actual)
        selected = history[int(fitted['selected_step'])]
        penalty = .1 * (np.square(fitted['coefficients']).mean() + float(fitted['offset']) ** 2)
        objective = (-np.log(actual[np.arange(len(labels)), labels]) * sample_weights(frame, labels)).mean() + penalty
        assert abs(objective - float(fitted['selected_objective'])) < 1e-6
    extremes = np.eye(3)
    assert np.isfinite(gate_features(extremes, extremes[::-1])).all()
    dump(ROOT / 'verification_evidence_gate.json', {'passed': True,
        'checks': ['Six features finite and invariant to class-name permutation',
            'Neutral coefficients yield exactly10%initial Gaussian weight',
            'All fitted weights in the predeclared cap, valid probability simplex preserved',
            'Identical experts reproduce their common probabilities',
            'Inference accepts no labels or patient identities',
            'Deterministic full-batch optimization selects only the penalized training objective',
            'Independent weighted NLL and penalty reconstruct the selected objective',
            'Degenerate and confident probabilities yield finite uncertainty features']})
    print('Evidence gate probability and fitting checks passed', flush=True)


if __name__ == '__main__':main()

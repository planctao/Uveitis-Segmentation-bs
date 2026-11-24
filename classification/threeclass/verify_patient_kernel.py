"""Check kernel residual geometry, patient weighting and an independent solver."""
import numpy as np
import pandas as pd
from experiment import ROOT, dump
from patient_kernel import patient_support, fit_kernel, predict_kernel, kernel_values, simplex_projection


def main():
    rng = np.random.default_rng(20261124)
    labels = np.repeat([0, 1, 2], 12)
    frame = pd.DataFrame({'patient_group': ['p' + str(i // 3) for i in range(36)], 'label': labels})
    z = rng.normal(size=(36, 9))
    p = rng.dirichlet([2., 2., 2.], size=36)
    q = rng.normal(size=(11, 9));base = rng.dirichlet([2., 2., 2.], size=11)
    support = patient_support(frame, z, p)
    assert all(np.isclose(support['weights'][support['labels'] == i].sum(), 1/3) for i in range(3))
    fitted = fit_kernel(support, .5, .01)
    K = kernel_values(support['centers'], support, .5)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    target = np.eye(3)[support['labels']]
    system = K + .01 * np.diag(1 / support['weights'])
    assert np.allclose(fitted['direct'], np.linalg.solve(system, target), atol=1e-12)
    assert np.allclose(fitted['residual'], np.linalg.solve(system, target - support['source_probability']), atol=1e-12)
    permutation = rng.permutation(len(frame))
    permuted = patient_support(frame.iloc[permutation], z[permutation], p[permutation])
    duplicates = np.r_[np.arange(len(frame)), [0, 1, 2, 0, 1, 2]]
    repeated = patient_support(frame.iloc[duplicates], z[duplicates], p[duplicates])
    rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    transformed = patient_support(frame, (z @ rotation) * 3 + 7, p)
    for mode in ['direct', 'residual']:
        actual = predict_kernel(support, fitted, q, base, mode, .25)
        assert np.isfinite(actual).all() and (actual >= 0).all() and np.allclose(actual.sum(1), 1)
        assert np.array_equal(predict_kernel(support, fitted, q, base, mode, 0), base)
        assert np.allclose(actual, predict_kernel(permuted, fit_kernel(permuted, .5, .01), q, base, mode, .25), atol=1e-7)
        assert np.allclose(actual, predict_kernel(repeated, fit_kernel(repeated, .5, .01), q, base, mode, .25), atol=1e-7)
        assert np.allclose(actual, predict_kernel(transformed, fit_kernel(transformed, .5, .01), (q @ rotation) * 3 + 7, base, mode, .25), atol=1e-7)
    perfect = patient_support(frame, z, np.eye(3)[labels])
    perfect_fit = fit_kernel(perfect, .5, .01)
    assert not perfect_fit['residual'].any()
    assert np.allclose(predict_kernel(perfect, perfect_fit, q, base, 'residual', .25), base, atol=1e-7)
    assert np.allclose(fitted['residual'].sum(1), 0, atol=1e-12)
    weak = fit_kernel(support, .5, 1e12)
    assert np.abs(weak['residual']).max() < 1e-10
    extreme = np.array([[3., -7., 2.], [0., 0., 0.], [.2, .3, .5]])
    assert np.allclose(simplex_projection(extreme), [[1., 0., 0.], [1/3]*3, [.2, .3, .5]])
    # Several images from a patient with different coarse labels get distinct support rows.
    mixed = frame.copy();mixed.loc[3, 'patient_group'] = 'p0'
    mixed.loc[3, 'label'] = 1
    mixed_support = patient_support(mixed, z, p)
    assert set(mixed_support['labels'][mixed_support['patients'] == 'p0']) == {0, 1}
    dump(ROOT / 'verification_patient_kernel.json', {'passed': True,
        'checks': ['Class support weights give exactly one third mass to each coarse class',
            'Symmetric weighted solution equals the independent nonsymmetric dual solution',
            'RBF fitting Gram matrix positive semidefinite',
            'Row order and whole-patient image replication do not affect predictions',
            'Translations, rotations and overall embedding scaling preserve predictions',
            'Zero correction weight exactly restores source probabilities',
            'Perfect-source targets have exactly zero residual and large ridge suppresses correction',
            'Signed residuals conserve total mass before a valid Euclidean simplex projection',
            'Mixed-label patients retain separate support rows for their observed classes']})
    print('Patient kernel invariants and independent solution passed', flush=True)


if __name__ == '__main__':
    main()

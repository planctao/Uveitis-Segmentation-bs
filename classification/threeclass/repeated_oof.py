"""Paired patient resampling of all jointly repeated outer predictions."""
import numpy as np


def paired_repeated(root, reference, new, frame, repeats):
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    tf = frame.iloc[original]
    _, group = np.unique(tf.split_group, return_inverse=True)
    count = group.max() + 1

    def confusion(arm):
        result = np.zeros((count, 3, 3), dtype=np.int64)
        for repeat in repeats:
            saved = [np.load(root / arm / f'repeat_{repeat}/fold_{fold}/outer_predictions.npz') for fold in range(3)]
            indices = np.concatenate([s['indices'] for s in saved])
            p = np.concatenate([s['p'] for s in saved])
            order = indices.argsort()
            assert np.array_equal(indices[order], original)
            result += np.bincount(group * 9 + tf.label.to_numpy() * 3 + p[order].argmax(1),
                                  minlength=count * 9).reshape(count, 3, 3)
        return result

    def score(c):
        diagonal = np.diagonal(c, axis1=-2, axis2=-1)
        denominator = c.sum(-1) + c.sum(-2)
        f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                       where=denominator > 0).mean(-1)
        return diagonal.sum(-1) / c.sum((-2, -1)), f1

    a, b = confusion(reference), confusion(new)
    samples = np.random.default_rng(20270124).integers(0, count, size=(2000, count))
    aa, af = score(a[samples].sum(1))
    ba, bf = score(b[samples].sum(1))
    pa, pf = score(a.sum(0))
    qa, qf = score(b.sum(0))
    return {'accuracy_difference': float(qa - pa), 'macro_f1_difference': float(qf - pf),
        'accuracy_ci95': np.quantile(ba - aa, [.025, .975]).tolist(),
        'macro_f1_ci95': np.quantile(bf - af, [.025, .975]).tolist(),
        'bootstrap_unit': 'Patient split group; all images and repeated predictions jointly retained.',
        'point_metric': 'Pooled repeated predictions; distinct from averaging per-fold macro-F1.'}

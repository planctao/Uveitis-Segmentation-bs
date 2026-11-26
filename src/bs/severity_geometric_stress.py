"""Exact square-grid symmetries used by a frozen severity stress test."""
import numpy as np

DIHEDRAL_CONDITIONS = (
    'identity', 'rot90', 'rot180', 'rot270',
    'reflect_lr', 'reflect_lr_rot90', 'reflect_lr_rot180', 'reflect_lr_rot270',
)


def square_symmetry(array, condition):
    """Permute the first two axes; keep all pixels, channels and dtype."""
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[0] != array.shape[1]:
        raise ValueError('A square spatial grid is required')
    if condition not in DIHEDRAL_CONDITIONS:
        raise ValueError(condition)
    index = DIHEDRAL_CONDITIONS.index(condition)
    result = np.fliplr(array) if index >= 4 else array
    return np.ascontiguousarray(np.rot90(result, index % 4, axes=(0, 1)))

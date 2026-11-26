import numpy as np
import pytest

from bs.severity_geometric_stress import DIHEDRAL_CONDITIONS, square_symmetry


def test_symmetry_orbit_preserves_pixels_and_channel_burdens():
    image = np.arange(7 * 7 * 3, dtype=np.uint8).reshape(7, 7, 3)
    mask = np.arange(49).reshape(7, 7) % 4
    distinct = set()
    for condition in DIHEDRAL_CONDITIONS:
        changed = square_symmetry(image, condition)
        transformed_mask = square_symmetry(mask, condition)
        assert changed.shape == image.shape and changed.dtype == image.dtype
        assert changed.flags.c_contiguous
        assert np.array_equal(np.sort(changed.reshape(-1, 3), axis=0),
                              np.sort(image.reshape(-1, 3), axis=0))
        assert np.array_equal(np.bincount(transformed_mask.ravel()), np.bincount(mask.ravel()))
        for labels in [(1, 3), (2, 3), (1, 2, 3)]:
            assert np.isin(transformed_mask, labels).mean() == np.isin(mask, labels).mean()
        distinct.add(changed.tobytes())
    assert len(distinct) == 8


def test_group_closure_and_inverses():
    image = np.arange(25).reshape(5, 5)
    orbit = {square_symmetry(image, c).tobytes() for c in DIHEDRAL_CONDITIONS}
    for first in DIHEDRAL_CONDITIONS:
        changed = square_symmetry(image, first)
        assert any(np.array_equal(square_symmetry(changed, c), image) for c in DIHEDRAL_CONDITIONS)
        for second in DIHEDRAL_CONDITIONS:
            assert square_symmetry(changed, second).tobytes() in orbit
    assert np.array_equal(square_symmetry(square_symmetry(image, 'rot90'), 'rot270'), image)
    assert np.array_equal(square_symmetry(square_symmetry(image, 'reflect_lr'), 'reflect_lr'), image)


def test_rejects_non_square_and_unknown_transform():
    with pytest.raises(ValueError):
        square_symmetry(np.zeros((3, 4, 3)), 'rot90')
    with pytest.raises(ValueError):
        square_symmetry(np.zeros((4, 4, 3)), 'crop')

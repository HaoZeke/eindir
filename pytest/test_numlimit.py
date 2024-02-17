import numpy as np

import pytest
from eindir.core.components import NumLimit, OutOfBounds


def test_numlimit_check():
    limits = NumLimit(low=np.array([0, 0]), high=np.array([5, 5]))
    pos_within = np.array([2, 2])
    pos_outside = np.array([6, 6])

    limits.check(pos_within)  # Should pass without exception

    with pytest.raises(OutOfBounds):
        limits.check(pos_outside)


def test_numlimit_mkpoint():
    limits = NumLimit(low=np.array([1, 1]), high=np.array([2, 2]))
    point = limits.mkpoint()
    assert all(point >= limits.low) and all(
        point <= limits.high
    ), "Generated point is out of bounds."


def test_numlimit_clip():
    limits = NumLimit(low=np.array([0, 0]), high=np.array([5, 5]))
    point = np.array([-1, 6])
    clipped_point = limits.clip(point)
    assert np.array_equal(
        clipped_point, np.array([0, 5])
    ), "Clip does not work as expected."

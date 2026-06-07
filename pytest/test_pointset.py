import numpy as np
import pytest

import eindir


def test_low_discrepancy_points_are_numpy_bounded_and_deterministic():
    low = np.array([-1.0, -2.0])
    high = np.array([1.0, 2.0])

    first = eindir.low_discrepancy_points(low, high, 8)
    second = eindir.low_discrepancy_points(low, high, 8)

    assert first.shape == (8, 2)
    assert np.all(first >= low)
    assert np.all(first <= high)
    assert np.allclose(first, second)


def test_low_discrepancy_points_reject_bad_bounds():
    with pytest.raises(ValueError, match="same length"):
        eindir.low_discrepancy_points([0.0], [1.0, 2.0], 4)

    with pytest.raises(ValueError, match="upper bound"):
        eindir.low_discrepancy_points([2.0], [1.0], 4)

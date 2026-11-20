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
    with pytest.raises(ValueError, match="different shapes"):
        eindir.low_discrepancy_points([0.0], [1.0, 2.0], 4)

    with pytest.raises(ValueError, match="must not exceed high"):
        eindir.low_discrepancy_points([2.0], [1.0], 4)


def test_point_sets_reject_an_empty_box():
    with pytest.raises(ValueError, match="at least one dimension"):
        eindir.low_discrepancy_points([], [], 2)
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="at least one dimension"):
        eindir._core.shifted_low_discrepancy_points(empty, empty, 2)


def test_point_sets_reject_a_coordinate_that_is_not_a_number():
    core = eindir._core
    nan_low = np.array([np.nan, 0.0])
    inverted_high = np.array([1.0, -1.0])
    for fn in (
        core.low_discrepancy_points,
        core.shifted_low_discrepancy_points,
    ):
        with pytest.raises(ValueError, match="numbers"):
            fn(np.array([np.nan]), np.array([1.0]), 2)
        with pytest.raises(ValueError, match="numbers"):
            fn(np.array([0.0]), np.array([np.nan]), 2)
        # NaN is reported before the inverted pair at the next index.
        with pytest.raises(ValueError, match="numbers"):
            fn(nan_low, inverted_high, 2)

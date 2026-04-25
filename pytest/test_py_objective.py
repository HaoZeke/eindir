"""Round-trip tests for the eindir._core PyObjective / Bounds / FPair classes."""

import numpy as np
import eindir
from eindir import _core as core


def test_bounds_construction_and_membership():
    b = core.Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-6)
    assert b.dims == 2
    assert b.contains(np.array([0.0, 0.0]))
    assert b.contains(np.array([0.5, -0.5]))
    assert not b.contains(np.array([2.0, 0.0]))


def test_bounds_clip():
    b = core.Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.0)
    out = b.clip(np.array([2.0, -3.0]))
    assert np.allclose(out, np.array([1.0, -1.0]))


def test_fpair_round_trip():
    pos = np.array([0.5, -0.5])
    fp = core.FPair(pos, 3.14)
    assert np.allclose(fp.pos, pos)
    assert fp.val == 3.14


def test_py_objective_calls_python_callable():
    bounds = core.Bounds(np.array([-5.0, -5.0]), np.array([5.0, 5.0]), 1e-9)
    obj = core.PyObjective(lambda x: float(np.sum(x ** 2)), bounds)
    assert obj.dim == 2
    assert abs(obj.eval(np.array([0.0, 0.0])) - 0.0) < 1e-9
    assert abs(obj.eval(np.array([1.0, 2.0])) - 5.0) < 1e-9
    assert abs(obj.eval(np.array([3.0, 4.0])) - 25.0) < 1e-9


def test_py_objective_bounds_round_trip():
    bounds = core.Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-3)
    obj = core.PyObjective(lambda x: 0.0, bounds)
    b = obj.bounds
    assert b.dims == 2
    assert abs(b.slack - 1e-3) < 1e-9

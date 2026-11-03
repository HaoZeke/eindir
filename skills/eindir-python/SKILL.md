---
name: eindir-python
description: Use the eindir Python package to build box bounds, wrap a Python function and optionally its gradient as an objective, and draw bounded Halton (low-discrepancy) start points, through eindir and its Rust extension eindir._core. Says which names exist in Python and which are Rust-only (the generalized Langevin equation (GLE) thermostat, the surrogates, the gradient adapters, the builtin objective types), and which argument types and exceptions trip callers up. Use when writing Python against eindir, when an example from the eindir manual fails with ImportError, or when an eindir call raises TypeError or PanicException.
license: See LICENSE in the eindir repository
compatibility: Python 3.10 or newer with eindir and NumPy, from PyPI (`pip install eindir`) or from a checkout with `maturin develop --features python`, which needs Rust 1.88 or newer.
---

# eindir from Python

## What exists where

The Python surface is two modules:

| Module | Names |
|---|---|
| `eindir` | `NumLimit`, `FPair`, `ObjectiveFunction`, `OutOfBounds`, `low_discrepancy_points`, `styblinski_tang`, and array API, DLPack and TVM-FFI helpers |
| `eindir._core` (Rust) | `Bounds`, `PyObjective`, `FPair`, `low_discrepancy_points`, `shifted_low_discrepancy_points`, `__version__` |

The rest of the crate has no Python binding: `GleThermostat`, `optimal_sampling_drift`, `matrix_exp`, `ldl_sqrt`, `halton_unit`, `AdditiveSurrogate`, `ChebyshevSurrogate`, `ReducedObjective`, the `Objective` trait, `FiniteDiffGradient`, `AnalyticGradient` and the builtin objective types (`Rastrigin`, `Rosenbrock`, `Ackley`, `StybTang2D`).
The manual's Quickstart and its Python quickstart, low-discrepancy, GLE and surrogate tutorials import some of these from `eindir`, so those examples stop with ImportError.
For the GLE thermostat and the surrogates, write Rust (the `eindir-rust` skill).

`scripts/surface.py` checks this table against the installed package.
The version string is `eindir._core.__version__`, and `eindir.__version__` does not exist.

## An objective with bounds and a gradient

`_core.Bounds(low, high, slack=1e-6)` is a box.
`_core.PyObjective(fn, bounds, grad_fn=None)` wraps a Python callable so Rust code can call it.
Both take float64 NumPy arrays.

```python
import numpy as np
from eindir import _core

low = np.array([-5.12, -5.12])
high = np.array([5.12, 5.12])
bounds = _core.Bounds(low, high)


def sphere(x):
    return float(np.sum((x - 1.0) ** 2))


def sphere_grad(x):
    return 2.0 * (x - 1.0)


obj = _core.PyObjective(sphere, bounds, grad_fn=sphere_grad)
x = np.array([0.0, 3.0])
assert obj.dim == 2 and obj.eval(x) == 5.0
assert np.array_equal(obj.grad(x), [-2.0, 4.0])
assert bounds.dims == 2 and bounds.contains(x)
assert np.array_equal(bounds.clip(np.array([9.0, -9.0])), [5.12, -5.12])
print(obj.eval(x), obj.grad(x), bounds.slack)
```

`Bounds` exposes `dims`, `slack`, `contains` and `clip`, but not the corners: keep your own `low` and `high` arrays.

## Start points

`eindir.low_discrepancy_points(low, high, n, skip=1)` returns an `(n, d)` float64 array of Halton points scaled to the box.
The same arguments give the same points.
`_core.shifted_low_discrepancy_points(low, high, n, skip=1, seed=0)` applies a seeded random shift, so different seeds give independent replicas of the design.
Like every `_core` function, it wants float64 arrays and returns a list of lists.

```python
import numpy as np
import eindir
from eindir import _core

low = np.array([-5.12, -5.12])
high = np.array([5.12, 5.12])
obj = _core.PyObjective(lambda x: float(np.sum((x - 1.0) ** 2)), _core.Bounds(low, high))

starts = eindir.low_discrepancy_points(low, high, 8)
assert starts.shape == (8, 2)
values = np.array([obj.eval(row) for row in starts])
best = int(np.argmin(values))
print("best start", starts[best], "value", values[best])

replica = np.asarray(_core.shifted_low_discrepancy_points(low, high, 8, 1, 3))
assert replica.shape == (8, 2) and not np.allclose(replica, starts)
assert np.all((replica >= low) & (replica <= high))
```

Rows of the array that `eindir.low_discrepancy_points` returns are contiguous, so they can go straight to `obj.eval`.

## The pure-Python classes

`NumLimit`, `FPair` and `ObjectiveFunction` in `eindir` are plain Python and separate from `_core.Bounds` and `_core.PyObjective`.
A subclass of `ObjectiveFunction` implements `singlepoint`, `multipoint` and `__repr__`.
Calling it counts calls in `.calls` and picks one of the two by comparing the input size with `limits.dims`.

```python
import numpy as np
import eindir


class Sphere(eindir.ObjectiveFunction):
    def singlepoint(self, pos):
        return float(np.sum(pos**2))

    def multipoint(self, pos):
        return np.sum(pos**2, axis=-1)

    def __repr__(self):
        return "Sphere"


low = np.array([-5.12, -5.12])
high = np.array([5.12, 5.12])
f = Sphere(eindir.NumLimit(low=low, high=high, dims=2))
assert f(np.array([1.0, 2.0])) == 5.0
assert np.array_equal(f(np.array([[1.0, 2.0], [0.0, 0.0]])), [5.0, 0.0])
assert f.calls == 2
f.limits.check(np.array([1.0, 2.0]))  # raises eindir.OutOfBounds outside the slack
```

The plain Python part also has a test function: `eindir.styblinski_tang(x)` evaluates Styblinski-Tang along the last dimension of any array API array.
Its minimum is about -39.166 per coordinate, at -2.9035 in each.

## Pitfalls

`scripts/pitfalls.py` reproduces each of these.

- `_core` functions take float64 NumPy arrays only.
  Lists and integer arrays raise TypeError ("cannot be converted to 'PyArray<T, D>'").
  A strided view such as `a[:, 0]` raises TypeError ("The given array is not contiguous").
  Use `np.ascontiguousarray(x, dtype=np.float64)`.
  `eindir.low_discrepancy_points` converts lists for you.
- `_core.Bounds` with any `low[i] > high[i]`, and `Bounds.clip` with a point of the wrong length, raise `pyo3_runtime.PanicException`.
  It derives from BaseException, so `except Exception` does not catch it, and Rust prints the panic to stderr first.
  Check the inputs before the call.
  `_core.low_discrepancy_points` raises ValueError for the same inverted bounds.
- eindir 0.5.1 from PyPI predates those checks, which came in 0.6.0.
  Its `Bounds` accepts `low > high`, and `contains` and `clip` look only at the coordinates the point has.
  A short point is judged or clipped on those alone, and a long one raises PanicException with `ndarray: index out of bounds`.
  Read `_core.__version__` before relying on either behaviour.
- Nothing checks lengths on evaluation: `obj.eval` of a 3-vector on a 2-D objective calls your function anyway, and from 0.6.0 `bounds.contains` of a wrong-length point returns False.
- `obj.grad` without `grad_fn` raises ValueError.
  Python has no finite-difference fallback, so pass a gradient or difference it yourself.
- A `PyObjective` belongs to the thread that made it.
  Calling it from another thread raises PanicException ("is unsendable, but sent to another thread").
- `skip=0` makes the first point the low corner of the box, and the default `skip=1` avoids it.
  `n=0` returns a 1-D empty array, not shape `(0, d)`.
- `NumLimit(low, high)` leaves `dims` at 1, and `ObjectiveFunction` then sends a single 2-D point to `multipoint`.
  Always pass `dims=len(low)`.
  Calling an `ObjectiveFunction` with a list raises AttributeError (`.ravel`).
- `pip install "eindir[all]"` installs nothing extra: the extras are `plotter` and `testing`.

## Scripts

- `scripts/surface.py` prints where each name lives and fails when the installed package no longer matches the table above.
- `scripts/pitfalls.py` asserts every pitfall above.

Run them with the Python that has eindir installed, from this directory:

```bash
python scripts/surface.py
python scripts/pitfalls.py
```

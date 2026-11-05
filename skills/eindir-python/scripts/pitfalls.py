"""Reproduce the traps listed under Pitfalls in SKILL.md.

    python scripts/pitfalls.py

Each check prints what the call did and exits non-zero if eindir no
longer behaves as SKILL.md says. Rust panics print their message and
location to stderr before Python sees them; that output is expected.
"""

import importlib.metadata
import sys
import threading

import numpy as np

import eindir
from eindir import _core

low = np.array([-5.12, -5.12])
high = np.array([5.12, 5.12])
bounds = _core.Bounds(low, high)
obj = _core.PyObjective(lambda x: float(np.sum(x**2)), bounds)
failures = 0


def outcome(fn):
    """('value', result) or ('raised', exception) for one call."""
    try:
        return "value", fn()
    except BaseException as exc:  # noqa: BLE001 - PanicException is a BaseException
        return "raised", exc


def check(label, fn, raises=None, value=None, base_exception_only=False):
    global failures
    kind, result = outcome(fn)
    if raises is None:
        ok = kind == "value" and (value is None or value(result))
        shown = repr(result)
    else:
        ok = kind == "raised" and type(result).__name__ == raises
        if ok and base_exception_only:
            ok = not isinstance(result, Exception)
        shown = f"{type(result).__name__}: {str(result).splitlines()[0]}" if kind == "raised" else repr(result)
    failures += not ok
    print(f"{'ok' if ok else 'FAIL':4} {label}: {' '.join(shown.split())}")


check("Bounds from lists", lambda: _core.Bounds([-1.0], [1.0]), raises="TypeError")
check("Bounds from an int array", lambda: _core.Bounds(np.array([-1]), np.array([1])), raises="TypeError")
# Bounds checks its arguments from 0.6.0 on.
if tuple(int(p) for p in _core.__version__.split(".")[:2]) >= (0, 6):
    check(
        "Bounds with low > high",
        lambda: _core.Bounds(np.array([1.0]), np.array([0.0])),
        raises="PanicException",
        base_exception_only=True,
    )
    check(
        "Bounds.clip with the wrong length",
        lambda: bounds.clip(np.array([0.0])),
        raises="PanicException",
        base_exception_only=True,
    )
    check("Bounds.contains with the wrong length", lambda: bounds.contains(np.array([0.0])), value=lambda r: r is False)
else:
    check("Bounds with low > high, before 0.6", lambda: _core.Bounds(np.array([1.0]), np.array([0.0])))
    check(
        "Bounds.clip of a short point, before 0.6",
        lambda: bounds.clip(np.array([9.0])),
        value=lambda r: np.array_equal(r, [5.12]),
    )
    check(
        "Bounds.contains of a short point, before 0.6",
        lambda: bounds.contains(np.array([0.0])),
        value=lambda r: r is True,
    )
    check(
        "Bounds.contains of a long point, before 0.6",
        lambda: bounds.contains(np.zeros(3)),
        raises="PanicException",
        base_exception_only=True,
    )
check("Bounds has no low or high attribute", lambda: bounds.low, raises="AttributeError")
check("PyObjective.eval of a list", lambda: obj.eval([1.0, 2.0]), raises="TypeError")
check("PyObjective.eval of a column view", lambda: obj.eval(np.arange(6.0).reshape(3, 2)[:, 0]), raises="TypeError")
check(
    "PyObjective.eval of the wrong length runs the function",
    lambda: obj.eval(np.array([1.0, 2.0, 3.0])),
    value=lambda r: r == 14.0,
)
check("PyObjective.grad without grad_fn", lambda: obj.grad(np.zeros(2)), raises="ValueError")

def eval_in_thread():
    box = {}
    worker = threading.Thread(target=lambda: box.update(outcome=outcome(lambda: obj.eval(np.zeros(2)))))
    worker.start()
    worker.join()
    kind, result = box["outcome"]
    if kind == "raised":
        raise result
    return result


check("PyObjective used from another thread", eval_in_thread, raises="PanicException", base_exception_only=True)

check(
    "_core.low_discrepancy_points with low > high",
    lambda: _core.low_discrepancy_points(np.array([1.0]), np.array([0.0]), 2),
    raises="ValueError",
)
check(
    "_core.low_discrepancy_points returns a list of lists",
    lambda: _core.low_discrepancy_points(low, high, 2),
    value=lambda r: isinstance(r, list) and isinstance(r[0], list),
)
check(
    "eindir.low_discrepancy_points accepts lists and returns an array",
    lambda: eindir.low_discrepancy_points([0.0, 0.0], [1.0, 1.0], 3),
    value=lambda r: isinstance(r, np.ndarray) and r.shape == (3, 2),
)
check(
    "skip=0 starts at the low corner",
    lambda: eindir.low_discrepancy_points([-1.0, -1.0], [1.0, 1.0], 1, skip=0),
    value=lambda r: np.array_equal(r, [[-1.0, -1.0]]),
)
check(
    "n=0 gives a 1-D empty array",
    lambda: eindir.low_discrepancy_points([0.0, 0.0], [1.0, 1.0], 0),
    value=lambda r: r.shape == (0,),
)
check("NumLimit dims defaults to 1", lambda: eindir.NumLimit(low=low, high=high).dims, value=lambda r: r == 1)


class Probe(eindir.ObjectiveFunction):
    def singlepoint(self, pos):
        return "singlepoint"

    def multipoint(self, pos):
        return "multipoint"

    def __repr__(self):
        return "Probe"


check(
    "ObjectiveFunction with dims left at 1 sends one 2-D point to multipoint",
    lambda: Probe(eindir.NumLimit(low=low, high=high))(np.zeros(2)),
    value=lambda r: r == "multipoint",
)
check(
    "ObjectiveFunction called with a list",
    lambda: Probe(eindir.NumLimit(low=low, high=high, dims=2))([0.0, 0.0]),
    raises="AttributeError",
)
check(
    "NumLimit.check outside the slack",
    lambda: eindir.NumLimit(low=low, high=high, dims=2).check(high + 1e-5),
    raises="OutOfBounds",
)
check("eindir has no __version__", lambda: eindir.__version__, raises="AttributeError")
check(
    "the extras are plotter and testing, with no 'all'",
    lambda: sorted(importlib.metadata.metadata("eindir").get_all("Provides-Extra") or []),
    value=lambda r: r == ["plotter", "testing"],
)

print(f"{failures} check(s) failed" if failures else "all checks passed")
sys.exit(1 if failures else 0)

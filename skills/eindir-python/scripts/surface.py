"""Check which eindir names are importable from Python, and from where.

    python scripts/surface.py

Prints one line per name and exits non-zero when the installed package
disagrees with the table in SKILL.md. When it does, update that table,
the pitfalls in AGENTS.md and the Python note in scripts/llms/llms.toml.
"""

import sys

import eindir
from eindir import _core

MODULES = {"eindir": eindir, "eindir._core": _core}
EXPECTED = {
    "FPair": {"eindir", "eindir._core"},
    "NumLimit": {"eindir"},
    "ObjectiveFunction": {"eindir"},
    "OutOfBounds": {"eindir"},
    "low_discrepancy_points": {"eindir", "eindir._core"},
    "styblinski_tang": {"eindir"},
    "array_namespace": {"eindir"},
    "dlpack_device": {"eindir"},
    "tvm_ffi_tensor": {"eindir"},
    "Bounds": {"eindir._core"},
    "PyObjective": {"eindir._core"},
    "shifted_low_discrepancy_points": {"eindir._core"},
    "__version__": {"eindir._core"},
}
RUST_ONLY = [
    "Objective",
    "GleThermostat",
    "optimal_sampling_drift",
    "matrix_exp",
    "ldl_sqrt",
    "halton_unit",
    "AdditiveSurrogate",
    "ChebyshevSurrogate",
    "ReducedObjective",
    "FiniteDiffGradient",
    "AnalyticGradient",
    "Rastrigin",
    "Rosenbrock",
    "Ackley",
    "StybTang2D",
]
EXPECTED.update({name: set() for name in RUST_ONLY})

print(f"eindir._core {_core.__version__}")
changed = 0
for name, places in EXPECTED.items():
    found = {label for label, module in MODULES.items() if hasattr(module, name)}
    status = "ok" if found == places else "CHANGED"
    changed += found != places
    print(f"{status:8} {name}: {', '.join(sorted(found)) or 'Rust only'}")

if changed:
    print(f"{changed} name(s) moved; update SKILL.md, AGENTS.md and scripts/llms/llms.toml")
    sys.exit(1)

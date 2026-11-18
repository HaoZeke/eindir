"""Objective functions, bounds and low-discrepancy points in N dimensions.

The Rust extension is ``eindir._core``; its version is
``eindir._core.__version__``. This package re-exports the pure-Python
components and the Array API and TVM-FFI helpers, and wraps the Halton
design as ``low_discrepancy_points``.
"""

import numpy as np

from eindir import _core as _core
from eindir._core import low_discrepancy_points as _core_low_discrepancy_points
from eindir.core.array_api import (
    ArrayPlacement,
    array_device,
    array_namespace,
    dlpack_device,
    placement,
    require_same_placement,
    styblinski_tang,
    to_namespace_device,
    to_reference,
)
from eindir.core.components import FPair, NumLimit, ObjectiveFunction
from eindir.core.exceptions import OutOfBounds
from eindir.core.tvm_ffi import (
    TvmFfiTensorMetadata,
    tvm_ffi_tensor,
    tvm_ffi_tensor_metadata,
)


def low_discrepancy_points(low, high, n: int, skip: int = 1):
    """Return ``n`` Halton points in the box ``[low, high]`` as an ``(n, d)`` array.

    ``low`` and ``high`` may be any sequence of numbers; they are converted
    to float64. Row ``r`` is Halton point ``skip + r``, so ``skip=1`` (the
    default) avoids the corner ``low``. The points match
    ``eindir_core::low_discrepancy_points`` in Rust.

    Raises ``ValueError`` when the lengths differ, the bounds are empty or
    ``high[i] < low[i]``.

    >>> low_discrepancy_points([0.0, 0.0], [1.0, 1.0], 2).tolist()
    [[0.5, 0.3333333333333333], [0.25, 0.6666666666666666]]
    """
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    return np.asarray(
        _core_low_discrepancy_points(low_arr, high_arr, int(n), int(skip)),
        dtype=np.float64,
    )


__all__ = [
    "ArrayPlacement",
    "FPair",
    "NumLimit",
    "ObjectiveFunction",
    "OutOfBounds",
    "TvmFfiTensorMetadata",
    "array_device",
    "array_namespace",
    "dlpack_device",
    "low_discrepancy_points",
    "placement",
    "require_same_placement",
    "styblinski_tang",
    "to_namespace_device",
    "to_reference",
    "tvm_ffi_tensor",
    "tvm_ffi_tensor_metadata",
]

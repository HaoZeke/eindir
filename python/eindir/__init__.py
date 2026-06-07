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
    """Return bounded low-discrepancy points as a NumPy array."""
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

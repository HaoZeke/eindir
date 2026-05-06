from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat as _array_api_compat


@dataclass(frozen=True)
class ArrayPlacement:
    namespace: Any
    device: Any


def _non_null_arrays(arrays: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(array for array in arrays if array is not None)


def array_namespace(*arrays: Any) -> Any:
    values = _non_null_arrays(arrays)
    if not values:
        raise ValueError("At least one array is required")
    return _array_api_compat.array_namespace(*values, use_compat=True)


def array_device(array: Any) -> Any:
    return _array_api_compat.device(array)


def placement(array: Any) -> ArrayPlacement:
    return ArrayPlacement(namespace=array_namespace(array), device=array_device(array))


def to_namespace_device(
    array: Any,
    *,
    xp: Any,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device

    try:
        converted = xp.asarray(array, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        converted = xp.asarray(array, **kwargs)

    if device is not None and array_device(converted) != device:
        converted = _array_api_compat.to_device(converted, device)
    return converted


def to_reference(array: Any, reference: Any, *, dtype: Any | None = None) -> Any:
    return to_namespace_device(
        array,
        xp=array_namespace(reference),
        device=array_device(reference),
        dtype=dtype,
    )


def require_same_placement(reference: Any, *arrays: Any) -> ArrayPlacement:
    reference_placement = placement(reference)
    for index, array in enumerate(arrays):
        current = placement(array)
        if current.namespace != reference_placement.namespace:
            raise ValueError(
                f"array {index} uses a different Array API namespace than the reference"
            )
        if current.device != reference_placement.device:
            raise ValueError(f"array {index} is on a different device than the reference")
    return reference_placement


def dlpack_device(array: Any) -> tuple[int, int]:
    device = getattr(array, "__dlpack_device__", None)
    if device is None:
        raise TypeError("array does not implement __dlpack_device__")
    return device()


def styblinski_tang(array: Any, *, axis: int = -1, xp: Any | None = None) -> Any:
    namespace = xp or array_namespace(array)
    x = namespace.asarray(array)
    x2 = x * x
    x4 = x2 * x2
    return namespace.sum(0.5 * (x4 - 16.0 * x2 + 5.0 * x), axis=axis)


__all__ = [
    "ArrayPlacement",
    "array_device",
    "array_namespace",
    "dlpack_device",
    "placement",
    "require_same_placement",
    "styblinski_tang",
    "to_namespace_device",
    "to_reference",
]

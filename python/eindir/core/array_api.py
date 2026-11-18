"""Array API helpers: find an array's namespace and device and move arrays.

The functions accept any array that ``array_api_compat`` recognises, such as
NumPy, CuPy, PyTorch or ``array_api_strict`` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat as _array_api_compat


@dataclass(frozen=True)
class ArrayPlacement:
    """The Array API namespace and the device of an array."""

    namespace: Any
    device: Any


def _non_null_arrays(arrays: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(array for array in arrays if array is not None)


def array_namespace(*arrays: Any) -> Any:
    """Return the Array API namespace shared by ``arrays``.

    ``None`` entries are skipped. The compat namespace is preferred when
    ``array_api_compat`` offers one.

    Raises ``ValueError`` when no array is given or the arrays come from
    different libraries, and ``TypeError`` for an unsupported array type.
    """
    values = _non_null_arrays(arrays)
    if not values:
        raise ValueError("At least one array is required")
    try:
        return _array_api_compat.array_namespace(*values, use_compat=True)
    except ValueError:
        return _array_api_compat.array_namespace(*values)


def array_device(array: Any) -> Any:
    """Return the device of ``array``, such as ``"cpu"`` for NumPy."""
    return _array_api_compat.device(array)


def placement(array: Any) -> ArrayPlacement:
    """Return the namespace and device of ``array`` together."""
    return ArrayPlacement(namespace=array_namespace(array), device=array_device(array))


def to_namespace_device(
    array: Any,
    *,
    xp: Any,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    """Convert ``array`` with ``xp.asarray`` and place it on ``device``.

    ``dtype`` and ``device`` are optional; ``None`` keeps what ``asarray``
    chooses. When ``xp.asarray`` does not take a ``device`` argument, the
    result is moved with ``array_api_compat.to_device`` instead.
    """
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
    """Convert ``array`` to the namespace and device of ``reference``."""
    return to_namespace_device(
        array,
        xp=array_namespace(reference),
        device=array_device(reference),
        dtype=dtype,
    )


def require_same_placement(reference: Any, *arrays: Any) -> ArrayPlacement:
    """Check that ``arrays`` share the namespace and device of ``reference``.

    Returns the placement of ``reference``. Raises ``ValueError`` naming the
    position of the first array that differs.
    """
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
    """Return ``array.__dlpack_device__()``, a ``(device_type, device_id)`` pair.

    Device type 1 is CPU in DLPack. Raises ``TypeError`` when the array does
    not implement ``__dlpack_device__``.
    """
    device = getattr(array, "__dlpack_device__", None)
    if device is None:
        raise TypeError("array does not implement __dlpack_device__")
    return device()


def styblinski_tang(array: Any, *, axis: int = -1, xp: Any | None = None) -> Any:
    """Styblinski-Tang function ``0.5 * sum(x**4 - 16 x**2 + 5 x)`` along ``axis``.

    Works in any Array API namespace; ``xp`` defaults to the namespace of
    ``array``. The minimum in each coordinate is near ``x = -2.903534``.

    >>> import numpy as np
    >>> float(styblinski_tang(np.zeros(2)))
    0.0
    """
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

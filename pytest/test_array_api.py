import numpy as np
import pytest

from eindir.core.array_api import (
    array_device,
    array_namespace,
    dlpack_device,
    require_same_placement,
    styblinski_tang,
    to_reference,
)


def test_numpy_namespace_device_and_styblinski_tang():
    points = np.asarray([[-2.903534, -2.903534], [0.0, 0.0]])

    xp = array_namespace(points)
    values = styblinski_tang(points)

    assert xp.asarray is not None
    assert array_device(points) == "cpu"
    assert np.asarray(values).shape == (2,)
    assert np.asarray(values)[0] == pytest.approx(-78.33198, abs=1e-4)
    assert np.asarray(values)[1] == pytest.approx(0.0)


def test_to_reference_preserves_reference_namespace_and_device():
    strict = pytest.importorskip("array_api_strict")
    reference = strict.asarray([1.0, 2.0])
    source = np.asarray([3.0, 4.0])

    converted = to_reference(source, reference)

    assert array_namespace(converted) == array_namespace(reference)
    assert array_device(converted) == array_device(reference)


def test_require_same_placement_reports_reference_placement():
    reference = np.asarray([1.0, 2.0])
    other = np.asarray([3.0, 4.0])

    placement = require_same_placement(reference, other)

    assert placement.namespace == array_namespace(reference)
    assert placement.device == "cpu"


def test_dlpack_device_numpy_reports_cpu():
    array = np.asarray([1.0, 2.0])

    assert dlpack_device(array) == (1, 0)

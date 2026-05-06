import numpy as np
import pytest

from eindir.core.tvm_ffi import (
    TvmFfiTensorMetadata,
    tvm_ffi_tensor,
    tvm_ffi_tensor_metadata,
)


class _FakeTvmFfi:
    def __init__(self):
        self.arrays = []

    def from_dlpack(self, array):
        self.arrays.append(array)
        return ("tvm", array)


def test_tvm_ffi_tensor_uses_dlpack_provider_without_required_dependency():
    array = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    fake = _FakeTvmFfi()

    converted = tvm_ffi_tensor(array, tvm_ffi=fake)

    assert converted == ("tvm", array)
    assert fake.arrays == [array]


def test_tvm_ffi_tensor_metadata_reports_dlpack_boundary():
    array = np.asarray([[1.0, 2.0]], dtype=np.float32)

    metadata = tvm_ffi_tensor_metadata(array)

    assert isinstance(metadata, TvmFfiTensorMetadata)
    assert metadata.shape == (1, 2)
    assert metadata.dtype == "float32"
    assert metadata.dlpack_device == (1, 0)


def test_tvm_ffi_tensor_rejects_non_dlpack_objects():
    with pytest.raises(TypeError, match="__dlpack__"):
        tvm_ffi_tensor([1.0, 2.0], tvm_ffi=_FakeTvmFfi())

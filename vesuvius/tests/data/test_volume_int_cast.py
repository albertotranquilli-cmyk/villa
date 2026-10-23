"""Volume(return_as_type=<int>) must not wrap or truncate."""
import numpy as np
import zarr

from vesuvius.data.volume import Volume


def _vol(tmp_path, data, **kw):
    g = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    g.zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)[...] = data
    return Volume(type="zarr", path=str(tmp_path / "v.zarr"), verbose=False, **kw)


def test_uint16_to_uint8_rescales_instead_of_wrapping(tmp_path):
    data = np.array([0, 255, 256, 511, 32768, 65535], np.uint16).reshape(1, 2, 3)
    out = _vol(tmp_path, data, return_as_type="np.uint8")[0:1, 0:2, 0:3]
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, (data >> 8).astype(np.uint8))


def test_int16_to_uint8_clips(tmp_path):
    data = np.array([-5, 0, 100, 300], np.int16).reshape(1, 2, 2)
    out = _vol(tmp_path, data, return_as_type="np.uint8")[0:1, 0:2, 0:2]
    np.testing.assert_array_equal(out.ravel(), [0, 0, 100, 255])


def test_uint8_to_uint16_is_plain_cast(tmp_path):
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    out = _vol(tmp_path, data, return_as_type="np.uint16")[0:2, 0:2, 0:2]
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, data)


def test_minmax_to_uint8_rounds(tmp_path):
    data = np.arange(1000, dtype=np.uint16).reshape(1, 1, -1)
    out = _vol(tmp_path, data, normalization_scheme="instance_minmax",
               return_as_type="np.uint8")[0:1, 0:1, 0:1000]
    scaled = (data.astype(np.float32) / np.float32(999)) * np.float32(255)
    np.testing.assert_array_equal(out, np.rint(scaled).astype(np.uint8))

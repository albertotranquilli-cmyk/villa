"""End-to-end Volume read path vs an independent Python-int oracle.

Every (source dtype, target dtype) pair is written to a small local zarr store
(v2, and v3 when zarr>=3 is installed), read back through ``Volume.__getitem__``
and compared element-wise with an oracle that uses only Python ints/Fractions.
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])
FORMATS = [2] + ([3] if _ZARR_MAJOR >= 3 else [])

INT_SRC = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOAT_SRC = ["float16", "float32", "float64"]
INT_TGT = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOAT_TGT = ["float32", "float64"]


def _src_values(src):
    dt = np.dtype(src)
    if dt.kind in "iu":
        i = np.iinfo(dt)
        vals = {i.min, i.min + 1, i.max, i.max - 1, 0, 1, 127, 128, 255, 256, 32767, 32768, 65535, 65536,
                2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**53 + 1, 2**63 - 1, -1, -128, -129, -32768, -32769, -2**31}
        return np.array(sorted(v for v in vals if i.min <= v <= i.max), dtype=dt)
    f = np.finfo(dt)
    vals = [0.0, -0.0, 0.4, 0.5, 1.5, 2.5, -0.5, -1.5, 127.5, 255.49, 255.5, 256.0, -128.5, 32767.5,
            65535.4, 2.0**31 - 1, 2.0**31, 2.0**32 - 1, 2.0**32, 2.0**53, 2.0**63, 2.0**64, -2.0**63,
            -2.0**63 - 4096, 1e30, -1e30, float(f.max), float(f.min), float(f.tiny),
            math.nan, math.inf, -math.inf]
    with np.errstate(over="ignore"):
        return np.array(vals, dtype=np.float64).astype(dt)


def _read(tmp_path, data, fmt, **kw):
    path = str(tmp_path / f"v{fmt}.zarr")
    if _ZARR_MAJOR >= 3:
        g = zarr.open_group(path, mode="w", zarr_format=fmt)
        a = g.create_array(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        g = zarr.open_group(path, mode="w")
        a = g.zeros(name="0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    a[...] = data
    v = Volume(type="zarr", path=path, verbose=False, **kw)
    return v[0:1, 0:1, 0:data.shape[-1]].ravel()


def _round_half_even(q: Fraction) -> int:
    return round(q)  # Python's round on Fraction is exact banker's rounding


def _oracle_none(x, src, tgt):
    s, t = np.dtype(src), np.dtype(tgt)
    ti = np.iinfo(t)
    if s.kind in "iu":
        v = int(x)
        if s.kind == "u" and t.kind == "u" and np.iinfo(s).max > ti.max:
            return v >> (np.iinfo(s).bits - ti.bits)
        return min(max(v, ti.min), ti.max)
    f = float(x)
    if math.isnan(f):
        return 0
    if math.isinf(f):
        return ti.max if f > 0 else ti.min
    return min(max(_round_half_even(Fraction(f)), ti.min), ti.max)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_int_target(tmp_path, fmt, src, tgt):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    expected = [_oracle_none(x, src, tgt) for x in data.ravel()]
    assert [int(v) for v in out] == expected


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", FLOAT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_float_target(tmp_path, fmt, src, tgt):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(out, data.ravel().astype(tgt))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_no_normalization_no_target_is_identity(tmp_path, fmt, src):
    data = _src_values(src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt)
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data.ravel())


def _minmax_tolerance(tgt, value):
    # <= 1 unit (half-way cases computed in float64) for <=32-bit targets; for
    # 64-bit targets float64 cannot represent the result, allow one float64 ulp.
    if np.iinfo(tgt).bits < 64:
        return 1
    return max(1, int(np.spacing(np.float64(abs(value)))))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
@pytest.mark.parametrize("src", INT_SRC + FLOAT_SRC)
def test_instance_minmax_int_target(tmp_path, fmt, src, tgt):
    vals = _src_values(src)
    if vals.dtype.kind == "f":
        # keep a finite range float32 can subtract without overflow
        vals = vals[np.isfinite(vals) & (np.abs(vals.astype(np.float64)) <= 2.0**64)]
    data = vals.reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    assert out.dtype == np.dtype(tgt)
    xs = [Fraction(int(x)) if vals.dtype.kind in "iu" else Fraction(float(x)) for x in vals]
    lo, hi = min(xs), max(xs)
    ti = np.iinfo(tgt)
    for x, got in zip(xs, out):
        exact = _round_half_even((x - lo) / (hi - lo) * ti.max)
        assert 0 <= int(got) <= ti.max
        assert abs(int(got) - exact) <= _minmax_tolerance(tgt, exact), (x, int(got), exact)
    assert int(out[int(np.argmin(xs))]) == 0
    assert int(out[int(np.argmax(xs))]) == ti.max


# Concrete witnesses of the float32 limit/scaling bug (fail on 79ed0d4).
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("src,vals,tgt,idx,exact", [
    # uint32 2**32-2 is not the max but float32 rounds it to 2**32 -> saturated to max
    ("uint32", [0, 1, 2**32 - 2, 2**32 - 1], "uint32", 2, 2**32 - 2),
    # float32 [0,1] quantization: 128/255 * (2**32-1) off by 127 units
    ("uint8", list(range(256)), "uint32", 128, 2155905152),
    ("uint8", list(range(256)), "int32", 128, 1077952576),
    # int32 -1 lands exactly on a float32 tie: off by one even for uint8
    ("int32", [-2**31, -1, 0, 1, 2**31 - 1], "uint8", 1, 127),
])
def test_instance_minmax_witnesses(tmp_path, fmt, src, vals, tgt, idx, exact):
    data = np.array(vals, dtype=src).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type=f"np.{tgt}")
    assert int(out[idx]) == exact


@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_minmax_uint64_witness(tmp_path, fmt):
    # float32 [0,1] scaled to uint64: was off by 2**39 units
    data = np.arange(256, dtype=np.uint8).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type="np.uint64")
    exact = round(Fraction(128, 255) * (2**64 - 1))
    assert abs(int(out[128]) - exact) <= 2048


@pytest.mark.parametrize("fmt", FORMATS)
def test_minmax_float_output_stays_float32(tmp_path, fmt):
    data = np.arange(10, dtype=np.uint16).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax")
    assert out.dtype == np.float32
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_minmax", return_as_type="np.float32")
    assert out.dtype == np.float32


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
def test_zscore_int_target_stays_float32(tmp_path, fmt, tgt):
    data = np.arange(10, dtype=np.uint16).reshape(1, 1, -1)
    out = _read(tmp_path, data, fmt, normalization_scheme="instance_zscore", return_as_type=f"np.{tgt}")
    assert out.dtype == np.float32


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tgt", INT_TGT)
def test_percentile_minmax_int_target_in_range(tmp_path, fmt, tgt):
    data = np.arange(1000, dtype=np.uint32).reshape(1, 1, -1) * 4000
    out = _read(tmp_path, data, fmt, normalization_scheme="percentile_minmax", return_as_type=f"np.{tgt}")
    ti = np.iinfo(tgt)
    assert out.dtype == np.dtype(tgt)
    assert int(out[0]) == 0 and int(out[-1]) == ti.max
    assert np.all(np.diff(out.astype(np.float64)) >= 0)

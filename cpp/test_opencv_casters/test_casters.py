import numpy as np
import pytest

import sys
import os

sys.path.append("./lib/")
import cvcasters_test as C


@pytest.mark.parametrize(
    "dtype,cvdepth",
    [
        (np.uint8, "CV_8U"),
        (np.int8, "CV_8S"),
        (np.uint16, "CV_16U"),
        (np.int16, "CV_16S"),
        (np.int32, "CV_32S"),
        (np.float32, "CV_32F"),
        (np.float64, "CV_64F"),
    ],
)
@pytest.mark.parametrize("shape", [(5, 7), (3, 4, 1), (3, 4, 3)])
def test_mat_roundtrip_identity(dtype, cvdepth, shape):
    a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    out = C.identity_mat(a)
    assert isinstance(out, np.ndarray)
    assert out.dtype == dtype
    expected_shape = (shape[0], shape[1]) if (len(shape) == 3 and shape[2] == 1) else shape
    assert out.shape == expected_shape
    # Values should match exactly - reshape input to match output shape for comparison
    if len(shape) == 3 and shape[2] == 1:
        # For single-channel 3D arrays, compare with 2D version
        np.testing.assert_array_equal(out, a.reshape(shape[0], shape[1]))
    else:
        np.testing.assert_array_equal(out, a)


def test_mat_empty():
    a = np.empty((0, 0), dtype=np.uint8)
    out = C.identity_mat(a)
    assert out.shape == (0, 0)
    assert out.dtype == np.uint8


def test_mat_non_contiguous_copy_in():
    base = np.arange(100, dtype=np.uint8).reshape(10, 10)
    a = base[::2, ::2]  # non-contiguous 5x5 view
    out = C.identity_mat(a)
    # Should still equal values
    np.testing.assert_array_equal(out, a)


def test_mat_int64_converts_to_float64():
    # Small values to avoid precision issues in float64
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    out = C.identity_mat(a)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == a.shape
    np.testing.assert_array_equal(out, a.astype(np.float64))


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_add_scalar(channels):
    H, W = 4, 5
    if channels == 1:
        a = np.zeros((H, W), dtype=np.uint8)
    else:
        a = np.zeros((H, W, channels), dtype=np.uint8)
    out = C.add_scalar(a, 10.0)
    np.testing.assert_array_equal(out, np.full_like(a, 10))


def test_vec3f_roundtrip():
    v = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    out = C.roundtrip_vec3f(v)
    assert out.shape == (3,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, v)


def test_point_roundtrip():
    p = (3, 7)
    out = C.roundtrip_point(p)
    assert tuple(out) == p


def test_point2f_roundtrip():
    p = (2.5, -1.25)
    out = C.roundtrip_point2f(p)
    assert pytest.approx(out[0]) == p[0]
    assert pytest.approx(out[1]) == p[1]


def test_rect_roundtrip():
    ri = (1, 2, 10, 20)
    rf = (1.5, -2.25, 10.0, 5.5)
    outi = C.roundtrip_recti(ri)
    outf = C.roundtrip_rectf(rf)
    assert tuple(outi) == ri
    assert tuple(outf) == rf


def test_dmatch_roundtrip():
    d = (5, 10, 2, 0.75)
    out = C.roundtrip_dmatch(d)
    assert tuple(out) == d


def test_keypoint_roundtrip_with_and_without_classid():
    k6 = (10.0, 20.0, 31.0, 45.0, 0.8, 2)
    k7 = (10.0, 20.0, 31.0, 45.0, 0.8, 2, 42)

    out6 = C.roundtrip_keypoint(k6)  # should add class_id=-1
    out7 = C.roundtrip_keypoint(k7)

    assert len(out6) == 7 and out6[-1] == -1
    assert len(out7) == 7 and out7[-1] == 42

    # common fields equal
    for i in range(6):
        assert pytest.approx(out6[i]) == k6[i]
        assert pytest.approx(out7[i]) == k7[i]


def test_vector_of_mats_roundtrip():
    mats = [
        np.arange(12, dtype=np.uint8).reshape(3, 4),
        np.ones((2, 2, 3), dtype=np.uint8) * 7,
    ]
    out = C.roundtrip_mats(mats)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_array_equal(out[0], mats[0])
    np.testing.assert_array_equal(out[1], mats[1])

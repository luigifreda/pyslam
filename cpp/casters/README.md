# OpenCV Type Casters

This directory contains the pybind11 type casters used to convert between
OpenCV types and Python/NumPy objects.

## Canonical Header

The active implementation is `opencv_type_casters.h`.

It is used by:

- multiple pybind11 modules under `cpp/`
- `pyslam/slam/cpp`, where `casters/opencv_type_casters.h` is a symlink to this file
- `thirdparty/orbslam2_features`

## Tests

The caster-specific checks referenced in the header live under
`cpp/test_opencv_casters/`.

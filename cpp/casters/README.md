# OpenCV Type Casters

This directory contains the pybind11 type casters used to convert between
OpenCV types and Python/NumPy objects.

## Canonical Header

The active implementation is `opencv_type_casters.h`.

It is used by:

- multiple pybind11 modules under `cpp/modules/`
- `cpp/core/slam`, which adds `cpp/` to its include path and includes `casters/opencv_type_casters.h`
- `thirdparty/orbslam2_features`

## Tests

The caster-specific checks referenced in the header live under
`cpp/modules/test_opencv_casters/`.

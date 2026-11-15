# Shared pybind11 Type Casters

This directory contains the pybind11 type casters shared by several extensions:

- `opencv_type_casters.h`: OpenCV types to and from Python/NumPy objects.
- `nlohmann_json_type_casters.h`: `nlohmann::json` to and from Python objects (used by the camera bindings).
- `dictionary_casters.h`: `pyslam::Dict` / `List` from `cpp/core/common/dictionary.h`.

SLAM-only casters (`slam_opaque_types.h`) stay in `cpp/core/slam/casters`.

## OpenCV Casters

The active implementation is `opencv_type_casters.h`.

It is used by:

- multiple pybind11 modules under `cpp/modules/`
- `cpp/core/slam`, which adds `cpp/` to its include path and includes `casters/opencv_type_casters.h`
- `thirdparty/orbslam2_features`

## Tests

The caster-specific checks referenced in the header live under
`cpp/modules/test_opencv_casters/`.

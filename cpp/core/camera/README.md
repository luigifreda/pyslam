# Camera models

The C++ camera classes used by `cpp_core` (sparse SLAM and direct SLAM). They are compiled into `cpp_core`. The header-only projection math, rig geometry, edges and factors are also shared with the Python-core `camera_models` extensions in [`cpp/modules/camera_models`](../../modules/camera_models/README.md), which documents the models in detail.

Include these headers with their folder, for example `#include "camera/camera.h"`. This folder depends only on [`cpp/core/common`](../common/README.md), Eigen and OpenCV. The g2o and gtsam headers also need their optimizer.

| File | Role |
| --- | --- |
| `camera.h/.cpp` | `Camera` base class: intrinsics, sensor type, stereo/RGBD parameters, projection and unprojection API |
| `camera_models.h/.cpp` | `PinholeCamera`, `KannalaBrandt4Camera`, rational pinhole and double-sphere cameras |
| `camera_model_math.h` | Projection polynomials and analytic Jacobians (Eigen only), the single copy for both cores |
| `camera_pose.h/.cpp` | `CameraPose` (`Tcw`) |
| `camera_rig_geometry.h` | Rig projection, rig triangulation, bearing-to-plane distance and stereo-rig search (templates) |
| `camera_handle.h` | Owning or non-owning camera reference held by edges and factors |
| `camera_projection_edges_g2o.h`, `camera_edges_g2o.h` | g2o projection edges (templates, then the `cpp_core` aliases) |
| `camera_projection_factors_gtsam.h`, `camera_factors_gtsam.h` | gtsam projection factors (templates, then the `cpp_core` aliases) |
| `camera_serialization_json.cpp`, `camera_serialization_py.cpp` | JSON and NumPy-state serialization |
| `py_module/` | pybind11 camera bindings (`camera_module.h`) and the g2o/gtsam edge bindings, compiled into `cpp_core` |

g2o and gtsam headers are never included in the same translation unit, so the two edge bindings are separate source files.

The tests stay with the `cpp_core` suites in [`cpp/core/slam/tests_cpp`](../slam/tests_cpp): `test_camera_model_math`, `test_camera_models`, `test_camera_edges_g2o`, `test_camera_factors_gtsam`, `test_numpy_camera` and `test_json_camera`.

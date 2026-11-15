# Shared C++ headers

Header-only utilities shared by [`camera`](../camera/README.md), [`slam`](../slam/README.md), [`slam_direct`](../slam_direct/README.md), `semantics`, and the `camera_models` / `color_utils` extensions. Include them with their folder, for example `#include "common/eigen_aliases.h"`. Nothing here includes `camera/`, `slam/` or `slam_direct/`.

| File | Role |
| --- | --- |
| `eigen_aliases.h` | `Vec3d`, `Mat3d`, `MatNx3`, ... Eigen type aliases |
| `eigen_helpers.h` | Small Eigen helpers (`skew`, ...) |
| `smart_pointers.h` | `DECLARE_SMART_POINTERS` and pointer helpers |
| `dictionary.h` | `Dict` / `List` settings containers (pybind casters in `cpp/casters/dictionary_casters.h`) |
| `messages.h` | Colored console messages (ANSI codes) |
| `serialization_json.h`, `base64_codec.h` | JSON serialization of Eigen/OpenCV types |
| `serialization_numpy.h`, `numpy_helpers.h` | NumPy-state serialization and conversions |
| `descriptor_helpers.h` | Descriptor norm types and Hamming/L2 distances |
| `owned_g2o_optimizer.h` | `g2o::SparseOptimizer` that retains every native allocation, plus a thread-safe abort flag |
| `feature_trails.h`, `color_helpers.h` | Feature-trail drawing and ID colors shared by `Frame` and `DirectSlam` |

SLAM-specific helpers (geometry, features, image processing, map access) stay in [`cpp/core/slam/utils`](../slam/utils).

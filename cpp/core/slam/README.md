# pySLAM C++ Core Module

<!-- TOC -->

- [pySLAM C++ Core Module](#pyslam-c-core-module)
  - [Code organization and overview](#code-organization-and-overview)
  - [Table of corresponding implementation files](#table-of-corresponding-implementation-files)
  - [Architecture and Design Goals](#architecture-and-design-goals)
  - [Core Principles](#core-principles)

<!-- /TOC -->


This directory contains the **C++ implementations** of the core SLAM classes, exposed to Python via **pybind11** bindings. The system provides a modular **sparse-SLAM core**, implemented in **both Python and C++**, allowing users to switch between high-performance/speed and high-flexibility modes. 


The C++ core reimplements the sparse SLAM originally implemented in Python, exposing core SLAM classes (frames, keyframes, map points, maps, cameras, optimizers, tracking, and local mapping) to Python via pybind11. The C++ implementation follows a streamlined design where all core data resides in C++, with Python serving as an interface layer. C++ classes mirror their Python counterparts, maintaining identical interfaces and data field names (see the [table](#table-of-corresponding-files) below). The bindings support zero-copy data exchange (e.g., descriptors) and safe memory ownership across the Python/C++ boundary, leveraging automatic zero-copy sharing of NumPy array memory with C++ when possible.

- The C++ sparse-SLAM core is **enabled by default** (`USE_CPP_CORE = True` in `pyslam/config_parameters.py`). Set it to `False`, or launch with `PYSLAM_USE_CPP=false`, to run the Python core instead.
- To **rebuild** the C++ core module, run
    ```bash
    . pyenv-activate.sh 
    ./build_cpp_core.sh
    ```

While this may be self-evident, it is important to keep in mind that when `USE_CPP_CORE = True`:
- The Python implementation of the sparse SLAM core is effectively bypassed, and any modifications to it will have no effect at runtime.
- All functional changes to the sparse SLAM C++ codebase must be rebuilt using `./build_cpp_core.sh` (as explained above) in order to take effect.

### macOS, conda Python, and local OpenCV

`cpp_core` links **OpenCV** from `thirdparty/opencv/install` and **TBB** must be the same library that **`libopencv_core`** was built against (CMake now infers it from `otool -L` / `ldd`). If OpenCV was built with **Homebrew** TBB but you run **conda** Python, you can get a **segfault on `import cpp_core`** (mixed allocators / duplicate TBB).

**Fix:** in the conda env install **`tbb` and `tbb-devel`** (the latter supplies `lib/cmake/TBB`): `conda install -y -c conda-forge tbb tbb-devel`. Without `tbb-devel`, OpenCV CMake usually picks Homebrew TBB while your script may still succeed. Verify with `otool -L thirdparty/opencv/install/lib/libopencv_core*.dylib | grep tbb` — paths should live under **`$CONDA_PREFIX`**, not `/opt/homebrew`. Then wipe `thirdparty/opencv/build` and `thirdparty/opencv/install`, rerun `./scripts/install_opencv_local.sh`, and rebuild **`cpp_core`**. CMake defaults `OpenCV_DIR` to `thirdparty/opencv/install/lib/cmake/opencv4` when unset.

**macOS `cpp_core` link:** do **not** link `libpython3.11.dylib` into the extension. The build uses `-undefined dynamic_lookup` and skips `${Python_LIBRARIES}` on `APPLE` so Python symbols resolve from the embedding interpreter. Linking `libpython` into the `.so` often crashes in `take_gil` during `PyInit_cpp_core` even when OpenCV/TBB are correct.


---

## Code organization and overview

```bash
├── cpp
│   ├── modules     # utility pybind11 extensions (solvers, volumetric, camera_models, ...)
│   │   └── tests   # smoke tests for those extensions
│   ├── casters     # shared OpenCV pybind11 type casters
│   └── core
│       ├── semantics
│       │   └── py_module   # pybind11 bindings compiled into cpp_core
│       └── slam            # sparse-SLAM C++ core
│           ├── py_module   # pybind11 module definitions for sparse slam
│           ├── casters     # pybind11 type casters for Eigen, JSON, dictionaries
│           ├── utils       # helpers for geometry, features, descriptors, etc.
│           ├── tests_cpp   # C++ unit tests
│           └── tests_py    # pybind11 integration tests
├── pyslam/slam/cpp/__init__.py   # Python facade that imports cpp_core
```

The codebase is organized into the following categories:

- **Core SLAM Classes** — Main C++ implementations (`frame.cpp/h`, `keyframe.cpp/h`, `map_point.cpp/h`, `map.cpp/h`, `camera.cpp/h`, `camera_pose.cpp/h`, `optimizer_g2o.cpp/h`, `slam.cpp/h`, `tracking.cpp/h`, `tracking_core.cpp/h`, `local_mapping.cpp/h`, `local_mapping_core.cpp/h`).

- **Serialization** — JSON and NumPy serialization modules (`*_serialization_json.cpp`, `*_serialization_py.cpp`) for data persistence and Python interop.

- **Python Bindings** — `py_module/` contains pybind11 module definitions (`cpp_core_module.cpp`, `*_module.h`) that expose C++ classes to Python.

- **Type Casters** — `casters/` provides pybind11 type casters for Eigen, JSON, and dictionary types. The OpenCV caster is the shared header in `cpp/casters`.

- **Utilities** — `utils/` houses helper functions for geometry, features, descriptors, image processing, NumPy/Eigen operations, and serialization.

- **Tests** — `tests_cpp/` and `tests_py/` contain C++ unit tests and Python integration tests respectively. See [tests/README.md](../../../tests/README.md) for how to run them.

---

## Table of corresponding implementation files 

| C++ Implementation | Python Implementation | Description |
|-------------------|----------------------|-------------|
| **Core SLAM Classes** |
| `frame.cpp/h` | `pyslam/slam/frame.py` | Frame class for storing image frames and features |
| `keyframe.cpp/h` | `pyslam/slam/keyframe.py` | Keyframe class for storing keyframes in the map |
| `map_point.cpp/h` | `pyslam/slam/map_point.py` | MapPoint class for 3D points in the map |
| `map.cpp/h` | `pyslam/slam/map.py` | Map class managing keyframes and map points |
| `camera.cpp/h`, `camera_models.cpp/h`, `camera_model_math.h` | `pyslam/slam/camera.py` | Pinhole, Kannala–Brandt4, rational pinhole, and double-sphere cameras |
| `camera_pose.cpp/h` | `pyslam/slam/camera_pose.py` | Camera pose representation |
| `slam.cpp/h` | `pyslam/slam/slam.py` | SLAM system; unported collaborators (feature tracker, loop closing, ...) stay Python objects |
| `slam_dynamic_config.cpp/h` | `pyslam/slam/slam_dynamic_config.py` | Running statistics that adapt the descriptor-distance and reprojection-error thresholds |
| `tracking.cpp/h` | `pyslam/slam/tracking.py` | Tracking control flow |
| `tracking_core.cpp/h` | `pyslam/slam/tracking_core.py` | Core tracking functionality |
| `local_mapping.cpp/h` | `pyslam/slam/local_mapping.py` | Local mapping queue, idle flag, and thread |
| `local_mapping_core.cpp/h` | `pyslam/slam/local_mapping_core.py` | Local mapping core operations |
| `geometry_matchers.cpp/h` | `pyslam/slam/geometry_matchers.py` | Geometric matching utilities |
| **Optimizers** |
| `optimizer_g2o.cpp/h` | `pyslam/slam/optimizer_g2o.py` | G2O-based bundle adjustment optimizer |
| `optimizer_gtsam.cpp/h` | `pyslam/slam/optimizer_gtsam.py` | GTSAM-based optimizer |
| **Supporting Classes** |
| `sim3_pose.h` | `pyslam/slam/sim3_pose.py` | Similarity transformation (Sim3) pose |
| `rotation_histogram.h` | `pyslam/slam/rotation_histogram.py` | Rotation histogram for loop closure |
| `config_parameters.cpp/h` | `pyslam/config_parameters.py` | Configuration parameters |
| `ckdtree_eigen.h` | `pyslam/slam/ckdtree.py` | KD-tree for efficient nearest neighbor search |
| **Serialization** |
| `*_serialization_json.cpp` | (in Python classes) | JSON serialization for persistence |
| `*_serialization_py.cpp` | (in Python classes) | NumPy/Python serialization for interop |
| **Python Bindings** |
| `py_module/*_module.h` | N/A | pybind11 module definitions exposing C++ to Python |
| **Utilities** |
| `utils/features.cpp/h` | `pyslam/utilities/features.py` | Feature extraction and matching utilities |
| `utils/geom_2views.cpp/h` | `pyslam/utilities/geometry.py` | Two-view geometry functions |
| `utils/*.h` | Various Python modules | Helper utilities (image processing, descriptors, etc.) |

---

## Architecture and Design Goals

The following design requirements guide the development of the C++ core  
(*note: the module is still being refined and optimized*):  

- C++ classes **mirror** their Python counterparts, maintaining identical interfaces and data field names.    
- Support for **zero-copy data exchange** (e.g., descriptors) and **safe memory ownership** across the Python/C++ boundary, *wherever feasible and covenient*.  

## Core Principles

The C++ core adopts a streamlined design philosophy:  

- **All core data resides in C++** — Python serves purely as an interface layer.  
- **Direct pybind11 exposure** — Python objects are lightweight views of underlying C++ objects.  
- **Automatic zero-copy** — pybind11 automatically shares NumPy array memory with C++ when possible.  
- **RAII-based ownership** — C++ smart pointers manage object lifetimes safely and efficiently.

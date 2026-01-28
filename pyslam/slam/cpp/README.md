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

- To **enable** the C++ sparse-SLAM core, set `USE_CPP_CORE = True` in `pyslam/config_parameters.py`. 
- To **rebuild** the C++ core module, run
    ```bash
    . pyenv-activate.sh 
    ./build_cpp_core.sh
    ```

While this may be self-evident, it is important to keep in mind that when `USE_CPP_CORE = True`:
- The Python implementation of the sparse SLAM core is effectively bypassed, and any modifications to it will have no effect at runtime.
- All functional changes to the sparse SLAM C++ codebase must be rebuilt using `./build_cpp_core.sh` (as explained above) in order to take effect.


---

## Code organization and overview

```bash
├── pyslam      # Core Python package
│   ...
│   ├── semantics
│       ├── cpp  # C++ core for semantics  
│           ├── py_module   # pybind11 module definitions for semantics 
│   ...
│   ├── slam
│       ├── cpp  # C++ core for sparse slam  
│           ├── py_module   # pybind11 module definitions for sparse slam 
│           ├── casters     # pybind11 type casters for OpenCV, Eigen, JSON, etc. 
│           ├── utils       # helpers for geometry, features, descriptors, etc.
│           ├── test_cpp    # C++ unit tests
│           ├── test_py     # pybind11 integration tests 
│   ...
```

The codebase is organized into the following categories:

- **Core SLAM Classes** — Main C++ implementations (`frame.cpp/h`, `keyframe.cpp/h`, `map_point.cpp/h`, `map.cpp/h`, `camera.cpp/h`, `camera_pose.cpp/h`, `optimizer_g2o.cpp/h`, `tracking_core.cpp/h`, `local_mapping_core.cpp/h`).

- **Serialization** — JSON and NumPy serialization modules (`*_serialization_json.cpp`, `*_serialization_py.cpp`) for data persistence and Python interop.

- **Python Bindings** — `py_module/` contains pybind11 module definitions (`cpp_core_module.cpp`, `*_module.h`) that expose C++ classes to Python.

- **Type Casters** — `casters/` provides pybind11 type casters for custom type conversions (OpenCV, Eigen, JSON, dictionary types).

- **Utilities** — `utils/` houses helper functions for geometry, features, descriptors, image processing, NumPy/Eigen operations, and serialization.

- **Tests** — `tests_cpp/` and `tests_py/` contain C++ unit tests and Python integration tests respectively.

---

## Table of corresponding implementation files 

| C++ Implementation | Python Implementation | Description |
|-------------------|----------------------|-------------|
| **Core SLAM Classes** |
| `frame.cpp/h` | `pyslam/slam/frame.py` | Frame class for storing image frames and features |
| `keyframe.cpp/h` | `pyslam/slam/keyframe.py` | Keyframe class for storing keyframes in the map |
| `map_point.cpp/h` | `pyslam/slam/map_point.py` | MapPoint class for 3D points in the map |
| `map.cpp/h` | `pyslam/slam/map.py` | Map class managing keyframes and map points |
| `camera.cpp/h` | `pyslam/slam/camera.py` | Camera model and calibration |
| `camera_pose.cpp/h` | `pyslam/slam/camera_pose.py` | Camera pose representation |
| `tracking_core.cpp/h` | `pyslam/slam/tracking_core.py` | Core tracking functionality |
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

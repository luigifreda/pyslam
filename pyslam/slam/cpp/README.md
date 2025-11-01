# PYSLAM C++ Core Module

<!-- TOC -->

- [PYSLAM C++ Core Module](#pyslam-c-core-module)
  - [Code overview](#code-overview)
  - [Architecture and Design Goals](#architecture-and-design-goals)
  - [Core Principles](#core-principles)

<!-- /TOC -->


This directory contains the **C++ implementations** of the core SLAM classes, exposed to Python via **pybind11** bindings.  
The module is currently **under active development**.


## Code overview

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

- **Core SLAM Classes** — Main C++ implementations (`frame.cpp/h`, `keyframe.cpp/h`, `map_point.cpp/h`, `map.cpp/h`, `camera.cpp/h`, `camera_pose.cpp/h`, `optimizer_g2o.cpp/h`, `tracking_core.cpp/h`, `local_mapping_core.cpp/h`)

- **Serialization** — JSON and NumPy serialization modules (`*_serialization_json.cpp`, `*_serialization_py.cpp`) for data persistence and Python interop

- **Python Bindings** — `py_module/` contains pybind11 module definitions (`cpp_core_module.cpp`, `*_module.h`) that expose C++ classes to Python

- **Type Casters** — `casters/` provides pybind11 type casters for custom type conversions (OpenCV, Eigen, JSON, dictionary types)

- **Utilities** — `utils/` houses helper functions for geometry, features, descriptors, image processing, NumPy/Eigen operations, and serialization

- **Tests** — `tests_cpp/` and `tests_py/` contain C++ unit tests and Python integration tests respectively


## Architecture and Design Goals

The following design requirements guide the development of the C++ core  
(*note: the module is still being refined and optimized*):  

- C++ classes **mirror** their Python counterparts, maintaining identical interfaces and data field names.    
- Support for **zero-copy data exchange** (e.g., descriptors) and **safe memory ownership** across the Python/C++ boundary, wherever feasible.  

## Core Principles

The C++ core adopts a streamlined design philosophy:  

- **All core data resides in C++** — Python serves purely as an interface layer.  
- **Direct pybind11 exposure** — Python objects are lightweight views of underlying C++ objects.  
- **Automatic zero-copy** — pybind11 automatically shares NumPy array memory with C++ when possible.  
- **RAII-based ownership** — C++ smart pointers manage object lifetimes safely and efficiently.

# PYSLAM C++ Core Module [WIP]

This directory contains the C++ implementations of core SLAM classes with Python bindings via pybind11.

## Architecture and requirements

- The C++ classes have the same exact interface and data field names of the corresponding python classes
- There are no C++ setters and getters unless there are the corresponding python methods
- Enable **zero-copy** exchange of descriptors and **safe ownership** across Python/C++ (where possible)

The C++ core module follows a simplified approach where:
- **All core data lives in C++** - no Python wrappers holding data
- **Direct pybind11 exposure** - Python objects are thin interfaces to C++ objects
- **Automatic zero-copy** - pybind11 handles numpy array sharing automatically (where possible)
- **RAII ownership** - C++ smart pointers handle lifetime automatically (where possible)
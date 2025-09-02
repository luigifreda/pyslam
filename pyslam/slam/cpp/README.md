# PYSLAM C++ Core Module

This directory contains the C++ implementations of core SLAM classes with Python bindings via pybind11.

## Architecture and requirements

- The C++ classes have the same exact interface and data field names of the corresponding python classes. 
- There are no C++ setters and getters unless there are the corresponding python methods.
- Enable **zero-copy** exchange of descriptors and **safe ownership** across Python/C++.

The C++ core module follows a simplified approach where:
- **All core data lives in C++** - no Python wrappers holding data
- **Direct pybind11 exposure** - Python objects are thin interfaces to C++ objects
- **Automatic zero-copy** - pybind11 handles numpy array sharing automatically
- **RAII ownership** - C++ smart pointers handle lifetime automatically


## Approach with zero-copy and safe ownership 

pybind11 should handle zero-copy and safe ownership automatically when we design our C++ classes properly.

Key Benefits of This Simplified Approach:

1. Automatic Zero-Copy
pybind11 automatically handles zero-copy conversion between cv::Mat and numpy arrays
No manual memory management or ownership tracking needed
Zero-copy happens transparently when accessing data

2. Automatic Safe Ownership
C++ objects own their data through RAII
pybind11 handles Python object lifetime automatically
No complex ownership registries or manual cleanup needed

3. Much Simpler Code
~80% less code compared to the complex zero-copy approach
No memory managers, ownership registries, or validation frameworks
Just clean C++ classes with direct pybind11 exposure

4. Better Performance
Contiguous memory layout in C++ objects
Direct memory access without Python overhead
Efficient thread safety with C++ mutexes

5. Easier Maintenance
Single source of truth (C++ objects)
Clear ownership model
Simple Python interfaces that just delegate to C++

The Core Philosophy:
Instead of trying to manage complex ownership between Python and C++, we simply:
- Keep all data in C++ - C++ objects own their data
- Expose C++ objects directly via pybind11
- Let pybind11 handle the integration - it's designed for this
- Use thin Python wrappers - just interface methods, no data

This approach is much more in line with how successful libraries like OpenCV, NumPy, and PyTorch work. They keep the core data in C++ and let pybind11 handle the Python integration automatically.
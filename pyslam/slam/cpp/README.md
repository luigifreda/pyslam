# PYSLAM C++ Core Module

This directory contains the **C++ implementations** of the core SLAM classes, exposed to Python via **pybind11** bindings.  
The module is currently **under active development**.

## Architecture and Design Goals

The following design requirements guide the development of the C++ core  
(*note: the module is still being refined and optimized*):  

- C++ classes **mirror** their Python counterparts, maintaining identical interfaces and data field names.  
- **Setters and getters** are implemented in C++ **only if** they exist in the corresponding Python classes.  
- Support for **zero-copy data exchange** (e.g., descriptors) and **safe memory ownership** across the Python/C++ boundary, wherever feasible.  

## Core Principles

The C++ core adopts a streamlined design philosophy:  

- **All core data resides in C++** — Python serves purely as an interface layer.  
- **Direct pybind11 exposure** — Python objects are lightweight views of underlying C++ objects.  
- **Automatic zero-copy** — pybind11 automatically shares NumPy array memory with C++ when possible.  
- **RAII-based ownership** — C++ smart pointers manage object lifetimes safely and efficiently.

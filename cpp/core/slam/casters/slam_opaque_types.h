/*
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pybind11/pybind11.h>
// Note: stl.h must be included AFTER PYBIND11_MAKE_OPAQUE declarations
#include <pybind11/stl_bind.h>

#include <vector>

// Include smart_pointers.h to get the actual KeyFramePtr, MapPointPtr, FramePtr typedefs
// This is safe because these are just typedefs, not full class definitions
#include "smart_pointers.h"

namespace py = pybind11;

// ----------------------------------------------------------------------------
// PYBIND11_MAKE_OPAQUE declarations for SLAM containers
// ----------------------------------------------------------------------------
//
// These declarations mark STL containers as "opaque" for pybind11, preventing
// automatic conversion to Python objects. This provides several benefits:
//
// 1. Performance: Avoids copying large containers when passing between C++ and Python
// 2. Mutation: Python-side mutations are reflected in C++ (important for output parameters)
// 3. Identity: Maintains object identity between C++ and Python
//
// Usage pattern:
//   - Declare PYBIND11_MAKE_OPAQUE(Type) BEFORE including pybind11/stl.h
//   - Then use py::bind_vector<Type> or py::bind_map<Type> to expose it
//
// Reference: https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// ----------------------------------------------------------------------------

#define ENABLE_OPAQUE_TYPES 0
// NOTE: It seems enabling opaque breaks smooth interoperation between C++ and Python,
//      especially when dealing and operating with lists/arrays of objects.
//      At present, we keep this section disabled.

#if ENABLE_OPAQUE_TYPES

// CRITICAL: These must be declared BEFORE including pybind11/stl.h
// Most frequently used vector types in SLAM operations
PYBIND11_MAKE_OPAQUE(std::vector<pyslam::KeyFramePtr>);
PYBIND11_MAKE_OPAQUE(std::vector<pyslam::MapPointPtr>);
PYBIND11_MAKE_OPAQUE(std::vector<pyslam::FramePtr>);

// Integer vectors (commonly used for indices, matches, etc.)
// Note: Making this opaque can improve performance for large index arrays
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<bool>); // For masks and flags

// TODO?:
// Map containers used in loop closure and optimization
// Note: These require std::hash specialization for KeyFramePtr/MapPointPtr
// We need to add custom hash functions
// For now, we'll comment these out as they may need additional setup
//
// PYBIND11_MAKE_OPAQUE(std::unordered_map<pyslam::KeyFramePtr, pyslam::Sim3Pose>);
// PYBIND11_MAKE_OPAQUE(std::unordered_map<pyslam::KeyFramePtr, std::vector<pyslam::KeyFramePtr>>);

// Note: std::set and std::unordered_set with custom comparators (like KeyFrameIdSet)
// are more complex to bind and may require custom type casters.
// They are typically accessed through wrapper functions rather than direct binding.

#endif

// Include stl.h AFTER opaque declarations
#include <pybind11/stl.h>
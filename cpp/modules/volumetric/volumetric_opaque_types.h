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

#include <array>
#include <vector>

#include "bounding_boxes.h"
#include "voxel_grid_data.h"

namespace py = pybind11;

// ----------------------------------------------------------------------------
// PYBIND11_MAKE_OPAQUE declarations for Volumetric containers
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

#define ENABLE_OPAQUE_TYPES 1

#if ENABLE_OPAQUE_TYPES

// CRITICAL: These must be declared BEFORE including pybind11/stl.h

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<float, 3>>);

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<bool>);

PYBIND11_MAKE_OPAQUE(std::vector<volumetric::OrientedBoundingBox3D>);
PYBIND11_MAKE_OPAQUE(std::vector<volumetric::OrientedBoundingBox3D::Ptr>);

PYBIND11_MAKE_OPAQUE(std::vector<volumetric::BoundingBox3D>);
PYBIND11_MAKE_OPAQUE(std::vector<volumetric::BoundingBox3D::Ptr>);

PYBIND11_MAKE_OPAQUE(std::vector<volumetric::ObjectData>);
PYBIND11_MAKE_OPAQUE(std::vector<volumetric::ObjectData::Ptr>);

PYBIND11_MAKE_OPAQUE(std::vector<volumetric::ClassData>);
PYBIND11_MAKE_OPAQUE(std::vector<volumetric::ClassData::Ptr>);

#endif

// Include stl.h AFTER opaque declarations
#include <pybind11/stl.h>
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

// This header intentionally includes the opaque type declarations (and stl.h)
// so the binding helpers can be used safely in the module init.
#include "volumetric_opaque_types.h"

// the vectors are serialized as a single tuple element containing the underlying list of values.
template <typename T> inline py::tuple vector_to_state(const std::vector<T> &self) {
    py::list data;
    for (const auto &value : self) {
        data.append(value);
    }
    return py::make_tuple(data);
}

template <typename T>
inline std::vector<T> vector_from_state(py::tuple state, const char *type_name) {
    if (state.size() != 1) {
        throw py::value_error(std::string("Invalid state for ") + type_name);
    }
    std::vector<T> self;
    const auto data = state[0].cast<py::list>();
    self.reserve(data.size());
    for (const auto &item : data) {
        self.push_back(py::cast<T>(item));
    }
    return self;
}

inline void bind_volumetric_opaque_containers(py::module &m) {
    // ------------------------------------------------------------
    // Volumetric-specific opaque containers
    //
    // These containers are declared opaque (see volumetric_opaque_types.h)
    // to avoid copying when passing between C++ and Python, and to enable
    // mutations in Python to be reflected in C++. This provides significant
    // performance benefits for large point clouds.

    // Pickle support for the opaque vectors so they can round-trip through
    // multiprocessing.

    auto point_double_vector =
        py::bind_vector<std::vector<std::array<double, 3>>>(m, "PointDoubleVector");
    point_double_vector.def(py::pickle(
        [](const std::vector<std::array<double, 3>> &self) { return vector_to_state(self); },
        [](py::tuple state) {
            return vector_from_state<std::array<double, 3>>(state, "PointDoubleVector");
        }));

    auto color_float_vector =
        py::bind_vector<std::vector<std::array<float, 3>>>(m, "ColorFloatVector");
    color_float_vector.def(py::pickle(
        [](const std::vector<std::array<float, 3>> &self) { return vector_to_state(self); },
        [](py::tuple state) {
            return vector_from_state<std::array<float, 3>>(state, "ColorFloatVector");
        }));

    auto oriented_bounding_box_vector =
        py::bind_vector<std::vector<volumetric::OrientedBoundingBox3D>>(
            m, "OrientedBoundingBox3DVector");
    oriented_bounding_box_vector.def(py::pickle(
        [](const std::vector<volumetric::OrientedBoundingBox3D> &self) {
            return vector_to_state(self);
        },
        [](py::tuple state) {
            return vector_from_state<volumetric::OrientedBoundingBox3D>(
                state, "OrientedBoundingBox3DVector");
        }));

    auto oriented_bounding_box_ptr_vector =
        py::bind_vector<std::vector<volumetric::OrientedBoundingBox3D::Ptr>>(
            m, "OrientedBoundingBox3DPtrVector");
    oriented_bounding_box_ptr_vector.def(py::pickle(
        [](const std::vector<volumetric::OrientedBoundingBox3D::Ptr> &self) {
            return vector_to_state(self);
        },
        [](py::tuple state) {
            return vector_from_state<volumetric::OrientedBoundingBox3D::Ptr>(
                state, "OrientedBoundingBox3DPtrVector");
        }));

    auto bounding_box_vector =
        py::bind_vector<std::vector<volumetric::BoundingBox3D>>(m, "BoundingBox3DVector");
    bounding_box_vector.def(py::pickle(
        [](const std::vector<volumetric::BoundingBox3D> &self) { return vector_to_state(self); },
        [](py::tuple state) {
            return vector_from_state<volumetric::BoundingBox3D>(state, "BoundingBox3DVector");
        }));

    auto bounding_box_ptr_vector =
        py::bind_vector<std::vector<volumetric::BoundingBox3D::Ptr>>(m, "BoundingBox3DPtrVector");
    bounding_box_ptr_vector.def(py::pickle(
        [](const std::vector<volumetric::BoundingBox3D::Ptr> &self) {
            return vector_to_state(self);
        },
        [](py::tuple state) {
            return vector_from_state<volumetric::BoundingBox3D::Ptr>(state,
                                                                     "BoundingBox3DPtrVector");
        }));

    auto int_vector = py::bind_vector<std::vector<int>>(m, "IntVector");
    int_vector.def(
        py::pickle([](const std::vector<int> &self) { return vector_to_state(self); },
                   [](py::tuple state) { return vector_from_state<int>(state, "IntVector"); }));

    auto bool_vector = py::bind_vector<std::vector<bool>>(m, "BoolVector");
    bool_vector.def(
        py::pickle([](const std::vector<bool> &self) { return vector_to_state(self); },
                   [](py::tuple state) { return vector_from_state<bool>(state, "BoolVector"); }));

    auto object_data_ptr_vector =
        py::bind_vector<std::vector<volumetric::ObjectData::Ptr>>(m, "ObjectDataPtrVector");
    object_data_ptr_vector.def(py::pickle(
        [](const std::vector<volumetric::ObjectData::Ptr> &self) { return vector_to_state(self); },
        [](py::tuple state) {
            return vector_from_state<volumetric::ObjectData::Ptr>(state, "ObjectDataPtrVector");
        }));

    auto class_data_ptr_vector =
        py::bind_vector<std::vector<volumetric::ClassData::Ptr>>(m, "ClassDataPtrVector");
    class_data_ptr_vector.def(py::pickle(
        [](const std::vector<volumetric::ClassData::Ptr> &self) { return vector_to_state(self); },
        [](py::tuple state) {
            return vector_from_state<volumetric::ClassData::Ptr>(state, "ClassDataPtrVector");
        }));
}

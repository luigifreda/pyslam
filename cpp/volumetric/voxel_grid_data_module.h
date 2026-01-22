/**
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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "voxel_grid_data.h"

namespace py = pybind11;

template <typename Tpos, typename Tcolor>
void bind_voxel_grid_data_family(py::module &m, const std::string &suffix) {
    using VoxelGridDataType = volumetric::VoxelGridDataT<Tpos, Tcolor>;
    using ObjectDataType = volumetric::ObjectDataT<Tpos, Tcolor>;
    using ObjectDataGroupType = volumetric::ObjectDataGroupT<Tpos, Tcolor>;
    using ClassDataType = volumetric::ClassDataT<Tpos, Tcolor>;
    using ClassDataGroupType = volumetric::ClassDataGroupT<Tpos, Tcolor>;

    const std::string voxel_grid_data_name = std::string("VoxelGridData") + suffix;
    py::class_<VoxelGridDataType, std::shared_ptr<VoxelGridDataType>>(m,
                                                                      voxel_grid_data_name.c_str())
        .def(py::init<>())
        .def_readwrite("points", &VoxelGridDataType::points, "Returns the points")
        .def_readwrite("colors", &VoxelGridDataType::colors, "Returns the colors")
        .def_readwrite("object_ids", &VoxelGridDataType::object_ids, "Returns the object IDs")
        .def_readwrite("class_ids", &VoxelGridDataType::class_ids, "Returns the class IDs")
        .def_readwrite("confidences", &VoxelGridDataType::confidences, "Returns the confidences")
        .def(py::pickle(
            [](const VoxelGridDataType &self) {
                return py::make_tuple(self.points, self.colors, self.object_ids, self.class_ids,
                                      self.confidences);
            },
            [](py::tuple state) {
                if (state.size() != 5) {
                    throw py::value_error("Invalid state for VoxelGridData");
                }
                VoxelGridDataType self;
                self.points = state[0].cast<decltype(self.points)>();
                self.colors = state[1].cast<decltype(self.colors)>();
                self.object_ids = state[2].cast<decltype(self.object_ids)>();
                self.class_ids = state[3].cast<decltype(self.class_ids)>();
                self.confidences = state[4].cast<decltype(self.confidences)>();
                return self;
            }));

    const std::string object_data_name = std::string("ObjectData") + suffix;
    py::class_<ObjectDataType, std::shared_ptr<ObjectDataType>>(m, object_data_name.c_str())
        .def(py::init<>())
        .def_readwrite("points", &ObjectDataType::points, "Returns the points")
        .def_readwrite("colors", &ObjectDataType::colors, "Returns the colors")
        .def_readwrite("class_id", &ObjectDataType::class_id, "Returns the class ID")
        .def_readwrite("confidence_min", &ObjectDataType::confidence_min,
                       "Returns the minimum confidence")
        .def_readwrite("confidence_max", &ObjectDataType::confidence_max,
                       "Returns the maximum confidence")
        .def_readwrite("object_id", &ObjectDataType::object_id, "Returns the object ID")
        .def_readwrite("oriented_bounding_box", &ObjectDataType::oriented_bounding_box,
                       "Returns the oriented bounding box")
        .def(py::pickle(
            [](const ObjectDataType &self) {
                return py::make_tuple(self.points, self.colors, self.class_id, self.confidence_min,
                                      self.confidence_max, self.object_id,
                                      self.oriented_bounding_box);
            },
            [](py::tuple state) {
                if (state.size() != 7) {
                    throw py::value_error("Invalid state for ObjectData");
                }
                ObjectDataType self;
                self.points = state[0].cast<decltype(self.points)>();
                self.colors = state[1].cast<decltype(self.colors)>();
                self.class_id = state[2].cast<decltype(self.class_id)>();
                self.confidence_min = state[3].cast<decltype(self.confidence_min)>();
                self.confidence_max = state[4].cast<decltype(self.confidence_max)>();
                self.object_id = state[5].cast<decltype(self.object_id)>();
                self.oriented_bounding_box = state[6].cast<decltype(self.oriented_bounding_box)>();
                return self;
            }));

    const std::string object_data_group_name = std::string("ObjectDataGroup") + suffix;
    py::class_<ObjectDataGroupType, std::shared_ptr<ObjectDataGroupType>>(
        m, object_data_group_name.c_str())
        .def(py::init<>())
        .def_readwrite("object_vector", &ObjectDataGroupType::object_vector,
                       "Returns the object vector")
        .def_readwrite("class_ids", &ObjectDataGroupType::class_ids, "Returns the class IDs")
        .def_readwrite("object_ids", &ObjectDataGroupType::object_ids, "Returns the object IDs")
        .def(py::pickle(
            [](const ObjectDataGroupType &self) {
                return py::make_tuple(self.object_vector, self.class_ids, self.object_ids);
            },
            [](py::tuple state) {
                if (state.size() != 3) {
                    throw py::value_error("Invalid state for ObjectDataGroup");
                }
                ObjectDataGroupType self;
                self.object_vector = state[0].cast<decltype(self.object_vector)>();
                self.class_ids = state[1].cast<decltype(self.class_ids)>();
                self.object_ids = state[2].cast<decltype(self.object_ids)>();
                return self;
            }));

    const std::string class_data_name = std::string("ClassData") + suffix;
    py::class_<ClassDataType, std::shared_ptr<ClassDataType>>(m, class_data_name.c_str())
        .def(py::init<>())
        .def_readwrite("points", &ClassDataType::points, "Returns the points")
        .def_readwrite("colors", &ClassDataType::colors, "Returns the colors")
        .def_readwrite("class_id", &ClassDataType::class_id, "Returns the class ID")
        .def_readwrite("confidence_min", &ClassDataType::confidence_min,
                       "Returns the minimum confidence")
        .def_readwrite("confidence_max", &ClassDataType::confidence_max,
                       "Returns the maximum confidence")
        .def(py::pickle(
            [](const ClassDataType &self) {
                return py::make_tuple(self.points, self.colors, self.class_id, self.confidence_min,
                                      self.confidence_max);
            },
            [](py::tuple state) {
                if (state.size() != 5) {
                    throw py::value_error("Invalid state for ClassData");
                }
                ClassDataType self;
                self.points = state[0].cast<decltype(self.points)>();
                self.colors = state[1].cast<decltype(self.colors)>();
                self.class_id = state[2].cast<decltype(self.class_id)>();
                self.confidence_min = state[3].cast<decltype(self.confidence_min)>();
                self.confidence_max = state[4].cast<decltype(self.confidence_max)>();
                return self;
            }));

    const std::string class_data_group_name = std::string("ClassDataGroup") + suffix;
    py::class_<ClassDataGroupType, std::shared_ptr<ClassDataGroupType>>(
        m, class_data_group_name.c_str())
        .def(py::init<>())
        .def_readwrite("class_vector", &ClassDataGroupType::class_vector,
                       "Returns the class vector")
        .def_readwrite("class_ids", &ClassDataGroupType::class_ids, "Returns the class IDs")
        .def(py::pickle(
            [](const ClassDataGroupType &self) {
                return py::make_tuple(self.class_vector, self.class_ids);
            },
            [](py::tuple state) {
                if (state.size() != 2) {
                    throw py::value_error("Invalid state for ClassDataGroup");
                }
                ClassDataGroupType self;
                self.class_vector = state[0].cast<decltype(self.class_vector)>();
                self.class_ids = state[1].cast<decltype(self.class_ids)>();
                return self;
            }));
}

void bind_voxel_grid_data(py::module &m) {
    // We use float for colors since it is the most common type and it is easy
    // and convenient to handle.
    bind_voxel_grid_data_family<double, float>(m, "");

    // Provide D aliases without re-registering the same C++ types.
    m.attr("VoxelGridDataD") = m.attr("VoxelGridData");
    m.attr("ObjectDataD") = m.attr("ObjectData");
    m.attr("ObjectDataGroupD") = m.attr("ObjectDataGroup");
    m.attr("ClassDataD") = m.attr("ClassData");
    m.attr("ClassDataGroupD") = m.attr("ClassDataGroup");

    bind_voxel_grid_data_family<float, float>(m, "F");
} // bind_voxel_grid_data
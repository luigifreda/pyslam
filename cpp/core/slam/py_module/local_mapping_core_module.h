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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "local_mapping_core.h"
#include "utils/messages.h"

namespace py = pybind11;

void bind_local_mapping_core(py::module &m) {

    py::class_<pyslam::LocalMappingCore, std::shared_ptr<pyslam::LocalMappingCore>>(
        m, "LocalMappingCore")
        .def(py::init([](pyslam::Map *map, py::object sensor_type_obj) {
                 pyslam::SensorType sensor_type;
                 try {
                     const int sensor_type_value = sensor_type_obj.attr("value").cast<int>();
                     sensor_type = static_cast<pyslam::SensorType>(sensor_type_value);
                 } catch (const py::cast_error &) {
                     MSG_ERROR("TrackingCore::count_tracked_and_non_tracked_close_points: "
                               "sensor_type is not a SensorType object");
                     sensor_type = sensor_type_obj.cast<pyslam::SensorType>();
                 }
                 return std::make_shared<pyslam::LocalMappingCore>(map, sensor_type);
             }),
             py::arg("map"), py::arg("sensor_type"))
        .def_property(
            "kf_cur",
            [](pyslam::LocalMappingCore &self) -> pyslam::KeyFramePtr & {
                return self.get_kf_cur();
            },
            [](pyslam::LocalMappingCore &self, pyslam::KeyFramePtr value) {
                self.set_kf_cur(value);
            })
        .def_property(
            "kid_last_BA",
            [](pyslam::LocalMappingCore &self) -> int { return self.get_kid_last_BA(); },
            [](pyslam::LocalMappingCore &self, int value) { self.set_kid_last_BA(value); })
        .def("reset",
             [](pyslam::LocalMappingCore &self) {
                 py::gil_scoped_release gil_release;
                 self.reset();
             })
        .def("add_points", &pyslam::LocalMappingCore::add_points)
        .def("remove_points", &pyslam::LocalMappingCore::remove_points)
        .def("set_opt_abort_flag", &pyslam::LocalMappingCore::set_opt_abort_flag)
        .def("clear_recent_points", &pyslam::LocalMappingCore::clear_recent_points)
        .def("num_recent_points", &pyslam::LocalMappingCore::num_recent_points)
        .def("get_recently_added_points", &pyslam::LocalMappingCore::get_recently_added_points)
        .def("local_BA",
             [](pyslam::LocalMappingCore &self) {
                 py::gil_scoped_release gil_release;
                 return self.local_BA();
             })
        .def("large_window_BA",
             [](pyslam::LocalMappingCore &self) {
                 py::gil_scoped_release gil_release;
                 return self.large_window_BA();
             })
        .def("process_new_keyframe",
             [](pyslam::LocalMappingCore &self) {
                 py::gil_scoped_release gil_release;
                 self.process_new_keyframe();
             })
        .def("cull_map_points",
             [](pyslam::LocalMappingCore &self) {
                 py::gil_scoped_release gil_release;
                 return self.cull_map_points();
             })
        .def("cull_keyframes",
             [](pyslam::LocalMappingCore &self, bool use_fov_centers_based_kf_generation,
                float max_fov_centers_distance) {
                 py::gil_scoped_release gil_release;
                 return self.cull_keyframes(use_fov_centers_based_kf_generation,
                                            max_fov_centers_distance);
             })
        .def(
            "fuse_map_points",
            [](pyslam::LocalMappingCore &self, float descriptor_distance_sigma) {
                py::gil_scoped_release gil_release;
                return self.fuse_map_points(descriptor_distance_sigma);
            },
            py::arg("descriptor_distance_sigma"));
}
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

#include "tracking_utils.h"

namespace py = pybind11;

void bind_tracking_utils(py::module &m) {

    py::class_<pyslam::TrackingUtils, std::shared_ptr<pyslam::TrackingUtils>>(m, "TrackingUtils")
        .def_static(
            "propagate_map_point_matches",
            [](const pyslam::FramePtr &f_ref, pyslam::FramePtr &f_cur,
               const std::vector<int> &idxs_ref, const std::vector<int> &idxs_cur,
               py::object max_descriptor_distance_obj) {
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                return pyslam::TrackingUtils::propagate_map_point_matches(
                    f_ref, f_cur, idxs_ref, idxs_cur, max_descriptor_distance);
            },
            py::arg("f_ref"), py::arg("f_cur"), py::arg("idxs_ref"), py::arg("idxs_cur"),
            py::arg("max_descriptor_distance") = py::none())
        .def_static("create_vo_points", &pyslam::TrackingUtils::create_vo_points, py::arg("frame"),
                    py::arg("max_num_points") = pyslam::Parameters::kMaxNumVisualOdometryPoints,
                    py::arg("color") = pyslam::Vec3b(0, 0, 255))
        .def_static("create_and_add_stereo_map_points_on_new_kf",
                    &pyslam::TrackingUtils::create_and_add_stereo_map_points_on_new_kf,
                    py::arg("frame"), py::arg("kf"), py::arg("map"), py::arg("img"));
}
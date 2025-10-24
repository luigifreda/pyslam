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

#include "tracking_core.h"
#include "utils/messages.h"

namespace py = pybind11;

void bind_tracking_core(py::module &m) {

    py::class_<pyslam::TrackingCore, std::shared_ptr<pyslam::TrackingCore>>(m, "TrackingCore")

        .def_static(
            "estimate_pose_by_fitting_ess_mat",
            [](const pyslam::FramePtr &f_ref, pyslam::FramePtr &f_cur,
               const std::vector<int> &idxs_ref, const std::vector<int> &idxs_cur) {
                py::gil_scoped_release release;
                return pyslam::TrackingCore::estimate_pose_by_fitting_ess_mat(f_ref, f_cur,
                                                                              idxs_ref, idxs_cur);
            },
            py::arg("f_ref"), py::arg("f_cur"), py::arg("idxs_ref"), py::arg("idxs_cur"))
        .def_static(
            "find_homography_with_ransac",
            [](pyslam::FramePtr &f_cur, const pyslam::FramePtr &f_ref,
               const std::vector<int> &idxs_cur, const std::vector<int> &idxs_ref,
               float reproj_threshold = pyslam::Parameters::kRansacReprojThreshold,
               int min_num_inliers = pyslam::Parameters::kRansacMinNumInliers) {
                py::gil_scoped_release release;
                return pyslam::TrackingCore::find_homography_with_ransac(
                    f_cur, f_ref, idxs_cur, idxs_ref, reproj_threshold, min_num_inliers);
            },
            py::arg("f_cur"), py::arg("f_ref"), py::arg("idxs_cur"), py::arg("idxs_ref"),
            py::arg("reproj_threshold") = pyslam::Parameters::kRansacReprojThreshold,
            py::arg("min_num_inliers") = pyslam::Parameters::kRansacMinNumInliers)
        .def_static(
            "propagate_map_point_matches",
            [](const pyslam::FramePtr &f_ref, pyslam::FramePtr &f_cur,
               const std::vector<int> &idxs_ref, const std::vector<int> &idxs_cur,
               py::object max_descriptor_distance_obj) {
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                py::gil_scoped_release release;
                return pyslam::TrackingCore::propagate_map_point_matches(
                    f_ref, f_cur, idxs_ref, idxs_cur, max_descriptor_distance);
            },
            py::arg("f_ref"), py::arg("f_cur"), py::arg("idxs_ref"), py::arg("idxs_cur"),
            py::arg("max_descriptor_distance") = py::none())
        .def_static(
            "create_vo_points",
            [](pyslam::FramePtr &frame,
               int max_num_points = pyslam::Parameters::kMaxNumVisualOdometryPoints,
               const pyslam::Vec3b &color = pyslam::Vec3b(0, 0, 255)) {
                py::gil_scoped_release release;
                return pyslam::TrackingCore::create_vo_points(frame, max_num_points, color);
            },
            py::arg("frame"),
            py::arg("max_num_points") = pyslam::Parameters::kMaxNumVisualOdometryPoints,
            py::arg("color") = pyslam::Vec3b(0, 0, 255))
        .def_static(
            "create_and_add_stereo_map_points_on_new_kf",
            [](pyslam::FramePtr &frame, pyslam::KeyFramePtr &kf, pyslam::MapPtr &map,
               const cv::Mat &img) {
                py::gil_scoped_release release;
                return pyslam::TrackingCore::create_and_add_stereo_map_points_on_new_kf(frame, kf,
                                                                                        map, img);
            },
            py::arg("frame"), py::arg("kf"), py::arg("map"), py::arg("img"))
        .def_static(
            "count_tracked_and_non_tracked_close_points",
            [](const pyslam::FramePtr &f_cur, py::object sensor_type_obj) {
                pyslam::SensorType sensor_type;
                try {
                    const int sensor_type_value = sensor_type_obj.attr("value").cast<int>();
                    sensor_type = static_cast<pyslam::SensorType>(sensor_type_value);
                } catch (const py::cast_error &) {
                    MSG_ERROR("TrackingCore::count_tracked_and_non_tracked_close_points: "
                              "sensor_type is not a SensorType object");
                    sensor_type = sensor_type_obj.cast<pyslam::SensorType>();
                }
                py::gil_scoped_release release;
                return pyslam::TrackingCore::count_tracked_and_non_tracked_close_points(
                    f_cur, sensor_type);
            },
            py::arg("f_cur"), py::arg("sensor_type"));
}
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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geometry_matchers.h"

namespace py = pybind11;

void bind_geometry_matchers(py::module &m) {
    py::class_<pyslam::ProjectionMatcher, std::shared_ptr<pyslam::ProjectionMatcher>>(
        m, "ProjectionMatcher")
        .def_static(
            "search_frame_by_projection",
            [](const pyslam::FramePtr &f_ref, pyslam::FramePtr &f_cur,
               py::object max_reproj_distance_obj, py::object max_descriptor_distance_obj,
               py::object ratio_test_obj, py::object is_monocular_obj,
               py::object already_matched_ref_idxs_obj) {
                float max_reproj_distance = max_reproj_distance_obj.is_none()
                                                ? pyslam::Parameters::kMaxReprojectionDistanceFrame
                                                : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                float ratio_test = ratio_test_obj.is_none() ? pyslam::Parameters::kMatchRatioTestMap
                                                            : ratio_test_obj.cast<float>();
                bool is_monocular =
                    is_monocular_obj.is_none() ? true : is_monocular_obj.cast<bool>();
                std::vector<int> already_matched_ref_idxs =
                    already_matched_ref_idxs_obj.is_none()
                        ? std::vector<int>()
                        : already_matched_ref_idxs_obj.cast<std::vector<int>>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_frame_by_projection(
                    f_ref, f_cur, max_reproj_distance, max_descriptor_distance, ratio_test,
                    is_monocular, already_matched_ref_idxs);
            },
            py::arg("f_ref"), py::arg("f_cur"),
            py::arg("max_reproj_distance") = pyslam::Parameters::kMaxReprojectionDistanceFrame,
            py::arg("max_descriptor_distance") = py::none(),
            py::arg("ratio_test") = pyslam::Parameters::kMatchRatioTestMap,
            py::arg("is_monocular") = true,
            py::arg("already_matched_ref_idxs") = py::cast(std::vector<int>()))
        .def_static(
            "search_keyframe_by_projection",
            [](const pyslam::KeyFramePtr &kf_ref, pyslam::FramePtr &f_cur,
               float max_reproj_distance, py::object max_descriptor_distance_obj,
               py::object ratio_test_obj, py::object already_matched_ref_idxs_obj) {
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                float ratio_test = ratio_test_obj.is_none() ? pyslam::Parameters::kMatchRatioTestMap
                                                            : ratio_test_obj.cast<float>();
                std::vector<int> already_matched_ref_idxs =
                    already_matched_ref_idxs_obj.is_none()
                        ? std::vector<int>()
                        : already_matched_ref_idxs_obj.cast<std::vector<int>>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_keyframe_by_projection(
                    kf_ref, f_cur, max_reproj_distance, max_descriptor_distance, ratio_test,
                    already_matched_ref_idxs);
            },
            py::arg("kf_ref"), py::arg("f_cur"), py::arg("max_reproj_distance"),
            py::arg("max_descriptor_distance") = py::none(),
            py::arg("ratio_test") = pyslam::Parameters::kMatchRatioTestMap,
            py::arg("already_matched_ref_idxs") = py::cast(std::vector<int>()))
        .def_static(
            "search_map_by_projection",
            [](const std::vector<pyslam::MapPointPtr> &points, pyslam::FramePtr &f_cur,
               py::object max_reproj_distance_obj, py::object max_descriptor_distance_obj,
               py::object ratio_test_obj, py::object far_points_threshold_obj) {
                float max_reproj_distance = max_reproj_distance_obj.is_none()
                                                ? pyslam::Parameters::kMaxReprojectionDistanceMap
                                                : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                float ratio_test = ratio_test_obj.is_none() ? pyslam::Parameters::kMatchRatioTestMap
                                                            : ratio_test_obj.cast<float>();
                float far_points_threshold = far_points_threshold_obj.is_none()
                                                 ? std::numeric_limits<float>::infinity()
                                                 : far_points_threshold_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_map_by_projection(
                    points, f_cur, max_reproj_distance, max_descriptor_distance, ratio_test,
                    far_points_threshold);
            },
            py::arg("points"), py::arg("f_cur"),
            py::arg("max_reproj_distance") = pyslam::Parameters::kMaxReprojectionDistanceMap,
            py::arg("max_descriptor_distance") = py::none(),
            py::arg("ratio_test") = pyslam::Parameters::kMatchRatioTestMap,
            py::arg("far_points_threshold") = py::cast(std::numeric_limits<float>::infinity()))
        .def_static(
            "search_local_frames_by_projection",
            [](pyslam::Map *map, pyslam::FramePtr &f_cur, py::object local_window_size_obj,
               py::object max_descriptor_distance_obj) {
                int local_window_size = local_window_size_obj.is_none()
                                            ? pyslam::Parameters::kLocalBAWindowSize
                                            : local_window_size_obj.cast<int>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_local_frames_by_projection(
                    map, f_cur, local_window_size, max_descriptor_distance);
            },
            py::arg("map"), py::arg("f_cur"),
            py::arg("local_window_size") = pyslam::Parameters::kLocalBAWindowSize,
            py::arg("max_descriptor_distance") = py::none())
        .def_static(
            "search_all_map_by_projection",
            [](pyslam::Map *map, pyslam::FramePtr &f_cur, py::object max_descriptor_distance_obj) {
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_all_map_by_projection(
                    map, f_cur, max_descriptor_distance);
            },
            py::arg("map"), py::arg("f_cur"), py::arg("max_descriptor_distance") = py::none())
        .def_static(
            "search_more_map_points_by_projection",
            [](const std::vector<pyslam::MapPointPtr> &points, pyslam::FramePtr &f_cur,
               const pyslam::Sim3Pose &Scw, std::vector<pyslam::MapPointPtr> &f_cur_matched_points,
               py::object f_cur_matched_points_idxs_obj, py::object max_reproj_distance_obj,
               py::object max_descriptor_distance_obj, py::object print_fun_obj) {
                std::vector<int> f_cur_matched_points_idxs =
                    f_cur_matched_points_idxs_obj.is_none()
                        ? std::vector<int>()
                        : f_cur_matched_points_idxs_obj.cast<std::vector<int>>();
                float max_reproj_distance = max_reproj_distance_obj.is_none()
                                                ? pyslam::Parameters::kMaxReprojectionDistanceMap
                                                : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_more_map_points_by_projection(
                    points, f_cur, Scw, f_cur_matched_points, f_cur_matched_points_idxs,
                    max_reproj_distance, max_descriptor_distance);
            },
            py::arg("points"), py::arg("f_cur"), py::arg("Scw"), py::arg("f_cur_matched_points"),
            py::arg("f_cur_matched_points_idxs") = py::cast(std::vector<int>()),
            py::arg("max_reproj_distance") = pyslam::Parameters::kMaxReprojectionDistanceMap,
            py::arg("max_descriptor_distance") = py::none(), py::arg("print_fun") = py::none())
        .def_static(
            "search_and_fuse",
            [](const std::vector<pyslam::MapPointPtr> &points, pyslam::KeyFramePtr &keyframe,
               py::object max_reproj_distance_obj, py::object max_descriptor_distance_obj,
               py::object ratio_test_obj) {
                float max_reproj_distance = max_reproj_distance_obj.is_none()
                                                ? pyslam::Parameters::kMaxReprojectionDistanceFuse
                                                : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                float ratio_test = ratio_test_obj.is_none() ? pyslam::Parameters::kMatchRatioTestMap
                                                            : ratio_test_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_and_fuse(
                    points, keyframe, max_reproj_distance, max_descriptor_distance, ratio_test);
            },
            py::arg("points"), py::arg("keyframe"),
            py::arg("max_reproj_distance") = pyslam::Parameters::kMaxReprojectionDistanceFuse,
            py::arg("max_descriptor_distance") = py::none(),
            py::arg("ratio_test") = pyslam::Parameters::kMatchRatioTestMap)
        .def_static(
            "search_and_fuse_for_loop_correction",
            [](const pyslam::KeyFramePtr &keyframe, const pyslam::Sim3Pose &Scw,
               const std::vector<pyslam::MapPointPtr> &points,
               std::vector<pyslam::MapPointPtr> &replace_points, py::object max_reproj_distance_obj,
               py::object max_descriptor_distance_obj) {
                float max_reproj_distance =
                    max_reproj_distance_obj.is_none()
                        ? pyslam::Parameters::kLoopClosingMaxReprojectionDistanceFuse
                        : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_and_fuse_for_loop_correction(
                    keyframe, Scw, points, replace_points, max_reproj_distance,
                    max_descriptor_distance);
            },
            py::arg("keyframe"), py::arg("Scw"), py::arg("points"), py::arg("replace_points"),
            py::arg("max_reproj_distance") =
                pyslam::Parameters::kLoopClosingMaxReprojectionDistanceFuse,
            py::arg("max_descriptor_distance") = py::none())
        .def_static(
            "search_by_sim3",
            [](const pyslam::KeyFramePtr &kf1, const pyslam::KeyFramePtr &kf2,
               const std::vector<int> &idxs1, const std::vector<int> &idxs2, float s12,
               const Eigen::Matrix3d &R12, const Eigen::Vector3d &t12,
               py::object max_reproj_distance_obj, py::object max_descriptor_distance_obj,
               py::object print_fun_obj) {
                float max_reproj_distance = max_reproj_distance_obj.is_none()
                                                ? pyslam::Parameters::kMaxReprojectionDistanceSim3
                                                : max_reproj_distance_obj.cast<float>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();

                py::gil_scoped_release release;
                return pyslam::ProjectionMatcher::search_by_sim3(kf1, kf2, idxs1, idxs2, s12, R12,
                                                                 t12, max_reproj_distance,
                                                                 max_descriptor_distance);
            },
            py::arg("kf1"), py::arg("kf2"), py::arg("idxs1"), py::arg("idxs2"), py::arg("s12"),
            py::arg("R12"), py::arg("t12"),
            py::arg("max_reproj_distance") = pyslam::Parameters::kMaxReprojectionDistanceSim3,
            py::arg("max_descriptor_distance") = py::none(), py::arg("print_fun") = py::none());

    py::class_<pyslam::EpipolarMatcher, std::shared_ptr<pyslam::EpipolarMatcher>>(m,
                                                                                  "EpipolarMatcher")
        .def_static(
            "search_frame_for_triangulation",
            [](const pyslam::KeyFramePtr &kf1, const pyslam::KeyFramePtr &kf2, py::object idxs1_obj,
               py::object idxs2_obj, py::object max_descriptor_distance_obj,
               py::object is_monocular_obj) {
                std::vector<int> idxs1 =
                    idxs1_obj.is_none() ? std::vector<int>() : idxs1_obj.cast<std::vector<int>>();
                std::vector<int> idxs2 =
                    idxs2_obj.is_none() ? std::vector<int>() : idxs2_obj.cast<std::vector<int>>();
                float max_descriptor_distance = max_descriptor_distance_obj.is_none()
                                                    ? -1.0f
                                                    : max_descriptor_distance_obj.cast<float>();
                bool is_monocular =
                    is_monocular_obj.is_none() ? true : is_monocular_obj.cast<bool>();

                py::gil_scoped_release release;
                return pyslam::EpipolarMatcher::search_frame_for_triangulation(
                    kf1, kf2, idxs1, idxs2, max_descriptor_distance, is_monocular);
            },
            py::arg("kf1"), py::arg("kf2"), py::arg("idxs1") = py::cast(std::vector<int>()),
            py::arg("idxs2") = py::cast(std::vector<int>()),
            py::arg("max_descriptor_distance") = py::none(), py::arg("is_monocular") = true);
}

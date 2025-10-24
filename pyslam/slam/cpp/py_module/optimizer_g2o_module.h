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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "optimizer_g2o_bind_helpers.h"

namespace py = pybind11;

void bind_optimizer_g2o(py::module &m) {

    // Bind result structures (keep for backward compatibility)
    py::class_<pyslam::BundleAdjustmentResult, std::shared_ptr<pyslam::BundleAdjustmentResult>>(
        m, "BundleAdjustmentResult")
        .def(py::init<>())
        .def_readwrite("mean_squared_error", &pyslam::BundleAdjustmentResult::mean_squared_error)
        .def_readwrite("keyframe_updates", &pyslam::BundleAdjustmentResult::keyframe_updates)
        .def_readwrite("point_updates", &pyslam::BundleAdjustmentResult::point_updates);

    py::class_<pyslam::PoseOptimizationResult>(m, "PoseOptimizationResult")
        .def(py::init<>())
        .def_readwrite("mean_squared_error", &pyslam::PoseOptimizationResult::mean_squared_error)
        .def_readwrite("is_ok", &pyslam::PoseOptimizationResult::is_ok)
        .def_readwrite("num_valid_points", &pyslam::PoseOptimizationResult::num_valid_points);

    py::class_<pyslam::Sim3OptimizationResult, std::shared_ptr<pyslam::Sim3OptimizationResult>>(
        m, "Sim3OptimizationResult")
        .def(py::init<>())
        .def(py::init<int, Eigen::Matrix3d, Eigen::Vector3d, double, double>(),
             py::arg("num_inliers"), py::arg("R"), py::arg("t"), py::arg("scale"),
             py::arg("delta_error"))
        .def_readwrite("num_inliers", &pyslam::Sim3OptimizationResult::num_inliers)
        .def_readwrite("R", &pyslam::Sim3OptimizationResult::R)
        .def_readwrite("t", &pyslam::Sim3OptimizationResult::t)
        .def_readwrite("scale", &pyslam::Sim3OptimizationResult::scale)
        .def_readwrite("delta_error", &pyslam::Sim3OptimizationResult::delta_error);

    // Bind main optimizer class with wrapper functions that match Python signatures
    py::class_<pyslam::OptimizerG2o>(m, "OptimizerG2o")
        .def_static("bundle_adjustment", &pyslam::bundle_adjustment_wrapper, py::arg("keyframes"),
                    py::arg("points"), py::arg("local_window_size") = std::nullopt,
                    py::arg("fixed_points") = false, py::arg("rounds") = 10,
                    py::arg("loop_kf_id") = 0, py::arg("use_robust_kernel") = false,
                    py::arg("abort_flag") = py::none(), py::arg("mp_abort_flag") = py::none(),
                    py::arg("result_dict") = py::none(), py::arg("verbose") = false,
                    py::arg("print") = py::none()) // print inot used here

        .def_static("global_bundle_adjustment", &pyslam::global_bundle_adjustment_wrapper,
                    py::arg("keyframes"), py::arg("points"), py::arg("rounds") = 10,
                    py::arg("loop_kf_id") = 0, py::arg("use_robust_kernel") = false,
                    py::arg("abort_flag") = py::none(), py::arg("mp_abort_flag") = py::none(),
                    py::arg("result_dict") = py::none(), py::arg("verbose") = false,
                    py::arg("print") = py::none()) // print inot used here

        .def_static("global_bundle_adjustment_map", &pyslam::global_bundle_adjustment_map_wrapper,
                    py::arg("map"), py::arg("rounds") = 10, py::arg("loop_kf_id") = 0,
                    py::arg("use_robust_kernel") = false, py::arg("abort_flag") = py::none(),
                    py::arg("mp_abort_flag") = py::none(), py::arg("result_dict") = py::none(),
                    py::arg("verbose") = false,
                    py::arg("print") = py::none()) // print inot used here

        .def_static("pose_optimization", &pyslam::pose_optimization_wrapper, py::arg("frame"),
                    py::arg("verbose") = false, py::arg("rounds") = 10)

        .def_static("local_bundle_adjustment", &pyslam::local_bundle_adjustment_wrapper,
                    py::arg("keyframes"), py::arg("points"),
                    py::arg("keyframes_ref") = std::vector<pyslam::KeyFrame *>{},
                    py::arg("fixed_points") = false, py::arg("verbose") = false,
                    py::arg("rounds") = 10, py::arg("abort_flag") = py::none(),
                    py::arg("map_lock") = py::none())

        .def_static("optimize_sim3", &pyslam::optimize_sim3_wrapper, py::arg("kf1"), py::arg("kf2"),
                    py::arg("map_points1"), py::arg("map_point_matches12"), py::arg("R12"),
                    py::arg("t12"), py::arg("s12"), py::arg("th2"), py::arg("fix_scale"),
                    py::arg("verbose") = false)

        .def_static("optimize_essential_graph", &pyslam::optimize_essential_graph_wrapper,
                    py::arg("map_object"), py::arg("loop_keyframe"), py::arg("current_keyframe"),
                    py::arg("non_corrected_sim3_map"), py::arg("corrected_sim3_map"),
                    py::arg("loop_connections"), py::arg("fix_scale"),
                    py::arg("print_fun") = py::none(), py::arg("verbose") = false);
}
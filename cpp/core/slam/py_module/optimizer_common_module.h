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

#include "optimizer_common.h"

namespace py = pybind11;

void bind_optimizer_common(py::module &m) {

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
}
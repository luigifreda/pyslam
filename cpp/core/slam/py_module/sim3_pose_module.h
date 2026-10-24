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

#include "sim3_pose.h"

namespace py = pybind11;

void bind_sim3_pose(py::module &m) {

    py::class_<pyslam::Sim3Pose, std::shared_ptr<pyslam::Sim3Pose>>(m, "Sim3Pose")
        .def(py::init<>())
        .def(py::init<const Eigen::Matrix4d &>())
        .def(py::init<const Eigen::Matrix3d &, const Eigen::Vector3d &, double>())
        .def_property(
            "R", [](pyslam::Sim3Pose &self) -> Eigen::Matrix3d & { return self.R(); },
            [](pyslam::Sim3Pose &self, const Eigen::Matrix3d &value) { self.R() = value; })
        .def_property(
            "t", [](pyslam::Sim3Pose &self) -> Eigen::Vector3d & { return self.t(); },
            [](pyslam::Sim3Pose &self, const Eigen::Vector3d &value) { self.t() = value; })
        .def_property(
            "s", [](pyslam::Sim3Pose &self) -> double & { return self.s(); },
            [](pyslam::Sim3Pose &self, const double &value) { self.s() = value; })
        .def("from_matrix", &pyslam::Sim3Pose::from_matrix)
        .def("from_se3_matrix", &pyslam::Sim3Pose::from_se3_matrix)
        .def("matrix", &pyslam::Sim3Pose::matrix)
        .def("inverse", &pyslam::Sim3Pose::inverse)
        .def("inverse_matrix", &pyslam::Sim3Pose::inverse_matrix)
        .def("to_se3_matrix", &pyslam::Sim3Pose::to_se3_matrix)
        .def("copy", &pyslam::Sim3Pose::copy)
        .def("map", &pyslam::Sim3Pose::map)
        .def("map_points", &pyslam::Sim3Pose::map_points)
        .def("__matmul__", [](const pyslam::Sim3Pose &self,
                              const pyslam::Sim3Pose &other) { return self * other; })
        .def("__matmul__", [](const pyslam::Sim3Pose &self,
                              const Eigen::Matrix4d &other) { return self * other; })
        .def("__str__", &pyslam::Sim3Pose::to_string);

} // bind_sim3_pose

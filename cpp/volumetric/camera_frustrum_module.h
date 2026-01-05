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

#include "bounding_boxes.h"
#include "camera_frustrum.h"

namespace py = pybind11;

void bind_camera_frustrum(py::module &m) {

    // ================================================================
    // Camera Frustrum
    // ================================================================

    // ImagePoint class
    py::class_<volumetric::ImagePoint>(m, "ImagePoint")
        .def(py::init<>(), "Default constructor")
        .def(py::init<int, int, float>(), "Constructor with u, v, depth", py::arg("u"),
             py::arg("v"), py::arg("depth"))
        .def_readwrite("u", &volumetric::ImagePoint::u)
        .def_readwrite("v", &volumetric::ImagePoint::v)
        .def_readwrite("depth", &volumetric::ImagePoint::depth);

    // CameraFrustrum class
    py::class_<volumetric::CameraFrustrum, std::shared_ptr<volumetric::CameraFrustrum>>(
        m, "CameraFrustrum")
        .def(py::init<const Eigen::Matrix3d &, const int, const int, const Eigen::Matrix4d &,
                      const double, const double>(),
             py::arg("K"), py::arg("width"), py::arg("height"), py::arg("T_cw"),
             py::arg("depth_max"), py::arg("depth_min"))
        .def(py::init<const float, const float, const float, const float, const int, const int,
                      const Eigen::Quaterniond &, const Eigen::Vector3d &, const double,
                      const double>(),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("width"),
             py::arg("height"), py::arg("orientation"), py::arg("translation"),
             py::arg("depth_max"), py::arg("depth_min"))
        .def(py::init<const float, const float, const float, const float, const int, const int,
                      const Eigen::Matrix4d &, const double, const double>(),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("width"),
             py::arg("height"), py::arg("T_cw"), py::arg("depth_max"), py::arg("depth_min"))
        .def("set_width", &volumetric::CameraFrustrum::set_width, py::arg("width"))
        .def("set_height", &volumetric::CameraFrustrum::set_height, py::arg("height"))
        .def("set_depth_max", &volumetric::CameraFrustrum::set_depth_max, py::arg("depth_max"))
        .def("set_depth_min", &volumetric::CameraFrustrum::set_depth_min, py::arg("depth_min"))
        .def(
            "set_intrinsics",
            [](volumetric::CameraFrustrum &self, const Eigen::Matrix3d &K) {
                self.set_intrinsics(K);
            },
            py::arg("K"))
        .def(
            "set_intrinsics",
            [](volumetric::CameraFrustrum &self, float fx, float fy, float cx, float cy) {
                self.set_intrinsics(fx, fy, cx, cy);
            },
            py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
        .def(
            "set_T_cw",
            [](volumetric::CameraFrustrum &self, const Eigen::Matrix4d &T_cw) {
                self.set_T_cw(T_cw);
            },
            py::arg("T_cw"))
        .def(
            "set_T_cw",
            [](volumetric::CameraFrustrum &self, const Eigen::Quaterniond &orientation,
               const Eigen::Vector3d &translation) { self.set_T_cw(orientation, translation); },
            py::arg("orientation"), py::arg("translation"))
        .def("get_width", &volumetric::CameraFrustrum::get_width)
        .def("get_height", &volumetric::CameraFrustrum::get_height)
        .def("get_fx", &volumetric::CameraFrustrum::get_fx)
        .def("get_fy", &volumetric::CameraFrustrum::get_fy)
        .def("get_cx", &volumetric::CameraFrustrum::get_cx)
        .def("get_cy", &volumetric::CameraFrustrum::get_cy)
        .def("get_K", &volumetric::CameraFrustrum::get_K)
        .def("get_T_cw", &volumetric::CameraFrustrum::get_T_cw)
        .def("get_R_cw", &volumetric::CameraFrustrum::get_R_cw)
        .def(
            "get_orientation_cw",
            [](const volumetric::CameraFrustrum &self) {
                return self.get_orientation_cw(); // Return Eigen::Quaterniond (now exposed as
                                                  // Python class)
            },
            "Get camera orientation as Quaterniond")
        .def("get_t_cw", &volumetric::CameraFrustrum::get_t_cw)
        .def("get_obb", &volumetric::CameraFrustrum::get_obb)
        .def("get_corners", &volumetric::CameraFrustrum::get_corners)
        .def(
            "is_in_bbox",
            [](const volumetric::CameraFrustrum &self, const Eigen::Vector3d &point_w) {
                return self.is_in_bbox(point_w);
            },
            py::arg("point_w"))
        .def(
            "is_in_obb",
            [](const volumetric::CameraFrustrum &self, const Eigen::Vector3d &point_w) {
                return self.is_in_obb(point_w);
            },
            py::arg("point_w"))
        .def(
            "contains",
            [](const volumetric::CameraFrustrum &self, const Eigen::Vector3d &point_w) {
                return self.contains(point_w);
            },
            py::arg("point_w"))
        .def("is_cache_valid", &volumetric::CameraFrustrum::is_cache_valid);
}
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

    // BoundingBox3D class
    py::class_<BoundingBox3D, std::shared_ptr<BoundingBox3D>>(m, "BoundingBox3D")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(), py::arg("min_point"),
             py::arg("max_point"))
        .def_readwrite("min_x", &BoundingBox3D::min_x)
        .def_readwrite("min_y", &BoundingBox3D::min_y)
        .def_readwrite("min_z", &BoundingBox3D::min_z)
        .def_readwrite("max_x", &BoundingBox3D::max_x)
        .def_readwrite("max_y", &BoundingBox3D::max_y)
        .def_readwrite("max_z", &BoundingBox3D::max_z)
        .def("get_min_point", &BoundingBox3D::get_min_point)
        .def("get_max_point", &BoundingBox3D::get_max_point)
        .def("get_center", &BoundingBox3D::get_center)
        .def("get_size", &BoundingBox3D::get_size)
        .def("get_volume", &BoundingBox3D::get_volume)
        .def("get_surface_area", &BoundingBox3D::get_surface_area)
        .def("get_diagonal_length", &BoundingBox3D::get_diagonal_length)
        .def("contains", [](const BoundingBox3D &self,
                            const Eigen::Vector3d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const BoundingBox3D &self, const std::vector<Eigen::Vector3d> &points) {
                 return self.contains(points);
             })
        .def("intersects", [](const BoundingBox3D &self, const BoundingBox3D &other) {
            return self.intersects(other);
        });

    // OrientedBoundingBox3D class
    py::class_<OrientedBoundingBox3D, std::shared_ptr<OrientedBoundingBox3D>>(
        m, "OrientedBoundingBox3D")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Quaterniond &,
                      const Eigen::Vector3d &>(),
             py::arg("center"), py::arg("orientation"), py::arg("size"))
        .def_readwrite("center", &OrientedBoundingBox3D::center)
        .def_readwrite("orientation", &OrientedBoundingBox3D::orientation)
        .def_readwrite("size", &OrientedBoundingBox3D::size)
        .def("get_volume", &OrientedBoundingBox3D::get_volume)
        .def("get_surface_area", &OrientedBoundingBox3D::get_surface_area)
        .def("get_diagonal_length", &OrientedBoundingBox3D::get_diagonal_length)
        .def("get_corners", &OrientedBoundingBox3D::get_corners)
        .def("contains", [](const OrientedBoundingBox3D &self,
                            const Eigen::Vector3d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const OrientedBoundingBox3D &self, const std::vector<Eigen::Vector3d> &points_w) {
                 return self.contains(points_w);
             })
        .def("intersects",
             [](const OrientedBoundingBox3D &self, const OrientedBoundingBox3D &other) {
                 return self.intersects(other);
             })
        .def("intersects", [](const OrientedBoundingBox3D &self, const BoundingBox3D &other) {
            return self.intersects(other);
        });

    // BoundingBox2D class
    py::class_<BoundingBox2D, std::shared_ptr<BoundingBox2D>>(m, "BoundingBox2D")
        .def(py::init<const Eigen::Vector2d &, const Eigen::Vector2d &>(), py::arg("min_point"),
             py::arg("max_point"))
        .def_readwrite("min_x", &BoundingBox2D::min_x)
        .def_readwrite("min_y", &BoundingBox2D::min_y)
        .def_readwrite("max_x", &BoundingBox2D::max_x)
        .def_readwrite("max_y", &BoundingBox2D::max_y)
        .def("get_min_point", &BoundingBox2D::get_min_point)
        .def("get_max_point", &BoundingBox2D::get_max_point)
        .def("get_center", &BoundingBox2D::get_center)
        .def("get_size", &BoundingBox2D::get_size)
        .def("get_area", &BoundingBox2D::get_area)
        .def("get_perimeter", &BoundingBox2D::get_perimeter)
        .def("get_diagonal_length", &BoundingBox2D::get_diagonal_length)
        .def("contains", [](const BoundingBox2D &self,
                            const Eigen::Vector2d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const BoundingBox2D &self, const std::vector<Eigen::Vector2d> &points_w) {
                 return self.contains(points_w);
             })
        .def("intersects", [](const BoundingBox2D &self, const BoundingBox2D &other) {
            return self.intersects(other);
        });

    // OrientedBoundingBox2D class
    py::class_<OrientedBoundingBox2D, std::shared_ptr<OrientedBoundingBox2D>>(
        m, "OrientedBoundingBox2D")
        .def(py::init<const Eigen::Vector2d &, const double, const Eigen::Vector2d &>(),
             py::arg("center"), py::arg("angle_rad"), py::arg("size"))
        .def_readwrite("center", &OrientedBoundingBox2D::center)
        .def_readwrite("angle_rad", &OrientedBoundingBox2D::angle_rad)
        .def_readwrite("size", &OrientedBoundingBox2D::size)
        .def("get_volume", &OrientedBoundingBox2D::get_volume)
        .def("get_area", &OrientedBoundingBox2D::get_area)
        .def("get_perimeter", &OrientedBoundingBox2D::get_perimeter)
        .def("get_diagonal_length", &OrientedBoundingBox2D::get_diagonal_length)
        .def("get_corners", &OrientedBoundingBox2D::get_corners)
        .def("contains", [](const OrientedBoundingBox2D &self,
                            const Eigen::Vector2d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const OrientedBoundingBox2D &self, const std::vector<Eigen::Vector2d> &points_w) {
                 return self.contains(points_w);
             })
        .def("intersects",
             [](const OrientedBoundingBox2D &self, const OrientedBoundingBox2D &other) {
                 return self.intersects(other);
             })
        .def("intersects", [](const OrientedBoundingBox2D &self, const BoundingBox2D &other) {
            return self.intersects(other);
        });

    // CameraFrustrum class
    py::class_<CameraFrustrum, std::shared_ptr<CameraFrustrum>>(m, "CameraFrustrum")
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
        .def("set_width", &CameraFrustrum::set_width, py::arg("width"))
        .def("set_height", &CameraFrustrum::set_height, py::arg("height"))
        .def("set_depth_max", &CameraFrustrum::set_depth_max, py::arg("depth_max"))
        .def("set_depth_min", &CameraFrustrum::set_depth_min, py::arg("depth_min"))
        .def(
            "set_intrinsics",
            [](CameraFrustrum &self, const Eigen::Matrix3d &K) { self.set_intrinsics(K); },
            py::arg("K"))
        .def(
            "set_intrinsics",
            [](CameraFrustrum &self, float fx, float fy, float cx, float cy) {
                self.set_intrinsics(fx, fy, cx, cy);
            },
            py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
        .def(
            "set_T_cw",
            [](CameraFrustrum &self, const Eigen::Matrix4d &T_cw) { self.set_T_cw(T_cw); },
            py::arg("T_cw"))
        .def(
            "set_T_cw",
            [](CameraFrustrum &self, const Eigen::Quaterniond &orientation,
               const Eigen::Vector3d &translation) { self.set_T_cw(orientation, translation); },
            py::arg("orientation"), py::arg("translation"))
        .def("get_K", &CameraFrustrum::get_K)
        .def("get_T_cw", &CameraFrustrum::get_T_cw)
        .def("get_R_cw", &CameraFrustrum::get_R_cw)
        .def("get_orientation_cw", &CameraFrustrum::get_orientation_cw)
        .def("get_t_cw", &CameraFrustrum::get_t_cw)
        .def("get_obb", &CameraFrustrum::get_obb)
        .def("get_corners", &CameraFrustrum::get_corners)
        .def("is_in_obb", &CameraFrustrum::is_in_obb, py::arg("point_w"))
        .def("is_in_frustum", &CameraFrustrum::is_in_frustum, py::arg("point_w"))
        .def("is_cache_valid", &CameraFrustrum::is_cache_valid);
}
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
    // Eigen::Quaterniond wrapper
    // ================================================================
    py::class_<Eigen::Quaterniond>(m, "Quaterniond")
        .def(py::init<>(), "Default constructor (identity quaternion)")
        .def(py::init<double, double, double, double>(), "Constructor from w, x, y, z components",
             py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(py::init([](const py::array_t<double> &arr) {
                 auto r = arr.unchecked<1>();
                 if (arr.size() != 4) {
                     throw std::runtime_error(
                         "Quaternion array must have exactly 4 elements [w, x, y, z]");
                 }
                 return Eigen::Quaterniond(r(0), r(1), r(2), r(3)); // w, x, y, z
             }),
             "Constructor from numpy array [w, x, y, z]", py::arg("coeffs"))
        .def(
            "w", [](const Eigen::Quaterniond &q) { return q.w(); }, "Get w component")
        .def(
            "x", [](const Eigen::Quaterniond &q) { return q.x(); }, "Get x component")
        .def(
            "y", [](const Eigen::Quaterniond &q) { return q.y(); }, "Get y component")
        .def(
            "z", [](const Eigen::Quaterniond &q) { return q.z(); }, "Get z component")
        .def(
            "coeffs",
            [](const Eigen::Quaterniond &q) {
                // Return as numpy array [w, x, y, z]
                py::array_t<double> result(4);
                auto r = result.mutable_unchecked<1>();
                r(0) = q.w();
                r(1) = q.x();
                r(2) = q.y();
                r(3) = q.z();
                return result;
            },
            "Get quaternion coefficients as numpy array [w, x, y, z]")
        .def("normalized", &Eigen::Quaterniond::normalized, "Get normalized quaternion")
        .def("normalize", &Eigen::Quaterniond::normalize, "Normalize this quaternion in-place")
        .def("conjugate", &Eigen::Quaterniond::conjugate, "Get conjugate quaternion")
        .def("inverse", &Eigen::Quaterniond::inverse, "Get inverse quaternion")
        .def("toRotationMatrix", &Eigen::Quaterniond::toRotationMatrix,
             "Convert to rotation matrix")
        .def("__repr__",
             [](const Eigen::Quaterniond &q) {
                 return "Quaterniond(w=" + std::to_string(q.w()) + ", x=" + std::to_string(q.x()) +
                        ", y=" + std::to_string(q.y()) + ", z=" + std::to_string(q.z()) + ")";
             })
        .def("__str__", [](const Eigen::Quaterniond &q) {
            return "Quaterniond(w=" + std::to_string(q.w()) + ", x=" + std::to_string(q.x()) +
                   ", y=" + std::to_string(q.y()) + ", z=" + std::to_string(q.z()) + ")";
        });

    // ================================================================
    // Bounding Boxes
    // ================================================================

    // BoundingBox3D class
    py::class_<volumetric::BoundingBox3D, std::shared_ptr<volumetric::BoundingBox3D>>(
        m, "BoundingBox3D")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(), py::arg("min_point"),
             py::arg("max_point"))
        .def_readwrite("min_x", &volumetric::BoundingBox3D::min_x)
        .def_readwrite("min_y", &volumetric::BoundingBox3D::min_y)
        .def_readwrite("min_z", &volumetric::BoundingBox3D::min_z)
        .def_readwrite("max_x", &volumetric::BoundingBox3D::max_x)
        .def_readwrite("max_y", &volumetric::BoundingBox3D::max_y)
        .def_readwrite("max_z", &volumetric::BoundingBox3D::max_z)
        .def("get_min_point", &volumetric::BoundingBox3D::get_min_point)
        .def("get_max_point", &volumetric::BoundingBox3D::get_max_point)
        .def("get_center", &volumetric::BoundingBox3D::get_center)
        .def("get_size", &volumetric::BoundingBox3D::get_size)
        .def("get_volume", &volumetric::BoundingBox3D::get_volume)
        .def("get_surface_area", &volumetric::BoundingBox3D::get_surface_area)
        .def("get_diagonal_length", &volumetric::BoundingBox3D::get_diagonal_length)
        .def("contains", [](const volumetric::BoundingBox3D &self,
                            const Eigen::Vector3d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const volumetric::BoundingBox3D &self, const std::vector<Eigen::Vector3d> &points) {
                 return self.contains(points);
             })
        .def("intersects",
             [](const volumetric::BoundingBox3D &self, const volumetric::BoundingBox3D &other) {
                 return self.intersects(other);
             });

    // OrientedBoundingBox3D class
    py::class_<volumetric::OrientedBoundingBox3D,
               std::shared_ptr<volumetric::OrientedBoundingBox3D>>(m, "OrientedBoundingBox3D")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Quaterniond &,
                      const Eigen::Vector3d &>(),
             py::arg("center"), py::arg("orientation"), py::arg("size"))
        .def(py::init([](const Eigen::Vector3d &center, const py::array_t<double> &orientation_arr,
                         const Eigen::Vector3d &size) {
                 auto r = orientation_arr.unchecked<1>();
                 if (orientation_arr.size() != 4) {
                     throw std::runtime_error(
                         "Orientation array must have exactly 4 elements [w, x, y, z]");
                 }
                 Eigen::Quaterniond orientation(r(0), r(1), r(2), r(3)); // w, x, y, z
                 return volumetric::OrientedBoundingBox3D(center, orientation, size);
             }),
             "Constructor from center, orientation array [w,x,y,z], and size", py::arg("center"),
             py::arg("orientation"), py::arg("size"))
        .def_readwrite("center", &volumetric::OrientedBoundingBox3D::center)
        .def_readwrite("orientation", &volumetric::OrientedBoundingBox3D::orientation)
        .def_readwrite("size", &volumetric::OrientedBoundingBox3D::size)
        .def("get_volume", &volumetric::OrientedBoundingBox3D::get_volume)
        .def("get_surface_area", &volumetric::OrientedBoundingBox3D::get_surface_area)
        .def("get_diagonal_length", &volumetric::OrientedBoundingBox3D::get_diagonal_length)
        .def("get_corners", &volumetric::OrientedBoundingBox3D::get_corners)
        .def("contains", [](const volumetric::OrientedBoundingBox3D &self,
                            const Eigen::Vector3d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const volumetric::OrientedBoundingBox3D &self,
                const std::vector<Eigen::Vector3d> &points_w) { return self.contains(points_w); })
        .def("intersects",
             [](const volumetric::OrientedBoundingBox3D &self,
                const volumetric::OrientedBoundingBox3D &other) { return self.intersects(other); })
        .def("intersects",
             [](const volumetric::OrientedBoundingBox3D &self,
                const volumetric::BoundingBox3D &other) { return self.intersects(other); });

    // BoundingBox2D class
    py::class_<volumetric::BoundingBox2D, std::shared_ptr<volumetric::BoundingBox2D>>(
        m, "BoundingBox2D")
        .def(py::init<const Eigen::Vector2d &, const Eigen::Vector2d &>(), py::arg("min_point"),
             py::arg("max_point"))
        .def_readwrite("min_x", &volumetric::BoundingBox2D::min_x)
        .def_readwrite("min_y", &volumetric::BoundingBox2D::min_y)
        .def_readwrite("max_x", &volumetric::BoundingBox2D::max_x)
        .def_readwrite("max_y", &volumetric::BoundingBox2D::max_y)
        .def("get_min_point", &volumetric::BoundingBox2D::get_min_point)
        .def("get_max_point", &volumetric::BoundingBox2D::get_max_point)
        .def("get_center", &volumetric::BoundingBox2D::get_center)
        .def("get_size", &volumetric::BoundingBox2D::get_size)
        .def("get_area", &volumetric::BoundingBox2D::get_area)
        .def("get_perimeter", &volumetric::BoundingBox2D::get_perimeter)
        .def("get_diagonal_length", &volumetric::BoundingBox2D::get_diagonal_length)
        .def("contains", [](const volumetric::BoundingBox2D &self,
                            const Eigen::Vector2d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const volumetric::BoundingBox2D &self,
                const std::vector<Eigen::Vector2d> &points_w) { return self.contains(points_w); })
        .def("intersects",
             [](const volumetric::BoundingBox2D &self, const volumetric::BoundingBox2D &other) {
                 return self.intersects(other);
             });

    // OrientedBoundingBox2D class
    py::class_<volumetric::OrientedBoundingBox2D,
               std::shared_ptr<volumetric::OrientedBoundingBox2D>>(m, "OrientedBoundingBox2D")
        .def(py::init<const Eigen::Vector2d &, const double, const Eigen::Vector2d &>(),
             py::arg("center"), py::arg("angle_rad"), py::arg("size"))
        .def_readwrite("center", &volumetric::OrientedBoundingBox2D::center)
        .def_readwrite("angle_rad", &volumetric::OrientedBoundingBox2D::angle_rad)
        .def_readwrite("size", &volumetric::OrientedBoundingBox2D::size)
        .def("get_volume", &volumetric::OrientedBoundingBox2D::get_volume)
        .def("get_area", &volumetric::OrientedBoundingBox2D::get_area)
        .def("get_perimeter", &volumetric::OrientedBoundingBox2D::get_perimeter)
        .def("get_diagonal_length", &volumetric::OrientedBoundingBox2D::get_diagonal_length)
        .def("get_corners", &volumetric::OrientedBoundingBox2D::get_corners)
        .def("contains", [](const volumetric::OrientedBoundingBox2D &self,
                            const Eigen::Vector2d &point_w) { return self.contains(point_w); })
        .def("contains",
             [](const volumetric::OrientedBoundingBox2D &self,
                const std::vector<Eigen::Vector2d> &points_w) { return self.contains(points_w); })
        .def("intersects",
             [](const volumetric::OrientedBoundingBox2D &self,
                const volumetric::OrientedBoundingBox2D &other) { return self.intersects(other); })
        .def("intersects",
             [](const volumetric::OrientedBoundingBox2D &self,
                const volumetric::BoundingBox2D &other) { return self.intersects(other); });

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
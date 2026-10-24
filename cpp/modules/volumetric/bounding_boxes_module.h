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

void bind_bounding_boxes(py::module &m) {

    // ================================================================
    // OBBComputationMethod
    // ================================================================
    py::enum_<volumetric::OBBComputationMethod>(m, "OBBComputationMethod")
        .value("PCA", volumetric::OBBComputationMethod::PCA)
        .value("CONVEX_HULL_MINIMAL", volumetric::OBBComputationMethod::CONVEX_HULL_MINIMAL)
        .export_values();

    // ================================================================
    // Bounding Boxes 3D
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
             })
        .def_static("compute_from_points",
                    [](const std::vector<Eigen::Matrix<double, 3, 1>> &points_w) {
                        return volumetric::BoundingBox3D::compute_from_points(points_w);
                    })
        .def_static("compute_from_points",
                    [](const std::vector<Eigen::Matrix<float, 3, 1>> &points_w) {
                        return volumetric::BoundingBox3D::compute_from_points(points_w);
                    })
        .def(py::pickle(
            [](const volumetric::BoundingBox3D &self) {
                return py::make_tuple(self.get_min_point(), self.get_max_point());
            },
            [](py::tuple state) {
                if (state.size() != 2) {
                    throw py::value_error("Invalid state for BoundingBox3D");
                }
                return volumetric::BoundingBox3D(state[0].cast<Eigen::Vector3d>(),
                                                 state[1].cast<Eigen::Vector3d>());
            }));

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
        .def("get_matrix", &volumetric::OrientedBoundingBox3D::get_matrix)
        .def("get_inverse_matrix", &volumetric::OrientedBoundingBox3D::get_inverse_matrix)
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
                const volumetric::BoundingBox3D &other) { return self.intersects(other); })
        .def_static(
            "compute_from_points",
            [](const std::vector<Eigen::Matrix<double, 3, 1>> &points_w,
               const volumetric::OBBComputationMethod &method) {
                return volumetric::OrientedBoundingBox3D::compute_from_points(points_w, method);
            },
            py::arg("points_w"), py::arg("method") = volumetric::OBBComputationMethod::PCA)
        .def_static(
            "compute_from_points",
            [](const std::vector<Eigen::Matrix<float, 3, 1>> &points_w,
               const volumetric::OBBComputationMethod &method) {
                return volumetric::OrientedBoundingBox3D::compute_from_points(points_w, method);
            },
            py::arg("points_w"), py::arg("method") = volumetric::OBBComputationMethod::PCA)
        .def(py::pickle(
            [](const volumetric::OrientedBoundingBox3D &self) {
                return py::make_tuple(self.center, self.orientation, self.size);
            },
            [](py::tuple state) {
                if (state.size() != 3) {
                    throw py::value_error("Invalid state for OrientedBoundingBox3D");
                }
                return volumetric::OrientedBoundingBox3D(state[0].cast<Eigen::Vector3d>(),
                                                         state[1].cast<Eigen::Quaterniond>(),
                                                         state[2].cast<Eigen::Vector3d>());
            }));

    // ================================================================
    // Bounding Boxes 2D
    // ================================================================

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
             })
        .def_static("compute_from_points",
                    [](const std::vector<Eigen::Matrix<double, 2, 1>> &points_w) {
                        return volumetric::BoundingBox2D::compute_from_points(points_w);
                    })
        .def_static("compute_from_points",
                    [](const std::vector<Eigen::Matrix<float, 2, 1>> &points_w) {
                        return volumetric::BoundingBox2D::compute_from_points(points_w);
                    })
        .def(py::pickle(
            [](const volumetric::BoundingBox2D &self) {
                return py::make_tuple(self.get_min_point(), self.get_max_point());
            },
            [](py::tuple state) {
                if (state.size() != 2) {
                    throw py::value_error("Invalid state for BoundingBox2D");
                }
                return volumetric::BoundingBox2D(state[0].cast<Eigen::Vector2d>(),
                                                 state[1].cast<Eigen::Vector2d>());
            }));

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
                const volumetric::BoundingBox2D &other) { return self.intersects(other); })
        .def_static(
            "compute_from_points",
            [](const std::vector<Eigen::Matrix<double, 2, 1>> &points_w,
               const volumetric::OBBComputationMethod &method) {
                return volumetric::OrientedBoundingBox2D::compute_from_points(points_w, method);
            },
            py::arg("points_w"), py::arg("method") = volumetric::OBBComputationMethod::PCA)
        .def_static(
            "compute_from_points",
            [](const std::vector<Eigen::Matrix<float, 2, 1>> &points_w,
               const volumetric::OBBComputationMethod &method) {
                return volumetric::OrientedBoundingBox2D::compute_from_points(points_w, method);
            },
            py::arg("points_w"), py::arg("method") = volumetric::OBBComputationMethod::PCA)
        .def(py::pickle(
            [](const volumetric::OrientedBoundingBox2D &self) {
                return py::make_tuple(self.center, self.angle_rad, self.size);
            },
            [](py::tuple state) {
                if (state.size() != 3) {
                    throw py::value_error("Invalid state for OrientedBoundingBox2D");
                }
                return volumetric::OrientedBoundingBox2D(state[0].cast<Eigen::Vector2d>(),
                                                         state[1].cast<double>(),
                                                         state[2].cast<Eigen::Vector2d>());
            }));
}
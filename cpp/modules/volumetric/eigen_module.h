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

#include <Eigen/Geometry>

namespace py = pybind11;

void bind_eigen(py::module &m) {

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
        .def("__str__",
             [](const Eigen::Quaterniond &q) {
                 return "Quaterniond(w=" + std::to_string(q.w()) + ", x=" + std::to_string(q.x()) +
                        ", y=" + std::to_string(q.y()) + ", z=" + std::to_string(q.z()) + ")";
             })
        .def(py::pickle(
            [](const Eigen::Quaterniond &q) { return py::make_tuple(q.w(), q.x(), q.y(), q.z()); },
            [](py::tuple state) {
                if (state.size() != 4) {
                    throw py::value_error("Invalid state for Quaterniond");
                }
                return Eigen::Quaterniond(state[0].cast<double>(), state[1].cast<double>(),
                                          state[2].cast<double>(), state[3].cast<double>());
            }));
}
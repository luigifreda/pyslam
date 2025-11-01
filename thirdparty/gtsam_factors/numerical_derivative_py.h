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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <functional>

#include <gtsam/geometry/Similarity3.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>

using namespace gtsam;
namespace py = pybind11;

namespace gtsam_factors {

// Python wrapper for generalized function
Eigen::MatrixXd numericalDerivative11_V2_Pose3(
    py::function py_func, const Pose3& x, double delta = 1e-5) {
    
    std::function<Vector2(const Pose3&)> func = 
        [py_func](const Pose3& sim) -> Vector2 {
            return py_func(sim).cast<Vector2>();
        };

    // Generalized numerical derivative function
    return numericalDerivative11General<Vector2, Pose3>(func, x, delta);
}
Eigen::MatrixXd numericalDerivative11_V3_Pose3(
    py::function py_func, const Pose3& x, double delta = 1e-5) {
    
    std::function<Vector3(const Pose3&)> func = 
        [py_func](const Pose3& sim) -> Vector3 {
            return py_func(sim).cast<Vector3>();
        };

    // Generalized numerical derivative function
    return numericalDerivative11General<Vector3, Pose3>(func, x, delta);
}

// Python wrapper for generalized function
Eigen::MatrixXd numericalDerivative11_V2_Sim3(
    py::function py_func, const Similarity3& x, double delta = 1e-5) {
    
    std::function<Vector2(const Similarity3&)> func = 
        [py_func](const Similarity3& sim) -> Vector2 {
            return py_func(sim).cast<Vector2>();
        };

    // Generalized numerical derivative function
    return numericalDerivative11General<Vector2, Similarity3>(func, x, delta);
}
Eigen::MatrixXd numericalDerivative11_V3_Sim3(
    py::function py_func, const Similarity3& x, double delta = 1e-5) {
    
    std::function<Vector3(const Similarity3&)> func = 
        [py_func](const Similarity3& sim) -> Vector3 {
            return py_func(sim).cast<Vector3>();
        };

    // Generalized numerical derivative function
    return numericalDerivative11General<Vector3, Similarity3>(func, x, delta);
}

// Python wrapper for other types, just in case
template <typename Y, typename X>
Eigen::MatrixXd numericalDerivative11_Any(
    py::function py_func, const X& x, double delta = 1e-5) {
    
    std::function<Y(const X&)> func = 
        [py_func](const X& arg) -> Y {
            return py_func(arg).template cast<Y>();
        };

    return numericalDerivative11General<Y, X>(func, x, delta);
}

} // namespace gtsam_factors
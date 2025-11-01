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

#include <gtsam/geometry/Similarity3.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>

#include <functional>
#include <Eigen/Dense>

using namespace gtsam;

namespace gtsam_factors {

// Generalized numericalDerivative11 template
template <typename Y, typename X>
Eigen::MatrixXd numericalDerivative11General(
    std::function<Y(const X&)> h, const X& x, double delta = 1e-5) {
  
    return numericalDerivative11<Y, X>(h, x, delta);
}


// Specialization for Pose3 -> Vector2
template <>
Eigen::MatrixXd numericalDerivative11General<Vector2, Pose3>(
    std::function<Vector2(const Pose3&)> h, const Pose3& x, double delta) {
  
    return numericalDerivative11<Vector2, Pose3>(h, x, delta);
}

// Specialization for Pose3 -> Vector3
template <>
Eigen::MatrixXd numericalDerivative11General<Vector3, Pose3>(
    std::function<Vector3(const Pose3&)> h, const Pose3& x, double delta) {
  
    return numericalDerivative11<Vector3, Pose3>(h, x, delta);
}

// Specialization for Similarity3 -> Vector2
template <>
Eigen::MatrixXd numericalDerivative11General<Vector2, Similarity3>(
    std::function<Vector2(const Similarity3&)> h, const Similarity3& x, double delta) {
  
    return numericalDerivative11<Vector2, Similarity3>(h, x, delta);
}

} // namespace gtsam_factors
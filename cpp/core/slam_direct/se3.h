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

#include "eigen_aliases.h"
#include "utils/eigen_helpers.h"

#include <Eigen/Geometry>

#include <cmath>

namespace pyslam {

// Left increment T <- exp(xi) * T.
//
// The photometric aligner uses the Sophus order xi = [translation, rotation].
// The geometric pose step uses g2o_left_se3_jacobian, whose order is
// xi = [rotation, translation]. Both are left perturbations of the same SE3.

// Small-angle series avoid division by a near-zero rotation angle.
inline Mat3d so3_exp(const Vec3d &phi) {
    const double theta = phi.norm();
    const Mat3d phi_hat = skew(phi);
    if (theta < 1e-8) {
        return Mat3d::Identity() + phi_hat + 0.5 * phi_hat * phi_hat;
    }
    const double theta2 = theta * theta;
    const double a = std::sin(theta) / theta;
    const double b = (1.0 - std::cos(theta)) / theta2;
    return Mat3d::Identity() + a * phi_hat + b * phi_hat * phi_hat;
}

inline Mat3d so3_left_jacobian(const Vec3d &phi) {
    const double theta = phi.norm();
    const Mat3d phi_hat = skew(phi);
    if (theta < 1e-8) {
        return Mat3d::Identity() + 0.5 * phi_hat;
    }
    const double theta2 = theta * theta;
    const double a = (1.0 - std::cos(theta)) / theta2;
    const double b = (theta - std::sin(theta)) / (theta2 * theta);
    return Mat3d::Identity() + a * phi_hat + b * phi_hat * phi_hat;
}

// The SE3 translation is J_left(phi) * rho, not rho alone for a finite rotation.
inline Eigen::Isometry3d exp_se3(const Vec3d &rho, const Vec3d &phi) {
    Eigen::Isometry3d increment = Eigen::Isometry3d::Identity();
    increment.linear() = so3_exp(phi);
    increment.translation() = so3_left_jacobian(phi) * rho;
    return increment;
}

// xi = [tx, ty, tz, wx, wy, wz]
inline void left_multiply_translation_rotation(Eigen::Isometry3d &pose, const Vec6d &xi) {
    pose = exp_se3(xi.head<3>(), xi.tail<3>()) * pose;
}

// xi = [wx, wy, wz, tx, ty, tz], matching g2o_left_se3_jacobian
inline void left_multiply_rotation_translation(Eigen::Isometry3d &pose, const Vec6d &xi) {
    pose = exp_se3(xi.tail<3>(), xi.head<3>()) * pose;
}

} // namespace pyslam

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

#include "common/eigen_aliases.h"

#include <Eigen/Dense>

#include <cmath>

namespace pyslam {

// Midpoint triangulation of two bearings: X = d0 * f_ref in the reference camera and
// R * X + t = d1 * f_cur in the current camera. Fails unless both ray distances are positive.
inline bool triangulate_midpoint(const Vec3d &f_ref, const Vec3d &f_cur, const Mat3d &R, const Vec3d &t,
                                 Vec3d &X_ref) {
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = R * f_ref;
    A.col(1) = -f_cur;
    const Eigen::Matrix2d AtA = A.transpose() * A;
    if (!(std::abs(AtA.determinant()) > 1e-12)) return false;
    const Eigen::Vector2d d = AtA.ldlt().solve(A.transpose() * (-t));
    if (!(d.x() > 0.0) || !(d.y() > 0.0) || !d.allFinite()) return false;
    const Vec3d X0 = d.x() * f_ref;
    const Vec3d X1 = R.transpose() * (d.y() * f_cur - t); // Current ray point, in the reference frame.
    X_ref = 0.5 * (X0 + X1);
    return X_ref.allFinite();
}

} // namespace pyslam

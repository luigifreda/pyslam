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


#include "slam/direct/geometry/ray_geometry.h"

#include <algorithm>
#include <cmath>

namespace pyslam {

bool lift_reference_point(const Camera &camera, const Vec2d &pixel, double depth, Vec3d &X_ref) {
    if (!(depth > 0.0) || !std::isfinite(depth)) {
        return false;
    }
    Vec3d bearing;
    constexpr double kMinBearingZ = 1e-4;
    if (!camera.unproject_bearing(pixel, bearing, nullptr) || !bearing.allFinite() ||
        !(bearing.z() > kMinBearingZ)) {
        return false;
    }
    X_ref = bearing * (depth / bearing.z());
    return X_ref.allFinite();
}

double depth_uncertainty_one_pixel(const Camera &camera, const Eigen::Isometry3d &T_cur_ref,
                                   const Vec2d &pixel_ref, double depth, double px_error_angle) {
    Vec3d f = Vec3d::Zero();
    if (!(depth > 0.0) || !camera.unproject_bearing(pixel_ref, f, nullptr) || !f.allFinite()) {
        return -1.0;
    }
    f.normalize();
    if (!(f.z() > 1e-8)) {
        return -1.0;
    }
    // rpg_svo DepthFilter::computeTau, with t the current camera center in the reference
    // frame and the distance along the unit ray f. The result is converted back to z.
    const Vec3d t = T_cur_ref.inverse().translation();
    const double t_norm = t.norm();
    const double range = depth / f.z();
    const Vec3d a = f * range - t;
    const double a_norm = a.norm();
    if (!(t_norm > 1e-12) || !(a_norm > 1e-12)) {
        return -1.0;
    }
    const double alpha = std::acos(std::clamp(f.dot(t) / t_norm, -1.0, 1.0));
    const double beta = std::acos(std::clamp(a.dot(-t) / (t_norm * a_norm), -1.0, 1.0));
    const double beta_plus = beta + px_error_angle;
    const double gamma_plus = 3.14159265358979323846 - alpha - beta_plus;
    if (!(gamma_plus > 1e-12)) {
        return -1.0;
    }
    const double range_plus = t_norm * std::sin(beta_plus) / std::sin(gamma_plus);
    const double tau = (range_plus - range) * f.z();
    return std::isfinite(tau) && tau > 0.0 ? tau : -1.0;
}

} // namespace pyslam

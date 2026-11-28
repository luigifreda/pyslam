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

#include "camera/camera.h"
#include "slam/direct/core/direct_params.h"

#include <Eigen/Geometry>

#include <vector>

namespace pyslam {

// Geometric (reprojection-error) pose solver. Shared by every tracking mode: klt and
// hybrid feed it LK tracks, semi_direct feeds it photometrically aligned feature pixels.

// A 3D-to-2D correspondence: point_ref can be in any common reference coordinate frame.
struct PoseObservation {
    Vec3d point_ref = Vec3d::Zero();
    Vec2d pixel_cur = Vec2d::Zero();
};

struct ReprojectionPoseResult {
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    int inliers = 0;
    double cost = 0.0; // Sum of squared pixel errors for final geometric inliers.
    std::vector<unsigned char> inlier_mask; // Aligned with the input observations, not LK indices.
};

// Gauss-Newton on reprojection error. T_cur_ref maps a reference-camera point
// into the current camera. The frontend uses world coordinates as the reference.
// The pose Jacobian is g2o_left_se3_jacobian.
ReprojectionPoseResult estimate_pose_reprojection(const Camera &camera,
                                                const std::vector<PoseObservation> &observations,
                                                const Eigen::Isometry3d &T_cur_ref_init,
                                                const DirectSlamParams &params);

} // namespace pyslam

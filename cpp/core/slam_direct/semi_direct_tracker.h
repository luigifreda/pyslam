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

#include "camera.h"
#include "slam_direct/direct_params.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// `occupied` is optional. Zero pixels are excluded, on top of the border mask.
// `max_corners` <= 0 uses params.num_features.
std::vector<Vec2d> detect_gftt(const cv::Mat &gray, const DirectSlamParams &params,
                               const cv::Mat &occupied = cv::Mat(), int max_corners = -1);

// Output arrays keep input-point indexing, including failed tracks (status == 0).
struct LkTrack {
    std::vector<Vec2d> points_cur;
    std::vector<unsigned char> status;
};

// Seed forward LK with predicted pixels, then reject inconsistent backward tracks.
LkTrack track_pyramidal_lk(const cv::Mat &img_ref, const cv::Mat &img_cur,
                           const std::vector<Vec2d> &points_ref,
                           const std::vector<Vec2d> &points_cur_init, const DirectSlamParams &params);

// A 3D-to-2D correspondence: point_ref can be in any common reference coordinate frame.
struct PoseObservation {
    Vec3d point_ref = Vec3d::Zero();
    Vec2d pixel_cur = Vec2d::Zero();
};

struct SemiDirectPoseResult {
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    int inliers = 0;
    double cost = 0.0; // Sum of squared pixel errors for final geometric inliers.
    std::vector<unsigned char> inlier_mask; // Aligned with the input observations, not LK indices.
};

// Gauss-Newton on reprojection error. T_cur_ref maps a reference-camera point
// into the current camera. The frontend uses world coordinates as the reference.
// The pose Jacobian is g2o_left_se3_jacobian.
SemiDirectPoseResult estimate_pose_reprojection(const Camera &camera,
                                                const std::vector<PoseObservation> &observations,
                                                const Eigen::Isometry3d &T_cur_ref_init,
                                                const DirectSlamParams &params);

} // namespace pyslam

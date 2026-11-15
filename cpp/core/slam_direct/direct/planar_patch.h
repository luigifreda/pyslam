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
#include "slam_direct/direct_params.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Pose from fronto-parallel patches. Each patch pixel is its own 3D point at the
// feature depth in the reference camera. T_cur_ref maps that camera into the current one.
struct PlanarPoseResult {
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    double cost = 0.0;
    int visible = 0;
    int inliers = 0;
    int iterations = 0;
    bool converged = false;
    std::vector<unsigned char> inlier_mask;
};

// Per-feature 2D correction after the pose is fixed. pixels_cur is the feature center
// in the current image. status is 0 when the patch residual or the shift is rejected.
struct PatchOffsetResult {
    std::vector<Vec2d> pixels_cur;
    std::vector<unsigned char> status;
};

PlanarPoseResult align_planar_patch_pose(const Camera &camera, const DirectSlamParams &params,
                                         const cv::Mat &img_ref, const cv::Mat &img_cur,
                                         const std::vector<Vec2d> &pixels_ref,
                                         const std::vector<double> &depth_ref,
                                         const Eigen::Isometry3d &T_cur_ref_init);

PatchOffsetResult align_planar_patch_offsets(const Camera &camera, const DirectSlamParams &params,
                                             const cv::Mat &img_ref, const cv::Mat &img_cur,
                                             const std::vector<Vec2d> &pixels_ref,
                                             const std::vector<double> &depth_ref,
                                             const Eigen::Isometry3d &T_cur_ref);

} // namespace pyslam

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
#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Diagnostics for the returned pose at the evaluated image level (finest for multilayer).
struct DirectAlignmentResult {
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    double cost = 0.0; // Mean squared gray-level error, averaged over patches and visible points.
    int visible = 0; // Reference points whose warped patches are inside the current image.
    int inliers = 0;
    int iterations = 0;
    bool converged = false;
    // One flag per input reference pixel; invalid/invisible points remain zero.
    std::vector<unsigned char> inlier_mask;
};

// Project X_cur and return ∂(u, v) / ∂ξ for the left increment ξ = [tx, ty, tz, wx, wy, wz].
// A pyramid level multiplies both the pixel and the Jacobian by scale. The projection
// itself uses the full-resolution camera, including its distortion Jacobian.
bool photometric_warp_jacobian(const Camera &camera, const Vec3d &X_cur, double scale, Vec2d &uv,
                               Eigen::Matrix<double, 2, 6> &J_pixel_xi);

// Sparse photometric alignment. T_cur_ref maps a point in the reference camera
// to the current camera: X_cur = T_cur_ref * X_ref. Images are CV_8U grayscale.
// Raw-pixel cameras are projected with their distortion model. Images are not remapped.
DirectAlignmentResult align_direct_single_layer(const Camera &camera, const DirectSlamParams &params,
                                                const cv::Mat &img_ref, const cv::Mat &img_cur,
                                                const std::vector<Vec2d> &pixels_ref,
                                                const std::vector<double> &depth_ref,
                                                const Eigen::Isometry3d &T_cur_ref_init);

// Carry a metric SE3 estimate from coarse to fine. Reference points are lifted once.
// Each level scales the projected pixel and its Jacobian.
DirectAlignmentResult align_direct_multilayer(const Camera &camera, const DirectSlamParams &params,
                                              const cv::Mat &img_ref, const cv::Mat &img_cur,
                                              const std::vector<Vec2d> &pixels_ref,
                                              const std::vector<double> &depth_ref,
                                              const Eigen::Isometry3d &T_cur_ref_init);

} // namespace pyslam

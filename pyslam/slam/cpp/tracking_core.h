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

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

#include "config_parameters.h"
#include "eigen_aliases.h"
#include "frame.h"
#include "map.h"
#include "map_point.h"
#include "smart_pointers.h"
#include "utils/cv_ops.h"

#include <tuple>
#include <vector>

namespace pyslam {

// Utility functions for tracking operations
class TrackingCore {

  public:
    static std::tuple<int, std::vector<int>, std::vector<int>>
    propagate_map_point_matches(const FramePtr &f_ref, FramePtr &f_cur,
                                const std::vector<int> &idxs_ref, const std::vector<int> &idxs_cur,
                                float max_descriptor_distance = -1.0f);

    static std::vector<MapPointPtr>
    create_vo_points(FramePtr &frame, int max_num_points = Parameters::kMaxNumVisualOdometryPoints,
                     const Vec3b &color = Vec3b(0, 0, 255));

    static int create_and_add_stereo_map_points_on_new_kf(FramePtr &frame, KeyFramePtr &kf,
                                                          MapPtr &map, const cv::Mat &img);

    static std::tuple<int, int, std::vector<bool>>
    count_tracked_and_non_tracked_close_points(const FramePtr &f_cur, const SensorType sensor_type);

    // Estimate a pose from a fitted essential mat;
    // since we do not have an interframe translation scale, this fitting can be used to detect
    // outliers, estimate interframe orientation and translation direction
    // N.B. read the NBs of the method estimate_pose_ess_mat(), where the limitations of this method
    // are explained
    static std::tuple<std::vector<int>, std::vector<int>, int>
    estimate_pose_by_fitting_ess_mat(const FramePtr &f_ref, FramePtr &f_cur,
                                     const std::vector<int> &idxs_ref,
                                     const std::vector<int> &idxs_cur);

    // Use a general homography RANSAC matcher with a large threshold of 5 pixels to model the
    // inter-frame transformation for a generic motion
    // NOTE: this method is used to find inliers and estimate the inter-frame transformation
    // (assuming frames are very close in space)
    static std::tuple<bool, std::vector<int>, std::vector<int>, int, int>
    find_homography_with_ransac(const FramePtr &f_cur, const FramePtr &f_ref,
                                const std::vector<int> &idxs_cur, const std::vector<int> &idxs_ref,
                                const double reproj_threshold = Parameters::kRansacReprojThreshold,
                                const int min_num_inliers = Parameters::kRansacMinNumInliers);
};

} // namespace pyslam
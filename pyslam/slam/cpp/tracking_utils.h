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
class TrackingUtils {
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
    //===============================================
    // WIP
    //===============================================
    // Essential matrix pose estimation
    static std::pair<Eigen::Matrix4d, cv::Mat> estimate_pose_ess_mat(const MatNx2f &kps_ref,
                                                                     const MatNx2f &kps_cur,
                                                                     int method = cv::USAC_MAGSAC,
                                                                     double prob = 0.999,
                                                                     double threshold = 0.0004);

    // Homography estimation with RANSAC
    static std::pair<cv::Mat, cv::Mat> find_homography_with_ransac(const MatNx2f &kps_cur,
                                                                   const MatNx2f &kps_ref,
                                                                   int method = cv::USAC_MAGSAC,
                                                                   double reproj_threshold = 5.0);
};

} // namespace pyslam
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
#include <unordered_map>

namespace pyslam {

// Result structures for optimization (same as optimizer_g2o for compatibility)
struct BundleAdjustmentResult {
    double mean_squared_error = -1.0;
    std::unordered_map<int, Eigen::Matrix4d> keyframe_updates;
    std::unordered_map<int, Eigen::Vector3d> point_updates;
};

struct PoseOptimizationResult {
    double mean_squared_error = -1.0;
    bool is_ok = false;
    int num_valid_points = 0;
};

struct Sim3OptimizationResult {
    Sim3OptimizationResult()
        : num_inliers(0), R(Eigen::Matrix3d::Zero()), t(Eigen::Vector3d::Zero()), scale(1.0),
          delta_error(0.0) {}
    Sim3OptimizationResult(int num_inliers, Eigen::Matrix3d R, Eigen::Vector3d t, double scale,
                           double delta_error)
        : num_inliers(num_inliers), R(R), t(t), scale(scale), delta_error(delta_error) {}
    int num_inliers = 0;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double scale = 1.0;
    double delta_error = 0.0;
};

} // namespace pyslam
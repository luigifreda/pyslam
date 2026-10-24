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

#include "optimizer_common.h"
#include "sim3_pose.h"
#include "smart_pointers.h"

#include <Eigen/Dense>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

// Forward declarations for g2o
namespace g2o {
class SparseOptimizer;
class VertexSE3Expmap;
class VertexSBAPointXYZ;
class VertexSim3Expmap;
class EdgeSE3ProjectXYZ;
class EdgeStereoSE3ProjectXYZ;
class EdgeSE3ProjectXYZOnlyPose;
class EdgeStereoSE3ProjectXYZOnlyPose;
class EdgeSim3ProjectXYZ;
class EdgeInverseSim3ProjectXYZ;
class EdgeSim3;
class RobustKernelHuber;
class SE3Quat;
class Sim3;
} // namespace g2o

namespace pyslam {

// Main optimizer class
class OptimizerG2o {
  public:
    // ------------------------------------------------------------
    // Bundle adjustment functions

    static BundleAdjustmentResult bundle_adjustment(
        const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
        std::optional<int> local_window_size = std::nullopt, bool fixed_points = false,
        int rounds = 10, int loop_kf_id = 0, bool use_robust_kernel = false,
        bool *abort_flag = nullptr, bool fill_result_dict = false, bool verbose = false);

    static BundleAdjustmentResult global_bundle_adjustment(
        const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
        int rounds = 10, int loop_kf_id = 0, bool use_robust_kernel = false,
        bool *abort_flag = nullptr, bool fill_result_dict = false, bool verbose = false);

    static BundleAdjustmentResult
    global_bundle_adjustment_map(MapPtr &map, int rounds = 10, int loop_kf_id = 0,
                                 bool use_robust_kernel = false, bool *abort_flag = nullptr,
                                 bool fill_result_dict = false, bool verbose = false);

    // ------------------------------------------------------------
    // Pose optimization
    static PoseOptimizationResult pose_optimization(FramePtr &frame, bool verbose = false,
                                                    int rounds = 10);

    // ------------------------------------------------------------
    // Local bundle adjustment
    // Return the mean squared error and the outlier_ratio
    template <typename LockType>
    static std::pair<double, double>
    local_bundle_adjustment(const std::vector<KeyFramePtr> &keyframes,
                            const std::vector<MapPointPtr> &points,
                            const std::vector<KeyFramePtr> &keyframes_ref = {},
                            bool fixed_points = false, bool verbose = false, int rounds = 10,
                            bool *abort_flag = nullptr, LockType *map_lock = nullptr);

    // ------------------------------------------------------------
    // Sim3 optimization

    static Sim3OptimizationResult optimize_sim3(KeyFramePtr &kf1, KeyFramePtr &kf2,
                                                const std::vector<MapPointPtr> &map_points1,
                                                const std::vector<MapPointPtr> &map_point_matches12,
                                                const Eigen::Matrix3d &R12,
                                                const Eigen::Vector3d &t12, double s12, double th2,
                                                bool fix_scale, bool verbose = false);

    // ------------------------------------------------------------
    // Essential graph optimization
    // Return the mean squared error
    static double optimize_essential_graph(
        MapPtr map_object, KeyFramePtr loop_keyframe, KeyFramePtr current_keyframe,
        const std::unordered_map<KeyFramePtr, Sim3Pose> &non_corrected_sim3_map,
        const std::unordered_map<KeyFramePtr, Sim3Pose> &corrected_sim3_map,
        const std::unordered_map<KeyFramePtr, std::vector<KeyFramePtr>> &loop_connections,
        bool fix_scale, bool verbose = false);
};

} // namespace pyslam

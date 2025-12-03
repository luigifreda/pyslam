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
#include "camera_pose.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "sim3_pose.h"

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

// Forward declarations
class Frame;
class KeyFrame;
class MapPoint;
class Map;

// Result structures for optimization
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
    static std::pair<double, double>
    local_bundle_adjustment(const std::vector<KeyFramePtr> &keyframes,
                            const std::vector<MapPointPtr> &points,
                            const std::vector<KeyFramePtr> &keyframes_ref = {},
                            bool fixed_points = false, bool verbose = false, int rounds = 10,
                            bool *abort_flag = nullptr, std::mutex *map_lock = nullptr);

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

  private:
    // Constants
    static constexpr double kChi2Mono = 5.991;      // chi-square 2 DOFs
    static constexpr double kChi2Stereo = 7.815;    // chi-square 3 DOFs
    static constexpr double kThHuberMono = 2.447;   // sqrt(5.991)
    static constexpr double kThHuberStereo = 2.796; // sqrt(7.815)
    static constexpr double kMaxOutliersRatioInPoseOptimization = 0.9;
};

} // namespace pyslam

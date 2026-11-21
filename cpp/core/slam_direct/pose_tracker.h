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
#include "slam_direct/direct_map.h"
#include "slam_direct/direct_params.h"
#include "slam_direct/live_point.h"
#include "common/smart_pointers.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace pyslam {

// One input frame, already converted to CV_8UC1 (depth keeps its sensor type).
struct TrackingFrame {
    const cv::Mat &gray;
    const cv::Mat &gray_right; // Empty unless the input is stereo.
    const cv::Mat &depth;      // Empty unless the input is RGBD.
    int img_id = -1;
};

// Read-only tracker input, built by DirectSlam for every frame after initialization.
struct TrackingInput {
    const TrackingFrame &frame;
    const cv::Mat &last_gray;                // Last accepted image.
    const std::vector<LivePoint> &points;    // Live landmarks from the last accepted image.
    const DirectMap &map;
    const Eigen::Isometry3d &T_prev_w;       // Last accepted pose (Tcw).
    const Eigen::Isometry3d &T_pred_w;       // Constant-velocity prediction (Tcw).
};

struct TrackingResult {
    bool accepted = false; // Enough inliers and a finite pose.
    Eigen::Isometry3d T_cur_w = Eigen::Isometry3d::Identity();
    int inliers = 0;
    // Aligned with TrackingInput::points: only tracked_ok points survive into the next image.
    std::vector<char> tracked_ok;
    std::vector<Vec2d> tracked_uv;

    // A rejected result with one cleared slot per input live point.
    static TrackingResult for_points(size_t num_points) {
        TrackingResult result;
        result.tracked_ok.assign(num_points, 0);
        result.tracked_uv.assign(num_points, Vec2d::Zero());
        return result;
    }
};

// Mutable map state a tracker may extend after an accepted pose (e.g. promoted depth seeds).
struct TrackerMapState {
    DirectMap &map;
    std::vector<LivePoint> &live_points;
    int64_t &next_point_id;
};

struct AcceptOutcome {
    bool force_keyframe = false;
};

// Pose tracking strategy behind DirectSlam. One implementation per DIRECT_SLAM.mode:
//   klt/KltPoseTracker               klt: LK tracks + reprojection pose (indirect)
//   HybridPoseTracker                hybrid: klt + sparse photometric refinement/fallback
//   direct/SemiDirectPoseTracker     semi_direct: SVO-style planar patches + depth seeds
// DirectSlam owns everything common: input checks, initialization (sensor depth or the
// monocular two-view initializer), the accept/reject decision, live-point bookkeeping, keyframes and local BA.
class PoseTracker {
  public:
    PoseTracker(CameraPtr camera, const DirectSlamParams &params)
        : camera_(std::move(camera)), params_(params) {}
    virtual ~PoseTracker() = default;
    PoseTracker(const PoseTracker &) = delete;
    PoseTracker &operator=(const PoseTracker &) = delete;

    virtual const char *name() const = 0;

    // Estimate the current pose from the last accepted state.
    virtual TrackingResult track(const TrackingInput &input) = 0;

    // Forget all tracker-owned state (DirectSlam::reset).
    virtual void reset() {}

    // Capabilities and state queried by DirectSlam.
    virtual bool supports_monocular() const { return false; }
    // Keyframes keep their gray image, and live points are re-anchored to each new keyframe.
    virtual bool uses_keyframe_images() const { return false; }
    // After an accepted pose and the live-point update, before corner replenishment.
    virtual AcceptOutcome after_accept(const TrackingFrame &, const Eigen::Isometry3d & /*T_cur_w*/,
                                       TrackerMapState &) {
        return {};
    }
    // After DirectSlam recorded a keyframe (including the first one).
    virtual void on_keyframe(const TrackingFrame &, int /*keyframe_id*/, TrackerMapState &) {}
    // A monocular loop correction scaled the geometry seen from keyframe i by 1/scales[i].
    virtual void apply_keyframe_scales(const std::vector<double> & /*scales*/) {}

  protected:
    CameraPtr camera_;
    DirectSlamParams params_;
};

std::unique_ptr<PoseTracker> make_pose_tracker(DirectTrackingMode mode, CameraPtr camera,
                                               const DirectSlamParams &params);

} // namespace pyslam

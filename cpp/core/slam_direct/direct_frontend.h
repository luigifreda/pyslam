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
#include "slam_direct/direct_backend.h"
#include "slam_direct/direct_params.h"
#include "slam_direct/geometry/monocular_initializer.h"
#include "slam_direct/live_point.h"
#include "slam_direct/pose_tracker.h"
#include "common/smart_pointers.h"

#include <opencv2/core.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pyslam {

enum class DirectTrackingStatus { Init = 0, Good = 1, Bad = 2, Lost = 3 };

// Owns the frontend and synchronous local BA; track() consumes one input frame.
// The mode-specific pose estimation lives in a PoseTracker (see pose_tracker.h);
// DirectSlam keeps what all modes share: initialization from sensor depth, the
// accept/reject decision, live points, keyframes and local BA.
class DirectSlam {
  public:
    DirectSlam(CameraPtr camera, const ConfigDict &config);

    // Failed tracking preserves the last accepted pose, image and map for recovery.
    void track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
               std::optional<double> timestamp);
    void reset();
    // Restart tracking after a loss at a pose found by relocalization (pyslam's Relocalizer on
    // the loop-closing keyframes). pixels[i] in this image observes points_w[i]; landmark_ids[i]
    // is the direct landmark it belongs to, or -1 to create a new landmark there. The matches
    // become live tracks and this frame becomes a keyframe, so tracking has a fresh reference.
    // Returns false (state unchanged) when fewer than num_features_tracking_bad + 1 usable
    // points remain, or when the map is not initialized.
    bool relocalize(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
                    double timestamp, const Eigen::Matrix4d &Tcw, const std::vector<Vec2d> &pixels,
                    const std::vector<Vec3d> &points_w, const std::vector<int64_t> &landmark_ids);
    // A manual solve is allowed even when automatic local_ba is disabled.
    DirectBAResult bundle_adjustment();
    // Replace archived keyframe poses and carry landmarks and the trajectory with them.
    // One finite Tcw is required per archived keyframe, in insertion order.
    // scales (empty, or one positive value per keyframe) is the monocular Sim(3) part of a
    // loop correction: geometry seen from keyframe i shrinks by 1/scales[i], as in ORB-SLAM's
    // point correction. Landmarks, inverse-depth priors, frame offsets and depth seeds follow.
    void apply_keyframe_poses(const std::vector<Eigen::Matrix4d> &Tcw, const std::vector<double> &scales = {});
    const DirectBAResult &last_ba() const { return last_ba_; }
    cv::Mat draw_feature_trails(const cv::Mat &image, bool with_level_radius = false,
                                int trail_max_length = 16) const;

    DirectTrackingStatus status() const { return status_; }
    std::string status_name() const;
    // Monocular only: input image id of the two-view initializer's reference frame while the
    // map is not initialized yet, otherwise -1.
    int init_reference_img_id() const {
        return mono_init_ && map_.keyframes().empty() ? mono_init_->reference_img_id() : -1;
    }
    const DirectSlamParams &params() const { return params_; }
    const CameraPose &pose() const { return current_pose_; }
    int last_inliers() const { return last_inliers_; }
    // Constant-velocity model: the per-frame motion T_current_previous used to predict the next pose.
    bool has_velocity() const { return has_velocity_; }
    const Eigen::Isometry3d &velocity() const { return velocity_; }
    const DirectMap &map() const { return map_; }
    const std::vector<LivePoint> &live_points() const { return live_points_; }

  private:
    std::vector<DirectPoint> make_points(const cv::Mat &gray, const cv::Mat &gray_right,
                                         const cv::Mat &depth) const;
    void initialize(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth, int img_id,
                    double timestamp);
    // Two keyframes (reference at the identity, current frame) from an accepted two-view geometry.
    void initialize_monocular_map(const cv::Mat &gray, int img_id, double timestamp,
                                  const MonocularInitResult &init);
    void record_keyframe(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                         int img_id, double timestamp, const Eigen::Isometry3d &Tcw);
    void set_status_from_inliers(int inliers);
    // Copy the corrected trajectory, motion seed and live landmarks back to the tracker.
    void refresh_tracking_state_from_map();
    std::vector<LivePoint> points_from_camera(const std::vector<DirectPoint> &points,
                                              const Eigen::Isometry3d &Tcw);
    void supplement_live_points(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                                const Eigen::Isometry3d &Tcw, int img_id, double timestamp);
    void note_tracking_failure(int inliers);
    TrackerMapState map_state() { return TrackerMapState{map_, live_points_, next_point_id_}; }

    CameraPtr camera_;
    DirectSlamParams params_;
    std::unique_ptr<PoseTracker> tracker_; // klt, hybrid or semi_direct pose estimation.
    std::unique_ptr<MonocularInitializer> mono_init_; // Monocular input only.
    DirectMap map_;
    DirectTrackingStatus status_ = DirectTrackingStatus::Init;
    CameraPose current_pose_; // Accepted world-to-camera transform Tcw.
    // Left motion increment T_current_previous, used only as a tracking seed.
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool has_velocity_ = false;
    Eigen::Isometry3d previous_pose_ = Eigen::Isometry3d::Identity();
    cv::Mat last_gray_;
    std::vector<LivePoint> live_points_;
    int consecutive_failures_ = 0;
    int last_inliers_ = 0;
    int64_t next_point_id_ = 0;
    int last_img_id_ = -1;
    DirectBAResult last_ba_;
};

} // namespace pyslam

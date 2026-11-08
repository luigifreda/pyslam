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
#include "slam_direct/direct_map.h"
#include "slam_direct/direct_backend.h"
#include "slam_direct/direct_params.h"
#include "smart_pointers.h"

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace pyslam {

enum class DirectTrackingStatus { Init = 0, Good = 1, Bad = 2, Lost = 3 };

// A point carried from the last accepted image (which may precede tracking failures). position_w is in the world frame.
struct LivePoint {
    int64_t id = -1;
    Vec2d uv = Vec2d::Zero();
    Vec3d position_w = Vec3d::Zero();
    std::vector<cv::Point> trail; // Newest first, capped at 64 accepted observations.
    int observations = 0;
};

// Owns the frontend and synchronous local BA; track() consumes one stereo/RGBD input frame.
class DirectSlam {
  public:
    DirectSlam(CameraPtr camera, const ConfigDict &config);

    // Failed tracking preserves the last accepted pose, image and map for recovery.
    void track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
               std::optional<double> timestamp);
    void reset();
    // A manual solve is allowed even when automatic local_ba is disabled.
    DirectBAResult bundle_adjustment();
    const DirectBAResult &last_ba() const { return last_ba_; }
    cv::Mat draw_feature_trails(const cv::Mat &image, bool with_level_radius = false,
                                int trail_max_length = 16) const;

    DirectTrackingStatus status() const { return status_; }
    std::string status_name() const;
    const DirectSlamParams &params() const { return params_; }
    const CameraPose &pose() const { return current_pose_; }
    int last_inliers() const { return last_inliers_; }
    const DirectMap &map() const { return map_; }
    const std::vector<LivePoint> &live_points() const { return live_points_; }

  private:
    std::vector<DirectPoint> make_points(const cv::Mat &gray, const cv::Mat &gray_right,
                                         const cv::Mat &depth) const;
    void initialize(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth, int img_id,
                    double timestamp);
    void record_keyframe(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                         int img_id, double timestamp, const Eigen::Isometry3d &Tcw);
    void set_status_from_inliers(int inliers);
    std::vector<LivePoint> points_from_camera(const std::vector<DirectPoint> &points,
                                              const Eigen::Isometry3d &Tcw);
    void supplement_live_points(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                                const Eigen::Isometry3d &Tcw, int img_id, double timestamp);

    CameraPtr camera_;
    DirectSlamParams params_;
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

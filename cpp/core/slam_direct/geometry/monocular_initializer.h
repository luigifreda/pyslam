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
#include "common/smart_pointers.h"
#include "slam_direct/direct_params.h"
#include "slam_direct/live_point.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Two-view geometry accepted by MonocularInitializer. Points are in the reference camera.
struct MonocularInitResult {
    int ref_img_id = -1;
    double ref_timestamp = 0.0;
    cv::Mat ref_gray;
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    std::vector<Vec2d> uv_ref;
    std::vector<Vec2d> uv_cur;
    std::vector<Vec3d> X_ref;
    // Inverse-depth variance of each point in the reference camera (0: no usable prior).
    std::vector<double> inverse_depth_sigma2;
    std::vector<std::vector<cv::Point>> trails; // Newest first, as LivePoint::trail.
};

// Monocular map bootstrap (SVO / ORB-SLAM style). Corners detected on a reference frame
// are tracked frame to frame with pyramidal LK. Once their median displacement exceeds
// init_min_disparity, the essential matrix (5-point RANSAC on bearings) gives the relative
// pose and the inliers are triangulated. Scale is fixed so that the median scene depth is
// the geometric mean of [seed_min_depth, seed_max_depth]. The reference is replaced when
// too few tracks survive or the geometry keeps failing at large disparity (e.g. pure rotation).
class MonocularInitializer {
  public:
    MonocularInitializer(CameraPtr camera, const DirectSlamParams &params);

    void reset();
    // Feed one frame. Returns true once result() holds an accepted two-view geometry.
    bool add_frame(const cv::Mat &gray, int img_id, double timestamp);

    const MonocularInitResult &result() const { return result_; }
    // Current reference-to-frame tracks, for drawing; uv_ref is on the reference frame.
    const std::vector<LivePoint> &tracks() const { return tracks_; }
    int num_tracks() const { return static_cast<int>(tracks_.size()); }
    // Input image id of the current reference frame, or -1 before the first frame.
    int reference_img_id() const { return ref_img_id_; }

  private:
    void set_reference(const cv::Mat &gray, int img_id, double timestamp);
    bool try_two_view();

    CameraPtr camera_;
    DirectSlamParams params_;
    cv::Mat ref_gray_;
    cv::Mat last_gray_;
    int ref_img_id_ = -1;
    int last_img_id_ = -1;
    double ref_timestamp_ = 0.0;
    std::vector<LivePoint> tracks_;
    MonocularInitResult result_;
};

} // namespace pyslam

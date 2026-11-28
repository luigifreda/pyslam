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

#include "slam/direct/photometric/depth_filter.h"
#include "slam/direct/core/pose_tracker.h"

#include <vector>

namespace pyslam {

// mode: semi_direct (SVO-style). Every frame is aligned to the latest keyframe image:
// a fronto-parallel patch pose (photometric), then an independent 2D alignment of each
// projected patch, then the shared reprojection pose. Corners without sensor depth become
// inverse-depth seeds and are promoted to landmarks once they converge. A seed is tracked
// frame to frame with LK and each correspondence is triangulated against its keyframe;
// a seed whose track was lost is updated by epipolar search instead. Monocular input starts from the two-view map built by MonocularInitializer.
class SemiDirectPoseTracker : public PoseTracker {
  public:
    SemiDirectPoseTracker(CameraPtr camera, const DirectSlamParams &params);
    const char *name() const override { return "semi_direct"; }

    TrackingResult track(const TrackingInput &input) override;
    void reset() override;

    bool supports_monocular() const override { return true; }
    bool uses_keyframe_images() const override { return true; }

    AcceptOutcome after_accept(const TrackingFrame &frame, const Eigen::Isometry3d &T_cur_w,
                               TrackerMapState &state) override;
    void on_keyframe(const TrackingFrame &frame, int keyframe_id, TrackerMapState &state) override;
    void apply_keyframe_scales(const std::vector<double> &scales) override;

  private:
    void spawn_depth_seeds(const TrackingFrame &frame, int keyframe_id, const TrackerMapState &state);
    void update_depth_seeds(const cv::Mat &gray, const Eigen::Isometry3d &T_cur_w, const DirectMap &map);
    int promote_depth_seeds(const Eigen::Isometry3d &T_cur_w, TrackerMapState &state);
    void release_idle_keyframe_images(DirectMap &map);

    std::vector<DepthSeed> seeds_;
    cv::Mat prev_gray_; // Image in which tracked seeds' pixel_cur lie.
    double prior_sigma2_ = 0.0;
};

} // namespace pyslam

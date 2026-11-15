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

#include "slam_direct/klt/klt_pose_tracker.h"

namespace pyslam {

// mode: hybrid. Deliberately combines both families, so it lives outside klt/ and
// direct/: the indirect KLT pose comes first, then sparse photometric alignment
// (direct/direct_aligner) refines it when the two agree, or replaces it when KLT fails.
class HybridPoseTracker : public PoseTracker {
  public:
    HybridPoseTracker(CameraPtr camera, const DirectSlamParams &params)
        : PoseTracker(camera, params), klt_(camera, params) {}
    const char *name() const override { return "hybrid"; }
    TrackingResult track(const TrackingInput &input) override;
    // Without local BA, refresh live-point depth from the current sensor measurement.
    AcceptOutcome after_accept(const TrackingFrame &frame, const Eigen::Isometry3d &T_cur_w,
                               TrackerMapState &state) override;

  private:
    KltPoseTracker klt_;
};

} // namespace pyslam

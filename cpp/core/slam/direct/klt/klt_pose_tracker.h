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

#include "slam/direct/core/pose_tracker.h"

namespace pyslam {

// mode: klt. An indirect tracker: pyramidal LK tracks every live point from the last
// accepted image (seeded by the constant-velocity projection), then PnP RANSAC and robust
// Gauss-Newton minimize reprojection error. No photometric error is optimized.
class KltPoseTracker : public PoseTracker {
  public:
    using PoseTracker::PoseTracker;
    const char *name() const override { return "klt"; }
    TrackingResult track(const TrackingInput &input) override;
};

} // namespace pyslam

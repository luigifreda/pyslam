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

#include "slam/direct/core/pose_tracker.h"

#include "slam/direct/semi_direct/semi_direct_pose_tracker.h"
#include "slam/direct/hybrid/hybrid_pose_tracker.h"
#include "slam/direct/klt/klt_pose_tracker.h"

#include <stdexcept>

namespace pyslam {

std::unique_ptr<PoseTracker> make_pose_tracker(DirectTrackingMode mode, CameraPtr camera,
                                               const DirectSlamParams &params) {
    switch (mode) {
    case DirectTrackingMode::Klt:
        return std::make_unique<KltPoseTracker>(std::move(camera), params);
    case DirectTrackingMode::Hybrid:
        return std::make_unique<HybridPoseTracker>(std::move(camera), params);
    case DirectTrackingMode::SemiDirect:
        return std::make_unique<SemiDirectPoseTracker>(std::move(camera), params);
    }
    throw std::invalid_argument("make_pose_tracker: unknown tracking mode");
}

} // namespace pyslam

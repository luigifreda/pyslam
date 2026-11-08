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
#include "slam_direct/direct_params.h"

namespace pyslam {

// Latest attempted solve; graph counts may be partial when applied is false.
struct DirectBAResult {
    bool applied = false; // True only after validated estimates have been written to the map.
    int keyframes = 0;
    int landmarks = 0;
    int observations = 0; // Number of residual edges, not unique landmark/keyframe pairs.
    int outliers = 0; // Rejected left, right, or depth edges in the applied solve.
    double initial_cost = 0.0; // Sum of robust chi-square costs over all graph edges.
    double final_cost = 0.0; // Same objective, including edges excluded during refinement.
};

// Bounded, synchronous local BA. Like vslam, jointly optimize poses and shared
// landmarks from left/right observations. Fix the oldest window pose for gauge.
DirectBAResult optimize_direct_map(DirectMap &map, const Camera &camera,
                                   const DirectSlamParams &params);

} // namespace pyslam

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

#include "frame.h"

#include <vector>

namespace pyslam {

// SLAM Dynamic Config - simplified version
class SLAMDynamicConfig {
  public:
    double max_descriptor_distance;

    SLAMDynamicConfig(double max_des_dist) : max_descriptor_distance(max_des_dist) {}

    double update_descriptor_stats(const FramePtr &f_ref, const FramePtr &f_cur,
                                   const std::vector<int> &idxs_ref,
                                   const std::vector<int> &idxs_cur) {
        // Simplified implementation - would compute dynamic statistics
        return max_descriptor_distance;
    }
};

} // namespace pyslam
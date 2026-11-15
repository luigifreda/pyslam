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

#include "common/eigen_aliases.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace pyslam {

// A point carried from the last accepted image (which may precede tracking failures). position_w is in the world frame.
struct LivePoint {
    int64_t id = -1;
    Vec2d uv = Vec2d::Zero();
    Vec2d uv_ref = Vec2d::Zero(); // Pixel on reference_keyframe. The current uv feeds trails.
    int reference_keyframe = -1;
    Vec3d position_w = Vec3d::Zero();
    std::vector<cv::Point> trail; // Newest first, capped at 64 accepted observations.
    int observations = 0;
};

} // namespace pyslam

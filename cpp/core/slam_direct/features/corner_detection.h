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
#include "slam_direct/direct_params.h"

#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Corner detection shared by every tracking mode (klt, hybrid, semi_direct).

// `occupied` is optional. Zero pixels are excluded, on top of the border mask.
// `max_corners` <= 0 uses params.num_features.
std::vector<Vec2d> detect_gftt(const cv::Mat &gray, const DirectSlamParams &params,
                               const cv::Mat &occupied = cv::Mat(), int max_corners = -1);

// Dispatches to GFTT or grid FAST according to params.corner_detector.
std::vector<Vec2d> detect_corners(const cv::Mat &gray, const DirectSlamParams &params,
                                  const cv::Mat &occupied = cv::Mat(), int max_corners = -1);

} // namespace pyslam

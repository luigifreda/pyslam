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

#include "eigen_aliases.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

namespace pyslam {

// Helper function for subpixel stereo matching
std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
stereo_match_subpixel_correlation(const std::vector<int> &idxs1, const std::vector<int> &idxs2,
                                  const MatNx2dRef kps, const MatNx2dRef kps_r,
                                  double min_disparity, double max_disparity,
                                  const cv::Mat &img_left, const cv::Mat &img_right);

} // namespace pyslam
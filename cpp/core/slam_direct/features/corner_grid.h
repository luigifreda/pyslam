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

#include <utility>
#include <vector>

namespace pyslam {

// Uniform FAST extraction. The image is split into a regular cell grid and each
// cell contributes its strongest corners, up to a shared per-cell quota. At most
// `num_features` corners are returned (the strongest ones across all cells).
// `cells` empty means every cell. A non-empty list restricts extraction to those
// (column, row) cells and is not resized to fit `num_features`.
// `allowed` uses the same convention as goodFeaturesToTrack: empty allows the
// whole image, and a CV_8UC1 mask keeps only nonzero pixels.
struct CornerGridRequest {
    int num_features = 200;
    int grid_cols = 5;
    int grid_rows = 5;
    int fast_threshold = 10;
    bool nonmax_suppression = true;
    int border = 0;
    bool refine_subpixel = true;
    int subpixel_window = 5;
    std::vector<std::pair<int, int>> cells;
};

std::vector<Vec2d> detect_grid_fast(const cv::Mat &gray, const CornerGridRequest &request,
                                    const cv::Mat &allowed = cv::Mat());

} // namespace pyslam

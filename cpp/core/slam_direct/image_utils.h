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
#include "slam_direct/direct_params.h"

#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Image helpers shared by the direct, klt and features modules. All images are CV_8UC1.

// Throws std::invalid_argument("<name> must be a non-empty CV_8UC1 image") otherwise.
void require_gray(const cv::Mat &img, const char *name);

// Bilinear gray value. Coordinates are clamped to [0, size - 2], so every sample reads one
// full 2x2 cell and a sample in the last column/row strip reads column/row size - 2.
// A 1-pixel-wide image returns its first pixel; an empty image returns 0.
// Used by the planar patches and the depth-filter epipolar search.
float sample_gray(const cv::Mat &img, float x, float y);

// Bilinear gray value of the photometric aligner. Same as sample_gray, except that
// a sample inside the last strip (size - 2, size - 1) still interpolates toward the
// last column/row; only coordinates >= size - 1 snap to size - 2. Kept separate
// because merging the two would change hybrid's gradients at the right/bottom edge.
float sample_gray_aligner(const cv::Mat &img, float x, float y);

// Level 0 is the input image (shared, not copied). Each further level is cv::resize'd
// (INTER_LINEAR) from the previous one by params.pyramid_scale, never below 1x1.
// scales[l] maps a full-resolution pixel to level l.
struct ImagePyramid {
    std::vector<cv::Mat> levels;
    std::vector<double> scales;
};
ImagePyramid build_pyramid(const cv::Mat &image, const DirectSlamParams &params);

// True when uv lies in [border, cols - border) x [border, rows - border). NaN is outside.
inline bool inside_border(const Vec2d &uv, const cv::Size &size, int border) {
    return uv.x() >= border && uv.y() >= border && uv.x() < size.width - border && uv.y() < size.height - border;
}

} // namespace pyslam

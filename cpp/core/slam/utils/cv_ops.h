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

#include <Eigen/Core>

#include "eigen_aliases.h"

#include <algorithm>
#include <limits>

#include <opencv2/core/core.hpp>
#include <stdexcept>

namespace pyslam {

// Compute the mean color of a patch in an image
inline cv::Vec3f compute_patch_mean(const cv::Mat &img, int u, int v, int delta,
                                    const cv::Vec3f &default_color) {
    if (img.empty()) {
        return default_color;
    }

    const int cols = img.cols;
    const int rows = img.rows;
    const int size = 2 * delta + 1;

    if (u - delta < 0 || v - delta < 0 || (u + delta) >= cols || (v + delta) >= rows) {
        return default_color;
    }

    const cv::Rect roi(u - delta, v - delta, size, size);
    const cv::Scalar mean_scalar = cv::mean(img(roi));
    if (img.channels() == 1) {
        const float val = static_cast<float>(mean_scalar[0]);
        return {val, val, val};
    }
    return {static_cast<float>(mean_scalar[0]), static_cast<float>(mean_scalar[1]),
            static_cast<float>(mean_scalar[2])};
}

// Convert a cv::Vec3f to a Vec3b
inline Vec3b to_color_vector(const cv::Vec3f &color) {
    Vec3b result;
    for (int c = 0; c < 3; ++c) {
        float value = std::clamp(color[c], 0.0f, 255.0f);
        result(c) = static_cast<unsigned char>(std::lround(value));
    }
    return result;
}

} // namespace pyslam
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

#include "utils/color_helpers.h"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace pyslam {

// Shared by Frame and DirectSlam: ID colors, antialiased trails, filled markers,
// and the optional octave-radius/track-age palette must stay identical.
template <bool with_level_radius>
inline void draw_feature_trail(cv::Mat &image, const cv::Point &uv, uint64_t id,
                                const std::vector<cv::Point> &trail, size_t observations,
                                int octave = 0) {
    cv::Scalar point_color;
    if constexpr (with_level_radius) {
        point_color = observations > 2 ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
    } else {
        const auto color = ColorTableGenerator::instance().color_from_int(id);
        point_color = cv::Scalar(color.r, color.g, color.b);
    }
    if (trail.size() > 1) {
        cv::Scalar trail_color = point_color;
        if constexpr (with_level_radius) {
            const auto color = Colors::myjet_color_x_255(static_cast<int>(trail.size()));
            trail_color = cv::Scalar(color[0], color[1], color[2]);
        }
        cv::polylines(image, trail, false, trail_color, 1, cv::LINE_AA);
    }
    if constexpr (with_level_radius)
        cv::circle(image, uv, octave * 5, point_color, 1);
    else
        cv::circle(image, uv, 4, point_color, -1);
}

} // namespace pyslam

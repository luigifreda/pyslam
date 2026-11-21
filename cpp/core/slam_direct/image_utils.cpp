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

#include "slam_direct/image_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pyslam {

void require_gray(const cv::Mat &img, const char *name) {
    if (img.empty() || img.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty CV_8UC1 image");
    }
}

float sample_gray(const cv::Mat &img, float x, float y) {
    if (img.cols < 2 || img.rows < 2) {
        return img.empty() ? 0.0f : static_cast<float>(img.at<uchar>(0, 0));
    }
    x = std::clamp(x, 0.0f, static_cast<float>(img.cols - 2));
    y = std::clamp(y, 0.0f, static_cast<float>(img.rows - 2));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float i00 = static_cast<float>(img.at<uchar>(y0, x0));
    const float i10 = static_cast<float>(img.at<uchar>(y0, x0 + 1));
    const float i01 = static_cast<float>(img.at<uchar>(y0 + 1, x0));
    const float i11 = static_cast<float>(img.at<uchar>(y0 + 1, x0 + 1));
    return (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 +
           fx * fy * i11;
}

float sample_gray_aligner(const cv::Mat &img, float x, float y) {
    if (img.cols < 2 || img.rows < 2) {
        return img.empty() ? 0.0f : static_cast<float>(img.at<uchar>(0, 0));
    }
    if (x < 0.0f) {
        x = 0.0f;
    }
    if (y < 0.0f) {
        y = 0.0f;
    }
    if (x >= static_cast<float>(img.cols - 1)) {
        x = static_cast<float>(img.cols - 2);
    }
    if (y >= static_cast<float>(img.rows - 1)) {
        y = static_cast<float>(img.rows - 2);
    }
    const float xx = x - std::floor(x);
    const float yy = y - std::floor(y);
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(img.cols - 1, x0 + 1);
    const int y1 = std::min(img.rows - 1, y0 + 1);
    const float i00 = static_cast<float>(img.at<uchar>(y0, x0));
    const float i10 = static_cast<float>(img.at<uchar>(y0, x1));
    const float i01 = static_cast<float>(img.at<uchar>(y1, x0));
    const float i11 = static_cast<float>(img.at<uchar>(y1, x1));
    return (1.0f - xx) * (1.0f - yy) * i00 + xx * (1.0f - yy) * i10 + (1.0f - xx) * yy * i01 +
           xx * yy * i11;
}

ImagePyramid build_pyramid(const cv::Mat &image, const DirectSlamParams &params) {
    ImagePyramid pyramid;
    pyramid.levels.reserve(static_cast<size_t>(std::max(0, params.num_pyramid_levels)));
    pyramid.scales.reserve(static_cast<size_t>(std::max(0, params.num_pyramid_levels)));
    for (int level = 0; level < params.num_pyramid_levels; ++level) {
        if (level == 0) {
            pyramid.levels.push_back(image);
            pyramid.scales.push_back(1.0);
            continue;
        }
        const cv::Mat &previous = pyramid.levels.back();
        const cv::Size size(std::max(1, static_cast<int>(std::lround(previous.cols * params.pyramid_scale))),
                            std::max(1, static_cast<int>(std::lround(previous.rows * params.pyramid_scale))));
        cv::Mat resized;
        cv::resize(previous, resized, size, 0.0, 0.0, cv::INTER_LINEAR);
        pyramid.levels.push_back(resized);
        pyramid.scales.push_back(pyramid.scales.back() * params.pyramid_scale);
    }
    return pyramid;
}

} // namespace pyslam

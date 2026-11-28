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

#include "slam/direct/features/corner_detection.h"

#include "slam/direct/features/corner_grid.h"
#include "slam/direct/core/image_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

std::vector<Vec2d> from_cv(const std::vector<cv::Point2f> &points) {
    std::vector<Vec2d> out;
    out.reserve(points.size());
    for (const cv::Point2f &point : points) {
        out.emplace_back(point.x, point.y);
    }
    return out;
}

// A border wider than a quarter of the image would leave nothing to detect.
int effective_border(const cv::Mat &gray, const DirectSlamParams &params) {
    return std::max(0, std::min(params.border, std::min(gray.cols, gray.rows) / 4));
}

// Detection mask shared by GFTT and grid FAST: border exclusion combined with the
// caller's occupancy mask. An occupancy mask of the wrong size or type is ignored
// by both detectors, independently of the border setting.
cv::Mat detection_mask(const cv::Mat &gray, int border, const cv::Mat &occupied) {
    cv::Mat mask(gray.size(), CV_8U, cv::Scalar(255));
    if (border > 0) {
        mask.rowRange(0, border).setTo(0);
        mask.rowRange(gray.rows - border, gray.rows).setTo(0);
        mask.colRange(0, border).setTo(0);
        mask.colRange(gray.cols - border, gray.cols).setTo(0);
    }
    if (!occupied.empty() && occupied.size() == gray.size() && occupied.type() == CV_8UC1) {
        cv::bitwise_and(mask, occupied, mask);
    }
    return mask;
}

} // namespace

std::vector<Vec2d> detect_gftt(const cv::Mat &gray, const DirectSlamParams &params,
                               const cv::Mat &occupied, int max_corners) {
    require_gray(gray, "detect_gftt");
    const int corners_to_find = max_corners > 0 ? max_corners : params.num_features;
    if (corners_to_find < 1) {
        return {};
    }
    // Preserve existing tracks by combining their occupancy mask with border exclusion.
    cv::Mat mask = detection_mask(gray, effective_border(gray, params), occupied);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, corners_to_find, params.gftt_quality, params.gftt_min_distance,
                            mask, params.gftt_block_size, false, 0.04);
    return from_cv(corners);
}

std::vector<Vec2d> detect_corners(const cv::Mat &gray, const DirectSlamParams &params,
                                  const cv::Mat &occupied, int max_corners) {
    if (params.corner_detector == CornerDetector::Gftt) {
        return detect_gftt(gray, params, occupied, max_corners);
    }
    CornerGridRequest request;
    request.num_features = max_corners > 0 ? max_corners : params.num_features;
    request.grid_cols = params.grid_cols;
    request.grid_rows = params.grid_rows;
    request.fast_threshold = params.fast_threshold;
    request.nonmax_suppression = params.fast_nonmax_suppression;
    // Use the same clamped border for the mask and the per-corner check.
    request.border = effective_border(gray, params);
    request.refine_subpixel = params.corner_subpix;
    request.subpixel_window = params.subpixel_window;
    return detect_grid_fast(gray, request, detection_mask(gray, request.border, occupied));
}

OccupancyMask::OccupancyMask(const cv::Size &size, const DirectSlamParams &params)
    : mask_(size, CV_8U, cv::Scalar(255)),
      radius_(std::max(1, static_cast<int>(std::lround(params.gftt_min_distance)))) {}

void OccupancyMask::block(const Vec2d &uv) {
    cv::circle(mask_, cv::Point(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y()))),
               radius_, cv::Scalar(0), cv::FILLED);
}

} // namespace pyslam

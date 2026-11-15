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

#include "slam_direct/klt/klt_tracker.h"

#include <opencv2/video/tracking.hpp>

#include <stdexcept>

namespace pyslam {
namespace {

std::vector<cv::Point2f> to_cv(const std::vector<Vec2d> &points) {
    std::vector<cv::Point2f> out;
    out.reserve(points.size());
    for (const Vec2d &point : points) {
        out.emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()));
    }
    return out;
}

void require_gray(const cv::Mat &img, const char *name) {
    if (img.empty() || img.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty CV_8UC1 image");
    }
}

} // namespace

LkTrack track_pyramidal_lk(const cv::Mat &img_ref, const cv::Mat &img_cur,
                           const std::vector<Vec2d> &points_ref,
                           const std::vector<Vec2d> &points_cur_init, const DirectSlamParams &params) {
    require_gray(img_ref, "track_pyramidal_lk img_ref");
    require_gray(img_cur, "track_pyramidal_lk img_cur");
    LkTrack tracked;
    tracked.points_cur.resize(points_ref.size(), Vec2d::Zero());
    tracked.status.assign(points_ref.size(), 0);
    if (points_ref.empty()) {
        return tracked;
    }
    std::vector<Vec2d> initial = points_cur_init.size() == points_ref.size() ? points_cur_init : points_ref;
    std::vector<cv::Point2f> prev = to_cv(points_ref);
    std::vector<cv::Point2f> curr = to_cv(initial);
    std::vector<uchar> status;
    std::vector<float> error;
    const cv::Size window(params.lk_patch_size, params.lk_patch_size);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img_ref, img_cur, prev, curr, status, error, window, params.lk_max_level, criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    // A forward-backward displacement check rejects plausible-looking but unstable matches.
    std::vector<cv::Point2f> backward = prev;
    std::vector<uchar> backward_status;
    cv::calcOpticalFlowPyrLK(img_cur, img_ref, curr, backward, backward_status, error, window,
                             params.lk_max_level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    const int border = params.border;
    constexpr float kForwardBackwardPx = 1.5f;
    for (size_t i = 0; i < status.size(); ++i) {
        const float dx = backward[i].x - prev[i].x;
        const float dy = backward[i].y - prev[i].y;
        const bool consistent = backward_status[i] && dx * dx + dy * dy <= kForwardBackwardPx * kForwardBackwardPx;
        const bool in_image = curr[i].x >= border && curr[i].y >= border &&
                              curr[i].x < img_cur.cols - border && curr[i].y < img_cur.rows - border;
        tracked.status[i] = status[i] && consistent && in_image ? 1 : 0;
        tracked.points_cur[i] = Vec2d(curr[i].x, curr[i].y);
    }
    return tracked;
}

} // namespace pyslam

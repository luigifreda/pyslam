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

#include "slam_direct/features/corner_grid.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

struct GridShape {
    int cols = 1;
    int rows = 1;
};

GridShape fitted_grid(const CornerGridRequest &request) {
    int cols = std::max(1, request.grid_cols);
    int rows = std::max(1, request.grid_rows);
    // A full-image request with more cells than corners would leave most cells empty.
    // Shrink the grid, keeping its aspect ratio, so each remaining cell can take a corner.
    // Round down so the fitted grid never has more cells than requested corners.
    if (request.cells.empty() && request.num_features < cols * rows) {
        const double ratio = static_cast<double>(cols) / static_cast<double>(rows);
        rows = std::max(1, static_cast<int>(std::floor(std::sqrt(request.num_features / ratio))));
        cols = std::max(1, std::min(request.num_features / rows,
                                    static_cast<int>(std::floor(static_cast<double>(rows) * ratio))));
    }
    return {cols, rows};
}

bool pixel_allowed(const cv::Mat &allowed, int x, int y) {
    return allowed.empty() || allowed.at<uchar>(y, x) != 0;
}

// Same acceptance test for raw FAST pixels and for sub-pixel refined corners.
bool corner_allowed(const cv::Mat &gray, const cv::Mat &allowed, int border, float px, float py) {
    const int x = static_cast<int>(std::lround(px));
    const int y = static_cast<int>(std::lround(py));
    if (x < border || y < border || x >= gray.cols - border || y >= gray.rows - border) {
        return false;
    }
    return pixel_allowed(allowed, x, y);
}

} // namespace

std::vector<Vec2d> detect_grid_fast(const cv::Mat &gray, const CornerGridRequest &request,
                                    const cv::Mat &allowed) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::invalid_argument("detect_grid_fast: image must be CV_8UC1");
    }
    if (!allowed.empty() && (allowed.size() != gray.size() || allowed.type() != CV_8UC1)) {
        throw std::invalid_argument("detect_grid_fast: allowed mask must be empty or CV_8UC1 of the image size");
    }
    if (request.num_features < 1) {
        return {};
    }

    const GridShape grid = fitted_grid(request);
    const int border = std::max(0, request.border);
    std::vector<std::pair<int, int>> cells = request.cells;
    if (cells.empty()) {
        cells.reserve(static_cast<size_t>(grid.cols * grid.rows));
        for (int row = 0; row < grid.rows; ++row) {
            for (int col = 0; col < grid.cols; ++col) {
                cells.emplace_back(col, row);
            }
        }
    }
    // Spread the budget over the cells actually searched. The rounded-up quota
    // can overshoot num_features; the strongest-first cap below removes the excess.
    const int num_cells = std::max<int>(1, static_cast<int>(cells.size()));
    const int quota = std::max(1, (request.num_features + num_cells - 1) / num_cells);

    std::vector<cv::KeyPoint> selected;
    selected.reserve(static_cast<size_t>(quota * cells.size()));
    for (const std::pair<int, int> &cell : cells) {
        const int col = cell.first;
        const int row = cell.second;
        if (col < 0 || row < 0 || col >= grid.cols || row >= grid.rows) {
            continue;
        }
        const int x0 = col * gray.cols / grid.cols;
        const int x1 = (col + 1) * gray.cols / grid.cols;
        const int y0 = row * gray.rows / grid.rows;
        const int y1 = (row + 1) * gray.rows / grid.rows;
        // FAST reads a radius-3 circle, so a smaller cell cannot host a corner.
        if (x1 - x0 < 7 || y1 - y0 < 7) {
            continue;
        }
        std::vector<cv::KeyPoint> found;
        cv::FAST(gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)), found, request.fast_threshold,
                 request.nonmax_suppression);
        std::sort(found.begin(), found.end(),
                  [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
        int kept = 0;
        for (const cv::KeyPoint &keypoint : found) {
            if (kept >= quota) {
                break;
            }
            if (!corner_allowed(gray, allowed, border, keypoint.pt.x + static_cast<float>(x0),
                                keypoint.pt.y + static_cast<float>(y0))) {
                continue;
            }
            cv::KeyPoint placed = keypoint;
            placed.pt.x += static_cast<float>(x0);
            placed.pt.y += static_cast<float>(y0);
            selected.push_back(placed);
            ++kept;
        }
    }

    // num_features is a hard maximum: keep the strongest corners across all cells.
    if (selected.size() > static_cast<size_t>(request.num_features)) {
        std::nth_element(selected.begin(), selected.begin() + request.num_features, selected.end(),
                         [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
        selected.resize(static_cast<size_t>(request.num_features));
    }

    std::vector<cv::Point2f> points;
    points.reserve(selected.size());
    for (const cv::KeyPoint &keypoint : selected) {
        points.push_back(keypoint.pt);
    }
    if (request.refine_subpixel && !points.empty()) {
        const int window = std::max(3, request.subpixel_window | 1);
        const int half = window / 2;
        std::vector<cv::Point2f> refinable;
        std::vector<int> refinable_index;
        refinable.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const cv::Point2f &point = points[i];
            if (point.x >= half && point.y >= half && point.x < gray.cols - half && point.y < gray.rows - half) {
                refinable.push_back(point);
                refinable_index.push_back(static_cast<int>(i));
            }
        }
        if (!refinable.empty()) {
            // cornerSubPix takes the half-window: Size(2, 2) searches a 5x5 neighborhood.
            cv::cornerSubPix(gray, refinable, cv::Size(half, half), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));
            for (size_t i = 0; i < refinable.size(); ++i) {
                // Refinement can move a corner by up to half a window. Keep the
                // validated FAST location if it would leave the border or the mask.
                const cv::Point2f &refined = refinable[i];
                if (std::isfinite(refined.x) && std::isfinite(refined.y) &&
                    corner_allowed(gray, allowed, border, refined.x, refined.y)) {
                    points[refinable_index[i]] = refined;
                }
            }
        }
    }

    std::vector<Vec2d> result;
    result.reserve(points.size());
    for (const cv::Point2f &point : points) {
        result.emplace_back(point.x, point.y);
    }
    return result;
}

} // namespace pyslam

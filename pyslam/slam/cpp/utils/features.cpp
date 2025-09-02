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

#include "utils/features.h"

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
                                  const cv::Mat &img_left, const cv::Mat &img_right) {
    constexpr int w = 5; // Window size
    constexpr int l = 5; // Half window size for disparity search

    std::vector<double> disparities(idxs1.size(), -1.0);
    std::vector<double> us_right(idxs1.size(), -1.0);
    std::vector<double> distances;
    std::vector<int> valid_indices;

    int img_h = img_left.rows;
    int img_w = img_left.cols;

    for (size_t i = 0; i < idxs1.size(); ++i) {
        int u_l = static_cast<int>(std::round(kps(idxs1[i], 0)));
        int v_l = static_cast<int>(std::round(kps(idxs1[i], 1)));
        int u_r0 = static_cast<int>(std::round(kps_r(idxs2[i], 0)));

        // Check bounds
        if (!(w <= v_l && v_l < img_h - w && w <= u_l && u_l < img_w - w)) {
            continue;
        }

        // Extract left patch
        cv::Mat left_patch = img_left(cv::Rect(u_l - w, v_l - w, 2 * w + 1, 2 * w + 1));
        cv::Mat left_patch_float;
        left_patch.convertTo(left_patch_float, CV_32F);
        float center_val = left_patch_float.at<float>(w, w);
        left_patch_float -= center_val;

        double best_dist = 1e10;
        int best_inc_r = 0;
        std::vector<double> dists(2 * l + 1, 1e10);

        // Search for best match
        for (int inc_r = -l; inc_r <= l; ++inc_r) {
            int start_u_r = u_r0 + inc_r - w;
            int end_u_r = u_r0 + inc_r + w + 1;

            if (start_u_r < 0 || end_u_r > img_w) {
                continue;
            }

            cv::Mat right_patch = img_right(cv::Rect(start_u_r, v_l - w, 2 * w + 1, 2 * w + 1));
            cv::Mat right_patch_float;
            right_patch.convertTo(right_patch_float, CV_32F);
            float center_val_r = right_patch_float.at<float>(w, w);
            right_patch_float -= center_val_r;

            // Compute SAD
            cv::Mat diff = left_patch_float - right_patch_float;
            cv::Mat abs_diff;
            cv::absdiff(left_patch_float, right_patch_float, abs_diff);
            double dist = cv::sum(abs_diff)[0];

            dists[l + inc_r] = dist;

            if (dist < best_dist) {
                best_dist = dist;
                best_inc_r = inc_r;
            }
        }

        // Check if best match is at boundary
        if (best_inc_r == -l || best_inc_r == l) {
            continue;
        }

        // Subpixel refinement
        int idx = l + best_inc_r;
        if (idx - 1 < 0 || idx + 1 >= static_cast<int>(dists.size())) {
            continue;
        }

        double dist1 = dists[idx - 1];
        double dist2 = dists[idx];
        double dist3 = dists[idx + 1];
        double denom = 2.0 * (dist1 + dist3 - 2.0 * dist2);

        if (std::abs(denom) < 1e-10) {
            continue;
        }

        double delta_r = (dist1 - dist3) / denom;
        if (delta_r < -1.0 || delta_r > 1.0) {
            continue;
        }

        double best_u_r = u_r0 + best_inc_r + delta_r;
        double disparity = kps(idxs1[i], 0) - best_u_r;

        if (disparity >= min_disparity && disparity < max_disparity) {
            if (disparity <= 0) {
                disparity = 0.01;
                best_u_r = kps(idxs1[i], 0) - 0.01;
            }

            disparities[i] = disparity;
            us_right[i] = best_u_r;
            distances.push_back(best_dist);
            valid_indices.push_back(i);
        }
    }

    // Filter by distance using MAD (Median Absolute Deviation)
    std::vector<int> final_valid_idxs;
    if (!distances.empty()) {
        std::vector<double> sorted_distances = distances;
        std::sort(sorted_distances.begin(), sorted_distances.end());

        double median_dist = sorted_distances[sorted_distances.size() / 2];
        double threshold_dist = 1.5 * 1.4826 * median_dist;

        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances[i] < threshold_dist) {
                final_valid_idxs.push_back(valid_indices[i]);
            } else {
                // Mark as invalid
                int idx = valid_indices[i];
                us_right[idx] = -1.0;
                disparities[idx] = -1.0;
            }
        }
    }

    return std::make_tuple(disparities, us_right, final_valid_idxs);
}

} // namespace pyslam
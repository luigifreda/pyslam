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
#include "utils/messages.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

namespace pyslam {
// Helper function for subpixel stereo matching
template <typename Scalar>
std::tuple<std::vector<Scalar>, std::vector<Scalar>, std::vector<int>>
stereo_match_subpixel_correlation(const std::vector<int> &idxs1, const std::vector<int> &idxs2,
                                  const MatNx2Ref<Scalar> kps, const MatNx2Ref<Scalar> kps_r,
                                  Scalar min_disparity, Scalar max_disparity, // Changed to Scalar
                                  const cv::Mat &img_left, const cv::Mat &img_right) {
    constexpr int w = 5; // Window size
    constexpr int l = 5; // Half window size for disparity search

    std::vector<Scalar> disparities(idxs1.size(), static_cast<Scalar>(-1.0));
    std::vector<Scalar> us_right(idxs1.size(), static_cast<Scalar>(-1.0));
    std::vector<Scalar> distances;
    std::vector<int> valid_indices;

    int img_h = img_left.rows;
    int img_w = img_left.cols;

    // Validate input sizes
    if (idxs1.size() != idxs2.size()) {
        MSG_ERROR("stereo_match_subpixel_correlation() - idxs1.size() != idxs2.size()");
        return std::make_tuple(disparities, us_right, std::vector<int>());
    }

    const size_t kps_rows = kps.rows();
    const size_t kps_r_rows = kps_r.rows();

    for (size_t i = 0; i < idxs1.size(); ++i) {
        // Add bounds checking for indices
        if (idxs1[i] < 0 || idxs1[i] >= kps_rows || idxs2[i] < 0 || idxs2[i] >= kps_r_rows) {
            continue;
        }

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

        Scalar best_dist = std::numeric_limits<Scalar>::max();
        int best_inc_r = 0;
        std::vector<Scalar> dists(2 * l + 1, static_cast<Scalar>(1e10)); // Use Scalar

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
            Scalar dist = static_cast<Scalar>(cv::sum(abs_diff)[0]); // Explicit cast

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

        Scalar dist1 = dists[idx - 1];
        Scalar dist2 = dists[idx];
        Scalar dist3 = dists[idx + 1];
        Scalar denom =
            static_cast<Scalar>(2.0) * (dist1 + dist3 - static_cast<Scalar>(2.0) * dist2);

        // Use std::abs for integral types, std::fabs for floating point
        if constexpr (std::is_floating_point_v<Scalar>) {
            if (std::fabs(denom) < static_cast<Scalar>(1e-10)) {
                continue;
            }
        } else {
            if (std::abs(denom) < static_cast<Scalar>(1e-10)) {
                continue;
            }
        }

        Scalar delta_r = (dist1 - dist3) / denom;
        if (delta_r < static_cast<Scalar>(-1.0) || delta_r > static_cast<Scalar>(1.0)) {
            continue;
        }

        Scalar best_u_r = static_cast<Scalar>(u_r0 + best_inc_r) + delta_r;
        Scalar disparity = kps(idxs1[i], 0) - best_u_r;

        if (disparity >= min_disparity && disparity < max_disparity) {
            if (disparity <= static_cast<Scalar>(0)) {
                disparity = static_cast<Scalar>(0.01);
                best_u_r = kps(idxs1[i], 0) - disparity;
            }

            disparities[i] = disparity;
            us_right[i] = best_u_r;
            distances.push_back(best_dist);
            valid_indices.push_back(i);
        }
    }

    // Filter by distance using MAD (Median Absolute Deviation)
    std::vector<int> final_valid_idxs;
    if (!distances.empty() && distances.size() > 2) {
        std::vector<Scalar> sorted_distances = distances;
        std::sort(sorted_distances.begin(), sorted_distances.end());

        Scalar median_dist;
        const size_t size = sorted_distances.size();
        if (size % 2 == 0 && size > 0) {
            // Even number: average of two middle values
            median_dist = (sorted_distances[size / 2 - 1] + sorted_distances[size / 2]) /
                          static_cast<Scalar>(2.0);
        } else {
            // Odd number: middle value
            median_dist = sorted_distances[size / 2];
        }
        Scalar threshold_dist =
            static_cast<Scalar>(1.5) * static_cast<Scalar>(1.4826) * median_dist;

        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances[i] < threshold_dist) {
                final_valid_idxs.push_back(valid_indices[i]);
            } else {
                // Mark as invalid
                int idx = valid_indices[i];
                us_right[idx] = static_cast<Scalar>(-1.0);
                disparities[idx] = static_cast<Scalar>(-1.0);
            }
        }
    }

    return std::make_tuple(disparities, us_right, final_valid_idxs);
}

// Explicit template instantiation for stereo_match_subpixel_correlation method used in Python
// bindings
template std::tuple<std::vector<float>, std::vector<float>, std::vector<int>>
pyslam::stereo_match_subpixel_correlation<float>(const std::vector<int> &idxs1,
                                                 const std::vector<int> &idxs2,
                                                 const MatNx2Ref<float> kps,
                                                 const MatNx2Ref<float> kps_r, float min_disparity,
                                                 float max_disparity, const cv::Mat &img_left,
                                                 const cv::Mat &img_right);

template std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
pyslam::stereo_match_subpixel_correlation<double>(
    const std::vector<int> &idxs1, const std::vector<int> &idxs2, const MatNx2Ref<double> kps,
    const MatNx2Ref<double> kps_r, double min_disparity, double max_disparity,
    const cv::Mat &img_left, const cv::Mat &img_right);

} // namespace pyslam
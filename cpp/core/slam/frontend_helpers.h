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

#include "feature_shared_resources.h"
#include "utils/descriptor_helpers.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
#endif

namespace pyslam {

inline float median_of(std::vector<float> values) {
    if (values.empty()) {
        return 0.f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    const float upper = values[mid];
    const float lower = *std::max_element(values.begin(), values.begin() + mid);
    return 0.5f * (upper + lower);
}

inline std::vector<float> descriptor_distances_for_matches(const cv::Mat &des1,
                                                           const std::vector<int> &idxs1,
                                                           const cv::Mat &des2,
                                                           const std::vector<int> &idxs2) {
    std::vector<float> distances;
    const size_t n = std::min(idxs1.size(), idxs2.size());
    distances.reserve(n);
    const NormType norm = FeatureSharedResources::norm_type;
    for (size_t i = 0; i < n; ++i) {
        distances.push_back(
            descriptor_distance(des1.row(idxs1[i]), des2.row(idxs2[i]), norm));
    }
    return distances;
}

inline std::pair<float, float> descriptor_sigma_mad(const std::vector<float> &distances,
                                                    bool use_v2) {
    if (distances.empty()) {
        return {0.f, 0.f};
    }
    const float dists_median = median_of(distances);
    if (!use_v2) {
        return {1.4826f * dists_median, dists_median};
    }
    std::vector<float> ads;
    ads.reserve(distances.size());
    for (float d : distances) {
        ads.push_back(std::fabs(d - dists_median));
    }
    return {1.4826f * median_of(ads), dists_median};
}

inline cv::Mat descriptor_rows(const cv::Mat &des, const std::vector<int> &idxs) {
    if (des.empty() || idxs.empty()) {
        return cv::Mat();
    }
    cv::Mat out(static_cast<int>(idxs.size()), des.cols, des.type());
    for (int i = 0; i < static_cast<int>(idxs.size()); ++i) {
        des.row(idxs[i]).copyTo(out.row(i));
    }
    return out;
}

inline MatNx2f keypoint_rows(const MatNx2f &kps, const std::vector<int> &idxs) {
    MatNx2f out(static_cast<int>(idxs.size()), 2);
    for (int i = 0; i < static_cast<int>(idxs.size()); ++i) {
        out.row(i) = kps.row(idxs[i]);
    }
    return out;
}

inline std::vector<int> apply_match_indices(const std::vector<int> &idxs,
                                            const std::vector<int> &valid) {
    std::vector<int> out;
    out.reserve(valid.size());
    for (int k : valid) {
        if (k >= 0 && k < static_cast<int>(idxs.size())) {
            out.push_back(idxs[k]);
        }
    }
    return out;
}

inline FeatureMatchingOutput match_descriptors(const cv::Mat &img1, const cv::Mat &img2,
                                               const cv::Mat &des1, const cv::Mat &des2,
                                               const MatNx2f &kps1, const MatNx2f &kps2,
                                               float ratio_test) {
    if (!FeatureSharedResources::feature_matching_callback || des1.empty() || des2.empty()) {
        return {};
    }
#ifdef USE_PYTHON
    pybind11::gil_scoped_acquire gil;
#endif
    return FeatureSharedResources::feature_matching_callback(img1, img2, des1, des2, kps1, kps2,
                                                             ratio_test, false, 0.f);
}

// cv2.triangulatePoints on normalized coordinates. Returns every triangulated point and a
// boolean mask of the points whose homogeneous w is not zero.
inline std::pair<std::vector<Eigen::Vector3d>, std::vector<bool>>
triangulate_normalized_points(const Eigen::Matrix4d &pose_1w, const Eigen::Matrix4d &pose_2w,
                              const MatNx2f &kpn_1, const MatNx2f &kpn_2) {
    const int n = std::min(kpn_1.rows(), kpn_2.rows());
    std::vector<Eigen::Vector3d> points(static_cast<size_t>(n), Eigen::Vector3d::Zero());
    std::vector<bool> mask(static_cast<size_t>(n), false);
    if (n <= 0) {
        return {points, mask};
    }
    cv::Mat P1(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            P1.at<double>(r, c) = pose_1w(r, c);
            P2.at<double>(r, c) = pose_2w(r, c);
        }
    }
    cv::Mat pts1(2, n, CV_64F);
    cv::Mat pts2(2, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        pts1.at<double>(0, i) = kpn_1(i, 0);
        pts1.at<double>(1, i) = kpn_1(i, 1);
        pts2.at<double>(0, i) = kpn_2(i, 0);
        pts2.at<double>(1, i) = kpn_2(i, 1);
    }
    cv::Mat hom;
    cv::triangulatePoints(P1, P2, pts1, pts2, hom);
    for (int i = 0; i < n; ++i) {
        const double w = hom.at<double>(3, i);
        if (w == 0.0) {
            continue;
        }
        points[static_cast<size_t>(i)] = Eigen::Vector3d(hom.at<double>(0, i) / w,
                                                         hom.at<double>(1, i) / w,
                                                         hom.at<double>(2, i) / w);
        mask[static_cast<size_t>(i)] = true;
    }
    return {points, mask};
}

inline std::vector<int> indices_where(const std::vector<bool> &mask) {
    std::vector<int> idxs;
    for (int i = 0; i < static_cast<int>(mask.size()); ++i) {
        if (mask[static_cast<size_t>(i)]) {
            idxs.push_back(i);
        }
    }
    return idxs;
}

inline std::vector<int> select_indices(const std::vector<int> &idxs, const std::vector<bool> &mask) {
    std::vector<int> out;
    const int n = std::min(static_cast<int>(idxs.size()), static_cast<int>(mask.size()));
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (mask[static_cast<size_t>(i)]) {
            out.push_back(idxs[static_cast<size_t>(i)]);
        }
    }
    return out;
}

class ImageGrid {
  public:
    ImageGrid(int width, int height, int num_div_x, int num_div_y)
        : width_(std::max(width, 1)), height_(std::max(height, 1)), num_div_x_(std::max(num_div_x, 1)),
          num_div_y_(std::max(num_div_y, 1)),
          counts_(static_cast<size_t>(num_div_x_ * num_div_y_), 0) {}

    void add_points(const MatNx2f &kps) {
        for (int i = 0; i < kps.rows(); ++i) {
            int cx = static_cast<int>(kps(i, 0) * num_div_x_ / width_);
            int cy = static_cast<int>(kps(i, 1) * num_div_y_ / height_);
            cx = std::clamp(cx, 0, num_div_x_ - 1);
            cy = std::clamp(cy, 0, num_div_y_ - 1);
            counts_[static_cast<size_t>(cy * num_div_x_ + cx)] += 1;
        }
    }

    bool is_each_cell_covered(int num_min_points) const {
        for (int count : counts_) {
            if (count < num_min_points) {
                return false;
            }
        }
        return true;
    }

    int num_cells_uncovered(int num_min_points) const {
        int uncovered = 0;
        for (int count : counts_) {
            if (count < num_min_points) {
                ++uncovered;
            }
        }
        return uncovered;
    }

  private:
    int width_;
    int height_;
    int num_div_x_;
    int num_div_y_;
    std::vector<int> counts_;
};

inline Eigen::Matrix4d pose_from_Rt(const cv::Mat &R, const cv::Mat &t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R.at<double>(r, c);
        }
        T(r, 3) = t.rows == 3 ? t.at<double>(r, 0) : t.at<double>(0, r);
    }
    return T;
}

inline std::vector<double> angles_as_double(const std::vector<float> &angles) {
    return std::vector<double>(angles.begin(), angles.end());
}

} // namespace pyslam

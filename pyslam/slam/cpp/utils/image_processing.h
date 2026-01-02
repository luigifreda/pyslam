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

#include "accessors.h"
#include "eigen_aliases.h"
#include "messages.h"

#include <Eigen/Dense>
#include <array>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <vector>

namespace pyslam {
namespace utils {

template <int num_channels = 3, typename PointsType = MatNx2d>
std::vector<cv::Vec3f> extract_mean_colors(const cv::Mat &img, const PointsType &img_coords,
                                           const int delta = 1,
                                           const cv::Vec3f &default_color = cv::Vec3f(0, 0, 0)) {

    const int H = img.rows;
    const int W = img.cols;
    const int C = img.channels();

    const int type = img.type();

    const int N = img_coords.size();

    if (C != num_channels) {
        MSG_ERROR("extract_mean_colors() - image channels do not match the number of channels");
        return std::vector<cv::Vec3f>(N, cv::Vec3f(static_cast<float>(default_color[0]),
                                                   static_cast<float>(default_color[1]),
                                                   static_cast<float>(default_color[2])));
    }

    if (type != CV_8UC1 && type != CV_8UC3) {
        MSG_ERROR("extract_mean_colors() - image type is not CV_8U");
        return std::vector<cv::Vec3f>(N, cv::Vec3f(static_cast<float>(default_color[0]),
                                                   static_cast<float>(default_color[1]),
                                                   static_cast<float>(default_color[2])));
    }

    const int patch_size = 1 + 2 * delta;
    const int patch_area = patch_size * patch_size;

    std::vector<cv::Vec3f> result(N);

    unsigned char *img_data = img.data;
    const int step = img.step;

    for (int i = 0; i < N; ++i) {
        const auto [x, y] = get_xy_at<PointsType, int>(img_coords, i);

        if (x - delta >= 0 && x + delta < W && y - delta >= 0 && y + delta < H) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
            // Pre-compute base row pointer
            unsigned char *base_row_ptr = img_data + y * step;
            for (int dy = -delta; dy <= delta; ++dy) {
                unsigned char *pixel_ptr = base_row_ptr + dy * step + x * num_channels;
                for (int dx = -delta; dx <= delta; ++dx) {
                    if constexpr (num_channels == 3) {
                        acc0 += static_cast<int>(pixel_ptr[0]);
                        acc1 += static_cast<int>(pixel_ptr[1]);
                        acc2 += static_cast<int>(pixel_ptr[2]);
                    } else if constexpr (num_channels == 1) {
                        acc0 += static_cast<int>(pixel_ptr[0]);
                    }
                    pixel_ptr += num_channels;
                }
            }

            if constexpr (num_channels == 3) {
                result[i] = cv::Vec3f(static_cast<float>(acc0) / patch_area,
                                      static_cast<float>(acc1) / patch_area,
                                      static_cast<float>(acc2) / patch_area);
            } else if constexpr (num_channels == 1) {
                result[i] = cv::Vec3f(static_cast<float>(acc0) / patch_area);
            }
        } else {
            result[i] = default_color;
        }
    }

    return result;
}

// This class is used to extract mean colors from an image
class ImageColorExtractor {
  private:
    const cv::Mat &img_;
    int H_, W_, C_;
    int type_;
    unsigned char *img_data_;
    int step_;
    int delta_;
    bool is_valid_;
    cv::Vec3f default_color_ = cv::Vec3f(0, 0, 0);
    int patch_size_;
    int patch_area_;

  public:
    // Constructor
    explicit ImageColorExtractor(const cv::Mat &img, int delta = 1,
                                 const cv::Vec3f &default_color = cv::Vec3f(0, 0, 0))
        : img_(img), delta_(delta), default_color_(default_color) {
        H_ = img.rows;
        W_ = img.cols;
        C_ = img.channels();
        type_ = img.type();

        if (type_ != CV_8UC1 && type_ != CV_8UC3) {
            MSG_WARN("ImageColorExtractor() - image type is not CV_8U");
            is_valid_ = false;
            return;
        }

        img_data_ = img_.data;
        step_ = img_.step;
        patch_size_ = 1 + 2 * delta_;
        patch_area_ = patch_size_ * patch_size_;
        is_valid_ = true;
    }

    // Check if the extractor is valid
    bool is_valid() const { return is_valid_; }

    // Get image dimensions
    int height() const { return H_; }
    int width() const { return W_; }
    int channels() const { return C_; }

    cv::Vec3f default_color() const { return default_color_; }
    void set_default_color(const cv::Vec3f &default_color) { default_color_ = default_color; }

    // Extract mean color for a single point
    template <bool is_color> cv::Vec3f extract_mean_color(int x, int y) const {

        if (!is_valid_) {
            return default_color_;
        }

        // Check bounds
        if (x - delta_ < 0 || x + delta_ >= W_ || y - delta_ < 0 || y + delta_ >= H_) {
            return default_color_;
        }

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

        // Pre-compute base row pointer
        unsigned char *base_row = img_data_ + y * step_;

        for (int dy = -delta_; dy <= delta_; ++dy) {
            unsigned char *pixel_ptr = base_row + dy * step_ + x * C_;
            for (int dx = -delta_; dx <= delta_; ++dx) {
                if constexpr (is_color) {
                    acc0 += static_cast<int>(pixel_ptr[0]);
                    acc1 += static_cast<int>(pixel_ptr[1]);
                    acc2 += static_cast<int>(pixel_ptr[2]);
                    pixel_ptr += 3;
                } else {
                    acc0 += static_cast<int>(pixel_ptr[0]);
                    pixel_ptr += 1;
                }
            }
        }

        if constexpr (is_color) {
            return cv::Vec3f(static_cast<float>(acc0) / patch_area_,
                             static_cast<float>(acc1) / patch_area_,
                             static_cast<float>(acc2) / patch_area_);
        } else {
            return cv::Vec3f(static_cast<float>(acc0) / patch_area_);
        }
    }

    // Extract mean color for a single point (template version for different point types)
    template <typename PointType, bool is_color>
    cv::Vec3f extract_mean_color(const PointType &point) const {
        const auto [x, y] = get_xy<PointType, int>(point);
        return extract_mean_color<is_color>(x, y);
    }

    template <typename PointType>
    cv::Vec3f extract_mean_color(const PointType &point, const bool is_color) const {
        const auto [x, y] = get_xy<PointType, int>(point);
        if (is_color) {
            return extract_mean_color<true>(x, y);
        } else {
            return extract_mean_color<false>(x, y);
        }
    }

    // Extract mean colors for multiple points (similar to original function)
    template <typename PointsType, bool is_color>
    std::vector<cv::Vec3f> extract_mean_colors(const PointsType &img_coords) const {
        if (!is_valid_) {
            return std::vector<cv::Vec3f>(img_coords.size(), default_color_);
        }

        const int N = img_coords.size();
        std::vector<cv::Vec3f> result(N);

        if (is_color) {
            for (int i = 0; i < N; ++i) {
                const auto [x, y] = get_xy_at<PointsType, int>(img_coords, i);
                result[i] = extract_mean_color<true>(x, y);
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const auto [x, y] = get_xy_at<PointsType, int>(img_coords, i);
                result[i] = extract_mean_color<false>(x, y);
            }
        }
        return result;
    }

    template <typename PointsType>
    std::vector<cv::Vec3f> extract_mean_colors(const PointsType &img_coords,
                                               const bool is_color) const {
        return extract_mean_colors<PointsType, is_color>(img_coords);
    }
};

} // namespace utils
} // namespace pyslam
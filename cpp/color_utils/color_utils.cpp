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

#include "color_utils.h"

#include <cstdint>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

// Include ColorTableGenerator from pyslam/slam/cpp/utils/color_helpers.h
// Note: We need to use relative path from cpp/color_utils to pyslam/slam/cpp/utils
#include "utils/color_helpers.h"

namespace pyslam {
namespace {

// NOTE: Templates provide compile-time specialization and avoid runtime branching.

template <bool UseHash, bool LutIsPow2>
inline size_t lut_index(const IdsColorTable &table, int64_t id_val, uint64_t id_unsigned,
                        size_t lut_mask, size_t lut_size) {
    if constexpr (UseHash) {
        return static_cast<size_t>(table.hash_id(id_val));
    } else if constexpr (LutIsPow2) {
        return id_unsigned & lut_mask;
    } else {
        return id_unsigned % lut_size;
    }
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
void fill_rgb_contiguous(const IdsColorTable &table, const int32_t *ids_ptr, uchar *rgb_ptr,
                         int total_elements, const cv::Vec3b &unlabeled_color,
                         const cv::Vec3b *color_lut_ptr, size_t lut_size, size_t lut_mask) {
    for (int i = 0; i < total_elements; ++i) {
        int64_t id_val = static_cast<int64_t>(ids_ptr[i]);
        const uint64_t id_unsigned = static_cast<uint64_t>(id_val);

        cv::Vec3b color;
        if (id_val < 0) {
            color = unlabeled_color;
        } else {
            const size_t idx =
                lut_index<UseHash, LutIsPow2>(table, id_val, id_unsigned, lut_mask, lut_size);
            color = color_lut_ptr[idx];
        }

        const int out_offset = i * 3;
        if constexpr (SwapRb) {
            rgb_ptr[out_offset + 0] = color[2]; // R
            rgb_ptr[out_offset + 1] = color[1]; // G
            rgb_ptr[out_offset + 2] = color[0]; // B
        } else {
            rgb_ptr[out_offset + 0] = color[0]; // B
            rgb_ptr[out_offset + 1] = color[1]; // G
            rgb_ptr[out_offset + 2] = color[2]; // R
        }
    }
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
void fill_rgb_row(const IdsColorTable &table, const int32_t *ids_row, cv::Vec3b *rgb_row, int cols,
                  const cv::Vec3b &unlabeled_color, const cv::Vec3b *color_lut_ptr, size_t lut_size,
                  size_t lut_mask) {
    for (int c = 0; c < cols; ++c) {
        int64_t id_val = static_cast<int64_t>(ids_row[c]);
        const uint64_t id_unsigned = static_cast<uint64_t>(id_val);

        cv::Vec3b color;
        if (id_val < 0) {
            color = unlabeled_color;
        } else {
            const size_t idx =
                lut_index<UseHash, LutIsPow2>(table, id_val, id_unsigned, lut_mask, lut_size);
            color = color_lut_ptr[idx];
        }

        if constexpr (SwapRb) {
            rgb_row[c][0] = color[2];
            rgb_row[c][1] = color[1];
            rgb_row[c][2] = color[0];
        } else {
            rgb_row[c] = color;
        }
    }
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
void fill_rgb_contiguous_float(const IdsColorTable &table, const int32_t *ids_ptr, float *rgb_ptr,
                               int total_elements, const cv::Vec3b &unlabeled_color,
                               const cv::Vec3b *color_lut_ptr, size_t lut_size, size_t lut_mask) {
    constexpr float kInv255 = 1.0f / 255.0f;
    for (int i = 0; i < total_elements; ++i) {
        int64_t id_val = static_cast<int64_t>(ids_ptr[i]);
        const uint64_t id_unsigned = static_cast<uint64_t>(id_val);

        cv::Vec3b color;
        if (id_val < 0) {
            color = unlabeled_color;
        } else {
            const size_t idx =
                lut_index<UseHash, LutIsPow2>(table, id_val, id_unsigned, lut_mask, lut_size);
            color = color_lut_ptr[idx];
        }

        const int out_offset = i * 3;
        if constexpr (SwapRb) {
            rgb_ptr[out_offset + 0] = static_cast<float>(color[2]) * kInv255; // R
            rgb_ptr[out_offset + 1] = static_cast<float>(color[1]) * kInv255; // G
            rgb_ptr[out_offset + 2] = static_cast<float>(color[0]) * kInv255; // B
        } else {
            rgb_ptr[out_offset + 0] = static_cast<float>(color[0]) * kInv255; // B
            rgb_ptr[out_offset + 1] = static_cast<float>(color[1]) * kInv255; // G
            rgb_ptr[out_offset + 2] = static_cast<float>(color[2]) * kInv255; // R
        }
    }
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
void fill_rgb_row_float(const IdsColorTable &table, const int32_t *ids_row, cv::Vec3f *rgb_row,
                        int cols, const cv::Vec3b &unlabeled_color, const cv::Vec3b *color_lut_ptr,
                        size_t lut_size, size_t lut_mask) {
    constexpr float kInv255 = 1.0f / 255.0f;
    for (int c = 0; c < cols; ++c) {
        int64_t id_val = static_cast<int64_t>(ids_row[c]);
        const uint64_t id_unsigned = static_cast<uint64_t>(id_val);

        cv::Vec3b color;
        if (id_val < 0) {
            color = unlabeled_color;
        } else {
            const size_t idx =
                lut_index<UseHash, LutIsPow2>(table, id_val, id_unsigned, lut_mask, lut_size);
            color = color_lut_ptr[idx];
        }

        if constexpr (SwapRb) {
            rgb_row[c][0] = static_cast<float>(color[2]) * kInv255;
            rgb_row[c][1] = static_cast<float>(color[1]) * kInv255;
            rgb_row[c][2] = static_cast<float>(color[0]) * kInv255;
        } else {
            rgb_row[c][0] = static_cast<float>(color[0]) * kInv255;
            rgb_row[c][1] = static_cast<float>(color[1]) * kInv255;
            rgb_row[c][2] = static_cast<float>(color[2]) * kInv255;
        }
    }
}

} // namespace

// ------------------------------------------------------------
// IdsColorTable implementation
// ------------------------------------------------------------

IdsColorTable::IdsColorTable() {
    // Build LUT from ColorTableGenerator (explicit instance, not singleton)
    const auto &color_gen = ColorTableGenerator::instance();
    color_lut_.reserve(ColorTableGenerator::TABLE_SIZE);
    static_assert(utils::is_power_of_2<size_t>(ColorTableGenerator::TABLE_SIZE),
                  "TABLE_SIZE must be a power of 2");

    for (size_t i = 0; i < ColorTableGenerator::TABLE_SIZE; ++i) {
        ColorTableGenerator::RGB rgb = color_gen.color_from_int(i);
        color_lut_.push_back(cv::Vec3b(rgb.b, rgb.g, rgb.r)); // Convert RGB to BGR for OpenCV
    }

    lut_size_ = color_lut_.size();
    lut_mask_ = lut_size_ - 1;
    lut_is_pow2_ = (lut_size_ & lut_mask_) == 0;
}

uint8_t IdsColorTable::hash_id(const int64_t id_val) const {
    // Convert signed to unsigned for hash function
    const uint64_t id_unsigned = static_cast<uint64_t>(id_val);
    const uint64_t x = ColorTableGenerator::splitmix64(id_unsigned);
    return static_cast<uint8_t>(x &
                                ColorTableGenerator::TABLE_MASK); // Bitwise AND instead of modulo
}

template <bool UseHash, bool SwapRb>
cv::Mat IdsColorTable::ids_to_rgb_dispatch(const cv::Mat &ids, const cv::Vec3b &unlabeled_color,
                                           bool lut_is_pow2) const {
    if (lut_is_pow2) {
        return _ids_to_rgb<UseHash, SwapRb, true>(ids, unlabeled_color);
    }
    return _ids_to_rgb<UseHash, SwapRb, false>(ids, unlabeled_color);
}

template <bool UseHash, bool SwapRb>
cv::Mat IdsColorTable::ids_to_rgb_float_dispatch(const cv::Mat &ids,
                                                 const cv::Vec3b &unlabeled_color,
                                                 bool lut_is_pow2) const {
    if (lut_is_pow2) {
        return _ids_to_rgb_float<UseHash, SwapRb, true>(ids, unlabeled_color);
    }
    return _ids_to_rgb_float<UseHash, SwapRb, false>(ids, unlabeled_color);
}

// ------------------------------------------------------------
// IdsColorTable::ids_to_rgb implementation
// ------------------------------------------------------------

cv::Mat IdsColorTable::ids_to_rgb(const cv::Mat &ids, bool bgr, const cv::Vec3b &unlabeled_color,
                                  bool use_hash) const {
    const bool swap_rb = !bgr;
    const bool lut_is_pow2 = lut_is_pow2_;
    if (use_hash) {
        return swap_rb ? ids_to_rgb_dispatch<true, true>(ids, unlabeled_color, lut_is_pow2)
                       : ids_to_rgb_dispatch<true, false>(ids, unlabeled_color, lut_is_pow2);
    }
    return swap_rb ? ids_to_rgb_dispatch<false, true>(ids, unlabeled_color, lut_is_pow2)
                   : ids_to_rgb_dispatch<false, false>(ids, unlabeled_color, lut_is_pow2);
}

cv::Mat IdsColorTable::ids_to_rgb_float(const cv::Mat &ids, bool bgr,
                                        const cv::Vec3b &unlabeled_color, bool use_hash) const {
    const bool swap_rb = !bgr;
    const bool lut_is_pow2 = lut_is_pow2_;
    if (use_hash) {
        return swap_rb ? ids_to_rgb_float_dispatch<true, true>(ids, unlabeled_color, lut_is_pow2)
                       : ids_to_rgb_float_dispatch<true, false>(ids, unlabeled_color, lut_is_pow2);
    }
    return swap_rb ? ids_to_rgb_float_dispatch<false, true>(ids, unlabeled_color, lut_is_pow2)
                   : ids_to_rgb_float_dispatch<false, false>(ids, unlabeled_color, lut_is_pow2);
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
cv::Mat IdsColorTable::_ids_to_rgb(const cv::Mat &ids, const cv::Vec3b &unlabeled_color) const {
    // Validate input
    if (ids.empty()) {
        // Return empty Mat with correct type
        return cv::Mat();
    }

    // Check if input has any elements
    if (ids.total() == 0) {
        // Return empty Mat with correct shape
        if (ids.dims == 2) {
            return cv::Mat(ids.rows, ids.cols, CV_8UC3);
        } else {
            return cv::Mat(0, 3, CV_8UC3);
        }
    }

    // Convert ids to appropriate type and handle shape
    CV_Assert(ids.channels() == 1);
    cv::Mat ids_converted;
    if (ids.type() == CV_32SC1) {
        ids_converted = ids;
    } else {
        // Convert to CV_32SC1 (int32) - handles CV_8UC1, CV_16SC1, CV_64F, etc.
        ids.convertTo(ids_converted, CV_32SC1);
    }

    // Determine if input is 1D or 2D. Treat row/col vectors as 1D for (N,3) output.
    bool is_2d = ids_converted.dims == 2 && ids_converted.rows > 1 && ids_converted.cols > 1;

    // Flatten for processing
    const int total_elements = static_cast<int>(ids_converted.total());
    if (total_elements == 0) {
        // Return empty Mat with correct shape
        if (is_2d) {
            return cv::Mat(ids_converted.rows, ids_converted.cols, CV_8UC3);
        } else {
            return cv::Mat(0, 3, CV_8UC3);
        }
    }

    cv::Mat flat_ids = ids_converted.reshape(1, total_elements);

    // Create output RGB image directly (can be faster than using cv::LUT)
    // For 1D: (total_elements, 3) - 1 channel, 3 columns
    // For 2D: (rows, cols, 3) - 3 channels
    cv::Mat rgb_output;
    if (is_2d) {
        rgb_output = cv::Mat(ids_converted.rows, ids_converted.cols, CV_8UC3);
    } else {
        // For 1D: create directly as (total_elements, 3) with 1 channel
        // Ensure it's continuous for proper NumPy conversion
        rgb_output = cv::Mat::zeros(total_elements, 3, CV_8UC1);
    }

    // Direct indexing into color_lut_ (can be faster than cv::LUT)
    // Process each element and directly write to output
    const cv::Vec3b *color_lut_ptr = color_lut_.data();

    if (is_2d) {
        // 2D case: use flat pointer when contiguous for speed
        if (ids_converted.isContinuous() && rgb_output.isContinuous()) {
            const int32_t *ids_ptr = ids_converted.ptr<int32_t>(0);
            uchar *rgb_ptr = rgb_output.ptr<uchar>(0);
            fill_rgb_contiguous<UseHash, SwapRb, LutIsPow2>(*this, ids_ptr, rgb_ptr, total_elements,
                                                            unlabeled_color, color_lut_ptr,
                                                            lut_size_, lut_mask_);
        } else {
            // Fallback: iterate through rows and cols using row pointers
            for (int r = 0; r < ids_converted.rows; ++r) {
                const int32_t *ids_row = ids_converted.ptr<int32_t>(r);
                cv::Vec3b *rgb_row = rgb_output.ptr<cv::Vec3b>(r);
                fill_rgb_row<UseHash, SwapRb, LutIsPow2>(*this, ids_row, rgb_row,
                                                         ids_converted.cols, unlabeled_color,
                                                         color_lut_ptr, lut_size_, lut_mask_);
            }
        }
    } else {
        // 1D case: iterate through elements using pointers (faster than .at<>)
        const int32_t *ids_ptr = flat_ids.ptr<int32_t>();
        uchar *rgb_ptr = rgb_output.ptr<uchar>();
        fill_rgb_contiguous<UseHash, SwapRb, LutIsPow2>(*this, ids_ptr, rgb_ptr, total_elements,
                                                        unlabeled_color, color_lut_ptr, lut_size_,
                                                        lut_mask_);
    }
    return rgb_output;
}

template <bool UseHash, bool SwapRb, bool LutIsPow2>
cv::Mat IdsColorTable::_ids_to_rgb_float(const cv::Mat &ids,
                                         const cv::Vec3b &unlabeled_color) const {
    if (ids.empty()) {
        return cv::Mat();
    }

    if (ids.total() == 0) {
        if (ids.dims == 2) {
            return cv::Mat(ids.rows, ids.cols, CV_32FC3);
        } else {
            return cv::Mat(0, 3, CV_32FC1);
        }
    }

    CV_Assert(ids.channels() == 1);
    cv::Mat ids_converted;
    if (ids.type() == CV_32SC1) {
        ids_converted = ids;
    } else {
        ids.convertTo(ids_converted, CV_32SC1);
    }

    bool is_2d = ids_converted.dims == 2 && ids_converted.rows > 1 && ids_converted.cols > 1;

    const int total_elements = static_cast<int>(ids_converted.total());
    if (total_elements == 0) {
        if (is_2d) {
            return cv::Mat(ids_converted.rows, ids_converted.cols, CV_32FC3);
        } else {
            return cv::Mat(0, 3, CV_32FC1);
        }
    }

    cv::Mat flat_ids = ids_converted.reshape(1, total_elements);

    cv::Mat rgb_output;
    if (is_2d) {
        rgb_output = cv::Mat(ids_converted.rows, ids_converted.cols, CV_32FC3);
    } else {
        rgb_output = cv::Mat::zeros(total_elements, 3, CV_32FC1);
    }

    const cv::Vec3b *color_lut_ptr = color_lut_.data();

    if (is_2d) {
        if (ids_converted.isContinuous() && rgb_output.isContinuous()) {
            const int32_t *ids_ptr = ids_converted.ptr<int32_t>(0);
            float *rgb_ptr = rgb_output.ptr<float>(0);
            fill_rgb_contiguous_float<UseHash, SwapRb, LutIsPow2>(
                *this, ids_ptr, rgb_ptr, total_elements, unlabeled_color, color_lut_ptr, lut_size_,
                lut_mask_);
        } else {
            for (int r = 0; r < ids_converted.rows; ++r) {
                const int32_t *ids_row = ids_converted.ptr<int32_t>(r);
                cv::Vec3f *rgb_row = rgb_output.ptr<cv::Vec3f>(r);
                fill_rgb_row_float<UseHash, SwapRb, LutIsPow2>(*this, ids_row, rgb_row,
                                                               ids_converted.cols, unlabeled_color,
                                                               color_lut_ptr, lut_size_, lut_mask_);
            }
        }
    } else {
        const int32_t *ids_ptr = flat_ids.ptr<int32_t>();
        float *rgb_ptr = rgb_output.ptr<float>();
        fill_rgb_contiguous_float<UseHash, SwapRb, LutIsPow2>(*this, ids_ptr, rgb_ptr,
                                                              total_elements, unlabeled_color,
                                                              color_lut_ptr, lut_size_, lut_mask_);
    }

    return rgb_output;
}

} // namespace pyslam

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

#include <cstddef>
#include <cstdint>
#include <opencv2/core/core.hpp>
#include <vector>

namespace pyslam {

/**
 * Color table manager for ID to RGB conversion.
 * Explicitly manages the color table and LUT to avoid singleton deallocation issues.
 */
class IdsColorTable {
  public:
    IdsColorTable();
    ~IdsColorTable() = default;

    // Non-copyable, movable
    IdsColorTable(const IdsColorTable &) = delete;
    IdsColorTable &operator=(const IdsColorTable &) = delete;
    IdsColorTable(IdsColorTable &&) = default;
    IdsColorTable &operator=(IdsColorTable &&) = default;

    /**
     * Hash an integer ID to table index using splitmix64.
     */
    uint8_t hash_id(int64_t id_val) const;

    /**
     * Get the color LUT (BGR format for OpenCV).
     */
    const std::vector<cv::Vec3b> &get_color_lut() const { return color_lut_; }

    /**
     * Get the color from the color LUT from an integer ID using a hash function.
     */
    inline cv::Vec3b get_color_from_id(const int64_t id_val) const {
        const uint8_t idx = hash_id(id_val);
        return color_lut_[idx];
    }
    inline std::array<uint8_t, 3> get_color_from_id_array(const int64_t id_val) const {
        const cv::Vec3b color = get_color_from_id(id_val);
        return {color[0], color[1], color[2]};
    }

    /**
     * Get the color from the color LUT using an integer ID without hash.
     */
    inline cv::Vec3b get_color_from_id_no_hash(const int64_t id_val) const {
        const uint64_t id_unsigned = static_cast<uint64_t>(id_val);
        // const size_t idx = lut_is_pow2_ ? (id_unsigned & lut_mask_) : (id_unsigned % lut_size_);
        const size_t idx =
            id_unsigned &
            lut_mask_; // we assume the lut is a power of 2 and we use bitwise AND instead of modulo
        return color_lut_[idx];
    }
    inline std::array<uint8_t, 3> get_color_from_id_array_no_hash(const int64_t id_val) const {
        const cv::Vec3b color = get_color_from_id_no_hash(id_val);
        return {color[0], color[1], color[2]};
    }

    /**
     * Converts IDs to RGB colors using a hash-based color table.
     * This function is designed for IDs which can be arbitrary integers
     * (including negative values like -1 for unlabeled). Unlike semantic class IDs,
     * instance IDs don't have a fixed range and need a hash-based color mapping.
     *
     * @param ids Input IDs (CV_32SC1, 1D or 2D)
     * @param bgr If true, return BGR format; otherwise RGB
     * @param unlabeled_color RGB tuple for unlabeled instances (default: black)
     * @return RGB/BGR image array of shape (H, W, 3) for 2D input or (N, 3) for 1D input,
     *         with dtype CV_8UC3 and values in [0, 255]
     */
    cv::Mat ids_to_rgb(const cv::Mat &ids, bool bgr = false,
                       const cv::Vec3b &unlabeled_color = cv::Vec3b(0, 0, 0),
                       bool use_hash = false) const;
    cv::Mat ids_to_rgb_float(const cv::Mat &ids, bool bgr = false,
                             const cv::Vec3b &unlabeled_color = cv::Vec3b(0, 0, 0),
                             bool use_hash = false) const;

  private:
    template <bool UseHash, bool SwapRb, bool LutIsPow2>
    cv::Mat _ids_to_rgb(const cv::Mat &ids,
                        const cv::Vec3b &unlabeled_color = cv::Vec3b(0, 0, 0)) const;
    template <bool UseHash, bool SwapRb, bool LutIsPow2>
    cv::Mat _ids_to_rgb_float(const cv::Mat &ids,
                              const cv::Vec3b &unlabeled_color = cv::Vec3b(0, 0, 0)) const;
    template <bool UseHash, bool SwapRb>
    cv::Mat ids_to_rgb_dispatch(const cv::Mat &ids, const cv::Vec3b &unlabeled_color,
                                bool lut_is_pow2) const;
    template <bool UseHash, bool SwapRb>
    cv::Mat ids_to_rgb_float_dispatch(const cv::Mat &ids, const cv::Vec3b &unlabeled_color,
                                      bool lut_is_pow2) const;

    std::vector<cv::Vec3b> color_lut_;
    size_t lut_size_ = 0;
    size_t lut_mask_ = 0;
    bool lut_is_pow2_ =
        true; // NOTE: We assume the table is a power of 2 and we static_assert in the constructor
};

} // namespace pyslam

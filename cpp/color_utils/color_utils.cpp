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

// ------------------------------------------------------------
// IdsColorTable implementation
// ------------------------------------------------------------

IdsColorTable::IdsColorTable() {
    // Build LUT from ColorTableGenerator (explicit instance, not singleton)
    const auto &color_gen = ColorTableGenerator::instance();
    color_lut_.reserve(ColorTableGenerator::TABLE_SIZE);

    for (size_t i = 0; i < ColorTableGenerator::TABLE_SIZE; ++i) {
        ColorTableGenerator::RGB rgb = color_gen.color_from_int(i);
        color_lut_.push_back(cv::Vec3b(rgb.b, rgb.g, rgb.r)); // Convert RGB to BGR for OpenCV
    }
}

uint8_t IdsColorTable::hash_id(int64_t id_val) const {
    // Convert signed to unsigned for hash function
    uint64_t id_unsigned = static_cast<uint64_t>(id_val);

    // Use the same splitmix64 implementation as ColorTableGenerator
    uint64_t x = id_unsigned;
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);

    return static_cast<uint8_t>(x & ColorTableGenerator::TABLE_MASK);
}

// ------------------------------------------------------------
// IdsColorTable::ids_to_rgb implementation
// ------------------------------------------------------------

cv::Mat IdsColorTable::ids_to_rgb(const cv::Mat &ids, bool bgr,
                                  const cv::Vec3b &unlabeled_color) const {
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
    cv::Mat ids_converted;
    if (ids.type() == CV_32SC1) {
        ids_converted = ids;
    } else {
        // Convert to CV_32SC1 (int32) - handles CV_8UC1, CV_16SC1, CV_64F, etc.
        ids.convertTo(ids_converted, CV_32SC1);
    }

    // Determine if input is 1D or 2D. Treat column/row vectors as 1D.
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

    // Create output RGB image directly (faster than using cv::LUT)
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

    // Direct indexing into color_lut_ (much faster than cv::LUT)
    // Process each element and directly write to output
    const cv::Vec3b *color_lut_ptr = color_lut_.data();
    const size_t lut_size = color_lut_.size();

    if (is_2d) {
        // 2D case: iterate through rows and cols using row pointers (faster than .at<>)
        for (int r = 0; r < ids_converted.rows; ++r) {
            const int32_t *ids_row = ids_converted.ptr<int32_t>(r);
            cv::Vec3b *rgb_row = rgb_output.ptr<cv::Vec3b>(r);

            for (int c = 0; c < ids_converted.cols; ++c) {
                int64_t id_val = static_cast<int64_t>(ids_row[c]);

                cv::Vec3b color;
                if (id_val == -1) {
                    // Unlabeled: use provided color (already in BGR format from OpenCV)
                    color = unlabeled_color;
                } else {
                    // Hash to table index and directly index color_lut_
                    uint8_t idx = hash_id(id_val);
                    color = color_lut_ptr[idx % lut_size];
                }

                // Write BGR directly to output
                rgb_row[c] = color;
            }
        }
    } else {
        // 1D case: iterate through elements using pointers (faster than .at<>)
        const int32_t *ids_ptr = flat_ids.ptr<int32_t>();
        uchar *rgb_ptr = rgb_output.ptr<uchar>();

        for (int i = 0; i < total_elements; ++i) {
            int64_t id_val = static_cast<int64_t>(ids_ptr[i]);

            cv::Vec3b color;
            if (id_val == -1) {
                // Unlabeled: use provided color (already in BGR format from OpenCV)
                color = unlabeled_color;
            } else {
                // Hash to table index and directly index color_lut_
                uint8_t idx = hash_id(id_val);
                color = color_lut_ptr[idx % lut_size];
            }

            // Write color to output as (N, 3) format
            // If bgr=true: write B,G,R to columns 0,1,2
            // If bgr=false: write R,G,B to columns 0,1,2 (swap B and R)
            const int row_offset = i * 3; // 3 columns per row
            if (bgr) {
                rgb_ptr[row_offset + 0] = color[0]; // B
                rgb_ptr[row_offset + 1] = color[1]; // G
                rgb_ptr[row_offset + 2] = color[2]; // R
            } else {
                rgb_ptr[row_offset + 0] = color[2]; // R
                rgb_ptr[row_offset + 1] = color[1]; // G
                rgb_ptr[row_offset + 2] = color[0]; // B
            }
        }
    }

    // For 2D case, convert BGR to RGB if needed
    if (!bgr && is_2d) {
        cv::cvtColor(rgb_output, rgb_output, cv::COLOR_BGR2RGB);
    }

    return rgb_output;
}

} // namespace pyslam

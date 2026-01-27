/**
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

#include "voxel_semantic_shared_data.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <type_traits>

namespace volumetric {

inline bool check_image_size(const cv::Mat &image, const int height, const int width,
                             const std::string &image_name) {
    if (image.rows == 0 || image.cols == 0 || image.rows != height || image.cols != width) {
        std::cout << "check_image_size: " << image_name << " size: " << image.rows << "x"
                  << image.cols << std::endl;
        std::cout << "\tExpected height: " << height << std::endl;
        std::cout << "\tExpected width: " << width << std::endl;
        std::cout << "\tImage size does not match expected size" << std::endl;
        return false;
    }
    return true;
}

inline cv::Mat convert_image_type_if_needed(const cv::Mat &image, int expected_type,
                                            const std::string &image_name) {
    if (image.empty()) {
        return image;
    }

    if (image.type() != expected_type) {
        cv::Mat converted_image;
        std::cout << "check_image_type: " << image_name << " type: " << image.type() << std::endl;
        std::cout << "\tConverting image to " << expected_type << std::endl;
        image.convertTo(converted_image, expected_type);
        return converted_image;
    } else {
        return image;
    }
}

// Remap the instance ids using the map
// Inputs:
// - instance_ids: the instance ids to remap
// - map: the map to use for remapping
// Returns: the remapped instance ids
// Template parameters:
// - MapType: the map type
// - InstanceIdType: the instance id type
template <typename MapType, typename InstanceIdType>
inline cv::Mat
remap_instance_ids(const cv::Mat &instance_ids, MapType &map,
                   const InstanceIdType invalid_instance_id = static_cast<InstanceIdType>(-1)) {

    if (instance_ids.empty()) {
        return instance_ids;
    }
    // check the type of the instance_ids
    // Check that it's single-channel and has the correct depth
    if (instance_ids.channels() != 1) {
        throw std::runtime_error("Instance ids must be single-channel");
    }

    const int mat_depth = CV_MAT_DEPTH(instance_ids.type());
    if constexpr (std::is_same_v<InstanceIdType, int32_t>) {
        if (mat_depth != CV_32S) {
            throw std::runtime_error("Instance ids must be int32");
        }
    } else if constexpr (std::is_same_v<InstanceIdType, int8_t>) {
        if (mat_depth != CV_8S) {
            throw std::runtime_error("Instance ids must be int8");
        }
    } else if constexpr (std::is_same_v<InstanceIdType, uint8_t>) {
        if (mat_depth != CV_8U) {
            throw std::runtime_error("Instance ids must be uint8");
        }
    } else if constexpr (std::is_same_v<InstanceIdType, int16_t>) {
        if (mat_depth != CV_16S) {
            throw std::runtime_error("Instance ids must be int16");
        }
    } else if constexpr (std::is_same_v<InstanceIdType, uint16_t>) {
        if (mat_depth != CV_16U) {
            throw std::runtime_error("Instance ids must be uint16");
        }
    } else {
        throw std::runtime_error("Unsupported instance id type");
    }

    cv::Mat remapped_instance_ids(instance_ids.rows, instance_ids.cols, instance_ids.type());

    if (map.empty()) {
#if 0        
        // Generate a new object id for each instance id and return the remapped instance ids
        std::unordered_map<InstanceIdType, InstanceIdType> fallback_map;
        fallback_map.reserve(64);
        for (int i = 0; i < instance_ids.rows; i++) {
            const auto *instance_id_row_ptr = instance_ids.ptr<InstanceIdType>(i);
            auto *remapped_instance_id_row_ptr = remapped_instance_ids.ptr<InstanceIdType>(i);
            for (int j = 0; j < instance_ids.cols; j++) {
                const InstanceIdType instance_id = instance_id_row_ptr[j];
                InstanceIdType &remapped_instance_id = remapped_instance_id_row_ptr[j];
                if constexpr (std::is_signed_v<InstanceIdType>) {
                    if (instance_id < 0) {
                        remapped_instance_id = invalid_instance_id;
                        continue;
                    }
                }
                if (instance_id == static_cast<InstanceIdType>(0)) {
                    // Background/stuff stays 0
                    remapped_instance_id = static_cast<InstanceIdType>(0);
                    continue;
                }
                const auto it = fallback_map.find(instance_id);
                if (it != fallback_map.end()) {
                    remapped_instance_id = it->second;
                } else {
                    const InstanceIdType new_id =
                        static_cast<InstanceIdType>(VoxelSemanticSharedData::get_next_object_id());
                    fallback_map.emplace(instance_id, new_id);
                    remapped_instance_id = new_id;
                }
            }
        }
#else
        // Set all instance ids to invalid instance id
        remapped_instance_ids.setTo(invalid_instance_id);
#endif
        return remapped_instance_ids;
    }

    for (int i = 0; i < instance_ids.rows; i++) {
        const auto &instance_id_row_ptr = instance_ids.ptr<InstanceIdType>(i);
        const auto &remapped_instance_id_row_ptr = remapped_instance_ids.ptr<InstanceIdType>(i);
        for (int j = 0; j < instance_ids.cols; j++) {
            const InstanceIdType &instance_id = instance_id_row_ptr[j];
            InstanceIdType &remapped_instance_id = remapped_instance_id_row_ptr[j];
            const auto &it = map.find(instance_id);
            if (it != map.end()) {
                remapped_instance_id = it->second;
            } else {
                remapped_instance_id = invalid_instance_id;
            }
        }
    }
    return remapped_instance_ids;
}

} // namespace volumetric
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

#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "camera_frustrum.h"
#include "image_utils.h"
#include "voxel_data.h"
#include "voxel_data_semantic.h"
#include "voxel_hashing.h"

#include <opencv2/opencv.hpp>

namespace volumetric {

using MapInstanceIdToObjectId = std::unordered_map<int, int>; // instance ID -> object ID

// Assign object IDs to instance IDs using semantic instances image and depth image.
// A voting is performed for each voxel that has the same class ID as the image class ID.
// If the voxel is in front of the depth image, it is carved if do_carving is true.
// If the voxel is behind the depth image, it is ignored.
// If the voxel is in the same depth as the depth image, it is counted as a vote.
// Inputs:
// - voxel_grid: voxel grid
// - camera_frustrum: camera frustrum
// - class_ids_image: class IDs image
// - semantic_instances_image: semantic instances image
// - depth_image: depth image (optional, default is empty)
// - depth_threshold: depth threshold (optional, default is 5e-2f)
// - do_carving: if true, do carving (optional, default is false)
// - min_vote_ratio: minimum vote ratio (optional, default is 0.5f)
// - min_votes: minimum votes (optional, default is 3)
// Returns: map of instance IDs to object IDs
// Template parameters:
// - VoxelGridT: the voxel grid type
// - VoxelDataT: the voxel data type
template <typename VoxelGridT, typename VoxelDataT>
MapInstanceIdToObjectId assign_object_ids_to_instance_ids(
    VoxelGridT &voxel_grid, const CameraFrustrum &camera_frustrum, const cv::Mat &class_ids_image,
    const cv::Mat &semantic_instances_image, const cv::Mat &depth_image,
    const float depth_threshold = 5e-2f, bool do_carving = false, const float min_vote_ratio = 0.5f,
    const int min_votes = 3) {
    MapInstanceIdToObjectId instance_id_to_object_id;

    // check if semantic instances image is empty
    if (semantic_instances_image.empty() || class_ids_image.empty()) {
        std::cout << "volumetric::assign_object_ids_to_instance_ids: class IDs or semantic "
                     "instances image is empty"
                  << std::endl;
        return instance_id_to_object_id;
    }
    // check the type of the semantic instances image
    if (semantic_instances_image.channels() != 1) {
        throw std::runtime_error("Instance ids must be single-channel");
    }
    // check if semantic instances image size matches camera frustrum size
    if (!check_image_size(semantic_instances_image, camera_frustrum.get_height(),
                          camera_frustrum.get_width(), "Semantic Instances")) {
        return instance_id_to_object_id;
    }

    // check the type of the class ids image
    if (class_ids_image.channels() != 1) {
        throw std::runtime_error("Class ids must be single-channel");
    }
    // check if depth image size matches camera frustrum size
    if (!check_image_size(class_ids_image, camera_frustrum.get_height(),
                          camera_frustrum.get_width(), "Class IDs")) {
        return instance_id_to_object_id;
    }

    // Check semantic instances image type
    cv::Mat semantic_instances_image_int =
        convert_image_type_if_needed(semantic_instances_image, CV_32S, "Semantic Instances");

    // check if class ids image type is int32
    cv::Mat class_ids_image_int =
        convert_image_type_if_needed(class_ids_image, CV_32S, "Class IDs");

    // Validate depth image if provided
    bool use_depth_filter = !depth_image.empty();
    do_carving = do_carving && use_depth_filter;
    cv::Mat depth_image_float;
    if (use_depth_filter) {
        if (!check_image_size(depth_image, camera_frustrum.get_height(),
                              camera_frustrum.get_width(), "Depth")) {
            use_depth_filter = false;
        } else {
            depth_image_float = convert_image_type_if_needed(depth_image, CV_32F, "Depth");
        }
    }

    // Map to track new object IDs assigned to instance IDs for points without object_id
    // This ensures all points from the same instance_id get the same new object_id
    std::unordered_map<int, int> instance_id_to_new_object_id;

    // Structure to store votes: instance_id -> (object_id -> count)
    // std::map is used to store the votes in order to get the most frequent object ID for each
    // instance ID (for a few votes, std::map is faster than std::unordered_map)
    std::unordered_map<int, std::map<int, int>>
        instance_id_to_object_votes; // instance ID -> (object ID -> count)

    const auto callback = [&](VoxelDataT &v, const VoxelKey &key,
                              const std::array<double, 3> &pos_w, const ImagePoint &image_point) {
        const float point_depth = image_point.depth;

        const int point_class_id = v.get_class_id();
        // Skip if the point class ID is invalid (<0)
        if (point_class_id < 0) {
            return;
        }
        const int image_class_id = class_ids_image_int.at<int>(image_point.v, image_point.u);
        // Skip if the image class ID is invalid (<0)
        if (image_class_id < 0) {
            return;
        }

        // Skip if the point class ID does not match the image class ID
        if (point_class_id != image_class_id) {
            return;
        }

        int point_object_id = v.get_object_id();
        const int image_instance_id =
            semantic_instances_image_int.at<int>(image_point.v, image_point.u);

        // Skip invalid instance IDs (<0)
        if (image_instance_id < 0) {
            return;
        }

        // Depth filtering: discard associations to objects that are behind others
        if (use_depth_filter) {
            const float image_depth = depth_image_float.at<float>(image_point.v, image_point.u);

            // Skip invalid depth values (0, NaN, or inf)
            if (image_depth <= 0.0f || !std::isfinite(image_depth)) {
                return;
            }

            if (do_carving) {
                // Carve voxels that are inconsistent with the depth image
                if (point_depth < image_depth - depth_threshold) {
                    v.reset();
                    return;
                }
            }

            // Filter out voxels that are significantly behind the depth image
            // These are likely occluded or inconsistent
            if (point_depth > image_depth + depth_threshold) {
                return;
            }
        }

        // For points without object_id, assign a new object_id based on the instance_id
        // All points from the same instance_id will get the same new object_id
        // Special case: instance_id=0 (background) -> object_id=0 (no specific object)
        if (point_object_id < 0) {
            if (image_instance_id == 0) {
                // Background/stuff: object_id should be 0 (no specific object)
                point_object_id = 0;
            } else {
                // Check if we've already assigned a new object_id for this instance_id
                auto it = instance_id_to_new_object_id.find(image_instance_id);
                if (it == instance_id_to_new_object_id.end()) {
                    // First time seeing this instance_id without object_id - create a new object_id
                    const int new_obj_id = VoxelSemanticSharedData::get_next_object_id();
                    point_object_id = new_obj_id;
                    instance_id_to_new_object_id[image_instance_id] = new_obj_id;
                } else {
                    // Reuse the existing new object_id for this instance_id
                    point_object_id = it->second;
                }
            }
            v.set_object_id(point_object_id);
        }

        // Count vote for this object ID for this instance ID
        instance_id_to_object_votes[image_instance_id][point_object_id]++;
    };
    voxel_grid.iterate_voxels_in_camera_frustrum(camera_frustrum, callback);

    // Perform voting: assign the most frequent object ID to each instance ID
    for (const auto &[instance_id, object_votes] : instance_id_to_object_votes) {
        if (object_votes.empty()) {
            continue;
        }

        // Find the object ID with the most votes
        int max_votes = 0;
        int winning_object_id = -1;
        int total_votes = 0;

        for (const auto &[object_id, vote_count] : object_votes) {
            total_votes += vote_count;
            if (vote_count > max_votes) {
                max_votes = vote_count;
                winning_object_id = object_id;
            }
        }

        // Apply thresholds: minimum votes and ratio threshold
        if (total_votes < min_votes) {
            // Insufficient votes - skip this instance ID
            continue;
        }

        const float vote_ratio = static_cast<float>(max_votes) / static_cast<float>(total_votes);
        if (vote_ratio < min_vote_ratio) {
            // Ambiguous assignment - the winning object ID doesn't have enough dominance
            continue;
        }

        // Assign the winning object ID to this instance ID
        instance_id_to_object_id[instance_id] = winning_object_id;
    }

    return instance_id_to_object_id;
}

} // namespace volumetric
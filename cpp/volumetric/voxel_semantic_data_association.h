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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "camera_frustrum.h"
#include "voxel_data.h"
#include "voxel_hashing.h"

#include <opencv2/opencv.hpp>

namespace volumetric {

using MapInstanceIdToObjectId = std::unordered_map<int, int>; // instance ID -> object ID

// Assign object IDs to instance IDs using semantic instances image and depth image
// Inputs:
// - voxel_grid: voxel grid
// - camera_frustrum: camera frustrum
// - semantic_instances_image: semantic instances image
// - depth_image: depth image (optional, default is empty)
// - depth_threshold: depth threshold (optional, default is 0.1f)
// - min_vote_ratio: minimum vote ratio (optional, default is 0.5f)
// - min_votes: minimum votes (optional, default is 3)
// Returns: map of instance IDs to object IDs
// Template parameters:
// - VoxelGridT: the voxel grid type
// - VoxelDataT: the voxel data type
template <typename VoxelGridT, typename VoxelDataT>
MapInstanceIdToObjectId
assign_object_ids_to_instance_ids(VoxelGridT &voxel_grid, const CameraFrustrum &camera_frustrum,
                                  const cv::Mat &semantic_instances_image,
                                  const cv::Mat &depth_image, const float depth_threshold,
                                  const float min_vote_ratio, const int min_votes) {
    MapInstanceIdToObjectId instance_id_to_object_id;

    if (semantic_instances_image.empty()) {
        std::cout << "VoxelBlockSemanticGridT::assign_object_ids_to_instance_ids: Semantic "
                     "instances image is empty"
                  << std::endl;
        return instance_id_to_object_id;
    }
    if (semantic_instances_image.rows == 0 || semantic_instances_image.cols == 0 ||
        semantic_instances_image.rows != camera_frustrum.get_height() ||
        semantic_instances_image.cols != camera_frustrum.get_width()) {
        std::cout << "VoxelBlockSemanticGridT::assign_object_ids_to_instance_ids: Semantic "
                     "instances image size: "
                  << semantic_instances_image.rows << "x" << semantic_instances_image.cols
                  << std::endl;
        std::cout << "\tCamera frustrum height: " << camera_frustrum.get_height() << std::endl;
        std::cout << "\tCamera frustrum width: " << camera_frustrum.get_width() << std::endl;
        std::cout << "\tSemantic instances image size does not match camera frustrum size"
                  << std::endl;
        return instance_id_to_object_id;
    }

    // Check semantic instances image type
    cv::Mat semantic_instances_image_int;
    if (semantic_instances_image.type() != CV_32S) {
        // Convert semantic instances image to int32
        std::cout << "VoxelBlockSemanticGridT::assign_object_ids_to_instance_ids: Semantic "
                     "instances image type: "
                  << semantic_instances_image.type() << std::endl;
        std::cout << "\tConverting semantic instances image to int32" << std::endl;
        semantic_instances_image.convertTo(semantic_instances_image_int, CV_32S);
    } else {
        semantic_instances_image_int = semantic_instances_image;
    }

    // Validate depth image if provided
    bool use_depth_filter = !depth_image.empty();
    cv::Mat depth_image_float;
    if (use_depth_filter) {
        if (depth_image.rows != camera_frustrum.get_height() ||
            depth_image.cols != camera_frustrum.get_width()) {
            std::cout << "VoxelBlockSemanticGridT::assign_object_ids_to_instance_ids: Depth "
                         "image size does not match camera frustrum size, disabling depth filter"
                      << std::endl;
            use_depth_filter = false;
        } else {
            // Convert depth image to float if needed
            if (depth_image.type() != CV_32F) {
                depth_image.convertTo(depth_image_float, CV_32F);
            } else {
                depth_image_float = depth_image;
            }
        }
    }

    // Structure to store votes: instance_id -> (object_id -> count)
    std::unordered_map<int, std::unordered_map<int, int>> instance_id_to_object_votes;

    const auto callback = [&](VoxelDataT &v, const VoxelKey &key,
                              const std::array<double, 3> &pos_w, const ImagePoint &image_point) {
        const float point_depth = image_point.depth;
        const int point_object_id = v.get_object_id();
        const int image_instance_id =
            semantic_instances_image_int.at<int>(image_point.v, image_point.u);

        // Skip invalid instance IDs (<0)
        if (image_instance_id < 0) {
            return;
        }

        // Skip invalid object IDs (<0)
        if (point_object_id < 0) {
            return;
        }

        // Depth filtering: remove associations to objects that are behind others
        if (use_depth_filter) {
            const float image_depth = depth_image_float.at<float>(image_point.v, image_point.u);

            // Skip invalid depth values (0, NaN, or inf)
            if (image_depth <= 0.0f || !std::isfinite(image_depth)) {
                return;
            }

            // Filter out voxels that are significantly behind the depth image
            // These are likely occluded or inconsistent
            if (point_depth > image_depth + depth_threshold) {
                return;
            }
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
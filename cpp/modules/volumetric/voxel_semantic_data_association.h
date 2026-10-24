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
#include <unordered_set>
#include <vector>

#include "camera_frustrum.h"
#include "image_utils.h"
#include "voxel_data.h"
#include "voxel_data_semantic.h"
#include "voxel_hashing.h"

#include <opencv2/opencv.hpp>

#ifdef TBB_FOUND
#include <tbb/enumerable_thread_specific.h>
#include <tbb/spin_mutex.h>
#endif

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
// - instance_ids_image: semantic instances image
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
MapInstanceIdToObjectId
assign_object_ids_to_instance_ids(VoxelGridT &voxel_grid, const CameraFrustrum &camera_frustrum,
                                  const cv::Mat &class_ids_image, const cv::Mat &instance_ids_image,
                                  const cv::Mat &depth_image, const float depth_threshold = 5e-2f,
                                  bool do_carving = false, const float min_vote_ratio = 0.5f,
                                  const int min_votes = 3, const int invalid_object_id = -1) {
    MapInstanceIdToObjectId instance_id_to_object_id;

    // check if semantic instances image is empty
    if (instance_ids_image.empty() || class_ids_image.empty()) {
        std::cout << "volumetric::assign_object_ids_to_instance_ids: class IDs or semantic "
                     "instances image is empty"
                  << std::endl;
        return instance_id_to_object_id;
    }
    // check the type of the semantic instances image
    if (instance_ids_image.channels() != 1) {
        throw std::runtime_error("Instance ids must be single-channel");
    }
    // check if semantic instances image size matches camera frustrum size
    if (!check_image_size(instance_ids_image, camera_frustrum.get_height(),
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
    cv::Mat instance_ids_image_int =
        convert_image_type_if_needed(instance_ids_image, CV_32S, "Semantic Instances");

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

    using VoteMap = std::unordered_map<int, std::map<int, int>>;
    using PendingMap = std::unordered_map<int, std::vector<VoxelDataT *>>;

#ifdef TBB_FOUND
    // Map to track new object IDs assigned to instance IDs for points without object_id
    // This ensures all points from the same instance_id get the same new object_id
    std::unordered_map<int, int> instance_id_to_new_object_id;
    tbb::spin_mutex instance_id_to_new_object_id_mutex;

    // Track votes: instance_id -> (object_id, count)
    // std::map is used to store the votes to get the most frequent object ID for each
    // instance ID (for a few votes, std::map is faster than std::unordered_map)
    tbb::enumerable_thread_specific<VoteMap> tls_votes;
    tbb::enumerable_thread_specific<PendingMap> tls_pending;
#else
    // Map to track new object IDs assigned to instance IDs for points without object_id
    // This ensures all points from the same instance_id get the same new object_id
    std::unordered_map<int, int> instance_id_to_new_object_id;

    // Track voxels that need object_id assignment after voting.
    std::unordered_map<int, std::vector<VoxelDataT *>> instance_id_to_pending_voxels;

    // Structure to store votes: instance_id -> (object_id, count)
    // std::map is used to store the votes to get the most frequent object ID for each
    // instance ID (for a few votes, std::map is faster than std::unordered_map)
    std::unordered_map<int, std::map<int, int>>
        instance_id_to_object_votes; // instance ID -> (object ID, count)
#endif

    auto assign_object_id = [&](VoxelDataT &v, const int image_instance_id, int &point_object_id,
                                auto &pending_map) {
#ifdef TBB_FOUND
        {
            tbb::spin_mutex::scoped_lock lock(instance_id_to_new_object_id_mutex);
            const auto it = instance_id_to_new_object_id.find(image_instance_id);
            if (it == instance_id_to_new_object_id.end()) {
                point_object_id = VoxelSemanticSharedData::get_next_object_id();
                instance_id_to_new_object_id[image_instance_id] = point_object_id;
            } else {
                point_object_id = it->second;
            }
        }
        pending_map[image_instance_id].push_back(&v);
#else
        const auto it = instance_id_to_new_object_id.find(image_instance_id);
        if (it == instance_id_to_new_object_id.end()) {
            point_object_id = VoxelSemanticSharedData::get_next_object_id();
            instance_id_to_new_object_id[image_instance_id] = point_object_id;
        } else {
            point_object_id = it->second;
        }
        pending_map[image_instance_id].push_back(&v);
#endif
    };

    auto process_point = [&](VoxelDataT &v, const ImagePoint &image_point, auto &votes_map,
                             auto &pending_map) {
        const float point_depth = image_point.depth;

        const int image_class_id = class_ids_image_int.at<int>(image_point.v, image_point.u);
        if (image_class_id < 0) {
            return;
        }

        const int point_class_id = v.get_class_id();
        if (point_class_id < 0) {
            return;
        }

        if (point_class_id != image_class_id) {
            return;
        }

        const int image_instance_id = instance_ids_image_int.at<int>(image_point.v, image_point.u);
        if (image_instance_id < 0) {
            return;
        }

        int point_object_id = v.get_object_id();

        if (use_depth_filter) {
            const float image_depth = depth_image_float.at<float>(image_point.v, image_point.u);
            // NOTE: If depth is invalid, skip the point
            if (image_depth <= 0.0f || !std::isfinite(image_depth)) {
                return;
            }

            if (do_carving) {
                if (point_depth < image_depth - depth_threshold) {
                    v.reset();
                    return;
                }
            }

            if (point_depth > image_depth + depth_threshold) {
                return;
            }
        }

        if (point_object_id < 0) {
            if (image_instance_id == 0) {
                point_object_id = 0;
                v.set_object_id(point_object_id);
            } else {
                assign_object_id(v, image_instance_id, point_object_id, pending_map);
            }
        }

        votes_map[image_instance_id][point_object_id]++;
    };

#ifdef TBB_FOUND
    const auto callback = [&](VoxelDataT &v, const VoxelKey &key,
                              const typename VoxelDataT::Pos3 &pos_w,
                              const ImagePoint &image_point) {
        (void)key;
        (void)pos_w;
        // Here, we process the point and update the votes and pending voxels in a thread-local
        // manner. This is done to avoid race conditions when updating the global votes and pending
        // voxels. After the parallel phase, we merge the thread-local votes and pending voxels into
        // the global votes and pending voxels.
        auto &local_votes = tls_votes.local();
        auto &local_pending = tls_pending.local();
        process_point(v, image_point, local_votes, local_pending);
    };
#else
    const auto callback = [&](VoxelDataT &v, const VoxelKey &key,
                              const typename VoxelDataT::Pos3 &pos_w,
                              const ImagePoint &image_point) {
        (void)key;
        (void)pos_w;
        process_point(v, image_point, instance_id_to_object_votes, instance_id_to_pending_voxels);
    };
#endif
    voxel_grid.iterate_voxels_in_camera_frustrum(camera_frustrum, callback);

#ifdef TBB_FOUND
    // Merge thread-local votes and pending voxels
    std::unordered_map<int, std::map<int, int>> instance_id_to_object_votes;
    std::unordered_map<int, std::vector<VoxelDataT *>> instance_id_to_pending_voxels;

    for (auto &local_votes : tls_votes) {
        for (auto &[instance_id, object_votes] : local_votes) {
            auto &dest_votes = instance_id_to_object_votes[instance_id];
            for (const auto &[object_id, count] : object_votes) {
                dest_votes[object_id] += count;
            }
        }
    }

    for (auto &local_pending : tls_pending) {
        for (auto &[instance_id, voxels] : local_pending) {
            auto &dest = instance_id_to_pending_voxels[instance_id];
            dest.insert(dest.end(), voxels.begin(), voxels.end());
        }
    }
#endif

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
            // Insufficient votes - mark as invalid
            instance_id_to_object_id[instance_id] = invalid_object_id;
            continue;
        }

        const float vote_ratio = static_cast<float>(max_votes) / static_cast<float>(total_votes);
        if (vote_ratio < min_vote_ratio) {
            // Ambiguous assignment - the winning object ID doesn't have enough dominance
            instance_id_to_object_id[instance_id] = invalid_object_id;
            continue;
        }

        // Assign the winning object ID to this instance ID
        instance_id_to_object_id[instance_id] = winning_object_id;
    }

    // Ensure every instance ID in the current image has a mapping.
    // This avoids reusing per-frame instance IDs as persistent object IDs.
    std::unordered_set<int> unmapped_image_instance_ids;
    unmapped_image_instance_ids.reserve(static_cast<size_t>(class_ids_image_int.rows) *
                                            static_cast<size_t>(class_ids_image_int.cols) / 16 +
                                        1);
    for (int r = 0; r < instance_ids_image_int.rows; ++r) {
        const int *instance_row = instance_ids_image_int.ptr<int>(r);
        const int *class_row = class_ids_image_int.ptr<int>(r);
        for (int c = 0; c < instance_ids_image_int.cols; ++c) {
            const int instance_id = instance_row[c];
            if (instance_id < 0) {
                continue;
            }
            const int class_id = class_row[c];
            if (class_id < 0) {
                continue;
            }
            unmapped_image_instance_ids.insert(instance_id);
        }
    }

    for (const int instance_id : unmapped_image_instance_ids) {
        if (instance_id == 0) {
            // Background/stuff: keep object_id = 0
            instance_id_to_object_id[0] = 0;
            continue;
        }
        if (instance_id_to_object_id.find(instance_id) == instance_id_to_object_id.end()) {
            instance_id_to_object_id[instance_id] =
                invalid_object_id; // VoxelSemanticSharedData::get_next_object_id();
        }
    }

    // Apply assignments for voxels that were deferred until voting finished.
    for (const auto &[instance_id, voxels] : instance_id_to_pending_voxels) {
        const auto it = instance_id_to_object_id.find(instance_id);
        if (it == instance_id_to_object_id.end()) {
            continue;
        }
        const int final_object_id = it->second;
        if (final_object_id < 0) {
            continue;
        }
        for (VoxelDataT *v : voxels) {
            if (v != nullptr) {
                v->set_object_id(final_object_id);
            }
        }
    }

    return instance_id_to_object_id;
}

} // namespace volumetric

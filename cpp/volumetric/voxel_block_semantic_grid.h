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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "camera_frustrum.h"
#include "voxel_block.h"
#include "voxel_block_grid.h"
#include "voxel_data.h"
#include "voxel_hashing.h"
#include "voxel_semantic_data_association.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#endif

#include <opencv2/opencv.hpp>

namespace py = pybind11;

namespace volumetric {

using MapInstanceIdToObjectId = std::unordered_map<int, int>; // instance ID -> object ID

// VoxelBlockSemanticGrid class with indirect voxel hashing (block-based) and semantic segmentation
// The space is divided into blocks of contiguous voxels (NxNxN)
// First, hashing identifies the block, then coordinates are transformed into the final voxel
template <typename VoxelDataT> class VoxelBlockSemanticGridT : public VoxelBlockGridT<VoxelDataT> {

    using Block = VoxelBlockT<VoxelDataT>;

  public:
    explicit VoxelBlockSemanticGridT(double voxel_size = 0.05, int block_size = 8);

    // Assign object IDs to 2d instance IDs using semantic instances image and depth image and class
    // IDs image. Inputs:
    // - camera_frustrum: camera frustrum
    // - semantic_instances_image: semantic instances image
    // - depth_image: depth image (optional, default is empty)
    // - depth_threshold: depth threshold (optional, default is 0.1f)
    // - min_vote_ratio: minimum vote ratio (optional, default is 0.5f)
    // - min_votes: minimum votes (optional, default is 3)
    // Returns: map of 2d instance IDs to object IDs
    MapInstanceIdToObjectId assign_object_ids_to_instance_ids(
        const CameraFrustrum &camera_frustrum, const cv::Mat &class_ids_image,
        const cv::Mat &semantic_instances_image, const cv::Mat &depth_image = cv::Mat(),
        const float depth_threshold = 0.1f, bool do_carving = false,
        const float min_vote_ratio = 0.5f, const int min_votes = 3);

    void set_depth_threshold(float depth_threshold);

    void set_depth_decay_rate(float depth_decay_rate);

    // Insert a segment of points (same instance and class IDs) into the voxel grid
    template <typename Tpos, typename Tcolor, typename Tinstance = int, typename Tclass = int>
    void integrate_segment(py::array_t<Tpos> points, py::array_t<Tcolor> colors,
                           const Tinstance &object_id, const Tclass &class_id);

    // Insert a cluster of points (same instance and class IDs) into the voxel grid
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = int,
              typename Tclass = int>
    void integrate_segment_raw(const Tpos *pts_ptr, const size_t num_points, const Tcolor *cols_ptr,
                               const Tclass &class_id, const Tinstance &object_id);

    // Merge two segments of voxels (different object IDs) into a single segment of voxels with
    // the same object ID instance_id1
    void merge_segments(const int instance_id1, const int instance_id2);

    // Remove all voxels with the specified object ID
    void remove_segment(const int object_id);

    // Remove all voxels with low confidence counter
    void remove_low_confidence_segments(const int min_confidence);

    // Get the class and object IDs of the voxel grid
    std::pair<std::vector<int>, std::vector<int>> get_ids() const;

    // Get clusters of voxels based on object IDs
    std::shared_ptr<volumetric::ObjectDataGroup>
    get_object_segments(const int min_count = 1, const float min_confidence = 0.0) const;

    // Get clusters of voxels based on class IDs
    std::shared_ptr<volumetric::ClassDataGroup>
    get_class_segments(const int min_count = 1, const float min_confidence = 0.0) const;
};

using VoxelBlockSemanticGrid = VoxelBlockSemanticGridT<VoxelSemanticData>;
using VoxelBlockSemanticGrid2 = VoxelBlockSemanticGridT<VoxelSemanticData2>;
using VoxelBlockSemanticProbabilisticGrid = VoxelBlockSemanticGridT<VoxelSemanticDataProbabilistic>;
using VoxelBlockSemanticProbabilisticGrid2 =
    VoxelBlockSemanticGridT<VoxelSemanticDataProbabilistic2>;

} // namespace volumetric

// Include implementation file
#include "voxel_block_semantic_grid.hpp"

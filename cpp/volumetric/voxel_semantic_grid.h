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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voxel_data.h"
#include "voxel_grid.h"
#include "voxel_grid_data.h"
#include "voxel_hashing.h"
#include "voxel_semantic_data_association.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

namespace py = pybind11;

namespace volumetric {

// VoxelSemanticGrid class with direct voxel hashing and semantic segmentation
template <typename VoxelDataT> class VoxelSemanticGridT : public VoxelGridT<VoxelDataT> {

  public:
    using PosScalar = typename VoxelDataT::PosScalar;
    using ColorScalar = typename VoxelDataT::ColorScalar;
    using Pos3 = typename VoxelDataT::Pos3;
    using Color3 = typename VoxelDataT::Color3;

    using VoxelGridDataType = VoxelGridDataT<PosScalar, ColorScalar>;

  public:
    explicit VoxelSemanticGridT(double voxel_size = 0.05);

    // Assign object IDs to 2d instance IDs using semantic instances image and depth image
    // Inputs:
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

    // Insert a cluster of points (same instance and class IDs) into the voxel grid
    // Uses pre-partitioning to avoid race conditions: groups points by voxel key first,
    // then processes each voxel group serially in parallel. This ensures each voxel is
    // updated by exactly one thread, eliminating concurrent access to the same VoxelDataT object.
    template <typename Tp, typename Tc>
    void integrate_segment_raw(const Tp *pts_ptr, const size_t num_points, const Tc *cols_ptr,
                               const int class_id, const int object_id);

    // Merge two segments of voxels (different object IDs) into a single segment of voxels with
    // the same object ID instance_id1
    void merge_segments(const int instance_id1, const int instance_id2);

    // Remove all voxels with the specified object ID
    void remove_segment(const int object_id);

    // Remove all voxels with low confidence counter
    void remove_low_confidence_segments(const int min_confidence);

    // // Get the points of the voxel grid
    // std::vector<Pos3> get_points() const;

    // // Get the colors of the voxel grid
    // std::vector<Color3> get_colors() const;

    // Get the class and object IDs of the voxel grid
    std::pair<std::vector<int>, std::vector<int>> get_ids() const;

    // Get clusters of voxels based on object IDs
    std::shared_ptr<volumetric::ObjectDataGroup>
    get_object_segments(const int min_count = 1, const float min_confidence = 0.0) const;

    // Get clusters of voxels based on class IDs
    std::shared_ptr<volumetric::ClassDataGroup>
    get_class_segments(const int min_count = 1, const float min_confidence = 0.0) const;

    // Clear the voxel grid
    void clear();

    // Get the size of the voxel grid
    size_t size() const;

    // Check if the voxel grid is empty
    bool empty() const;

  private:
    double voxel_size_;
    double inv_voxel_size_; // Precomputed 1.0 / voxel_size_ for faster division
#ifdef TBB_FOUND
    tbb::concurrent_unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#else
    std::unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#endif
};

using VoxelSemanticGrid = VoxelSemanticGridT<VoxelSemanticData>;
using VoxelSemanticGrid2 = VoxelSemanticGridT<VoxelSemanticData2>;
using VoxelSemanticGridProbabilistic = VoxelSemanticGridT<VoxelSemanticDataProbabilistic>;
using VoxelSemanticGridProbabilistic2 = VoxelSemanticGridT<VoxelSemanticDataProbabilistic2>;

} // namespace volumetric

// Include implementation file
#include "voxel_semantic_grid.hpp"

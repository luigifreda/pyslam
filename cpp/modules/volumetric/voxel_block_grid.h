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
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "camera_frustrum.h"
#include "voxel_block.h"
#include "voxel_data.h"
#include "voxel_data_semantic.h"
#include "voxel_grid_carving.h"
#include "voxel_grid_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

#include <opencv2/opencv.hpp>

namespace py = pybind11;

namespace volumetric {

// VoxelBlockGridT class with indirect voxel hashing (block-based)
// The space is divided into blocks of contiguous voxels (NxNxN)
// First, hashing identifies the block, then coordinates are transformed into the final voxel
template <typename VoxelDataT> class VoxelBlockGridT {

  public:
    using PosScalar = typename VoxelDataT::PosScalar;
    using ColorScalar = typename VoxelDataT::ColorScalar;
    using Pos3 = typename VoxelDataT::Pos3;
    using Color3 = typename VoxelDataT::Color3;

    using Block = VoxelBlockT<VoxelDataT>;
    using VoxelGridDataType = VoxelGridDataT<PosScalar, ColorScalar>;

  public:
    VoxelBlockGridT(float voxel_size = 0.05, int block_size = 8);

    // Integrate with std::vector inputs
    // Inputs:
    // - points: vector of points
    // - colors: vector of colors (optional, can be empty)
    // - instance_ids: vector of instance ids (optional, can be empty)
    // - class_ids: vector of class ids (optional, can be empty)
    // - depths: vector of camera depths (w.r.t camera frame) (optional, can be empty)
    template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
    void integrate(const std::vector<Tpos> &points, const std::vector<Tcolor> &colors,
                   const std::vector<Tclass> &class_ids, const std::vector<Tinstance> &instance_ids,
                   const std::vector<Tdepth> &depths);

    // Raw integration method with raw pointers (can be called without GIL)
    // Inputs:
    // - pts_ptr: pointer to the points array
    // - num_points: number of points
    // - cols_ptr: pointer to the colors array (optional)
    // - class_ids_ptr: pointer to the class ids array (optional)
    // - instance_ids_ptr: pointer to the instance ids array (optional)
    // - depths_ptr: pointer to the camera depths array (optional)
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void integrate_raw(const Tpos *pts_ptr, size_t num_points, const Tcolor *cols_ptr = nullptr,
                       const Tclass *class_ids_ptr = nullptr,
                       const Tinstance *instance_ids_ptr = nullptr,
                       const Tdepth *depths_ptr = nullptr);

    // Carve the voxel grid using the camera frustrum and camera depths.
    // Reset all the voxels that are inside the camera frustrum and have a depth less than their
    // camera depth - depth threshold.
    // Inputs:
    // - camera_frustrum: the camera frustrum
    // - depth_image: the depth image
    // - depth_threshold: the depth threshold for carving (default: 1e-3)
    void carve(const CameraFrustrum &camera_frustrum, const cv::Mat &depth_image,
               const float depth_threshold = 1e-2);

    // Remove all voxels with low confidence counter
    void remove_low_count_voxels(const int min_count);
    // Remove all voxels with low confidence counter
    void remove_low_confidence_voxels(const float min_confidence);

    std::vector<Pos3> get_points() const;

    std::vector<Color3> get_colors() const;

    VoxelGridDataType get_voxels(int min_count = 1, float min_confidence = 0.0) const;

    // Get voxels within a spatial interval (bounding box)
    // Returns points and colors for voxels whose centers fall within [min_xyz, max_xyz]
    // If IncludeSemantics is true and VoxelDataT is a SemanticVoxel, also returns semantic data
    template <bool IncludeSemantics = false>
    VoxelGridDataType get_voxels_in_bb(const BoundingBox3D &bbox, const int min_count = 1,
                                       float min_confidence = 0.0) const;

    // Get voxels within a camera frustrum
    // Returns points and colors for voxels whose centers fall within the camera frustrum
    // If IncludeSemantics is true and VoxelDataT is a SemanticVoxel, also returns semantic data
    template <bool IncludeSemantics = false>
    VoxelGridDataType get_voxels_in_camera_frustrum(const CameraFrustrum &camera_frustrum,
                                                    const int min_count = 1,
                                                    float min_confidence = 0.0) const;

    // Iterate over voxels in a spatial interval with a callback function
    // The callback receives (voxel_key, voxel_data) for each voxel in the interval
    template <typename Callback>
    void iterate_voxels_in_bb(const BoundingBox3D &bbox, Callback &&callback, int min_count = 1,
                              float min_confidence = 0.0) const;

    // Iterate over voxels in a spatial interval with a callback function
    // The callback receives (voxel_key, voxel_data) for each voxel in the interval
    template <typename Callback>
    void iterate_voxels_in_camera_frustrum(const CameraFrustrum &camera_frustrum,
                                           Callback &&callback, int min_count = 1,
                                           float min_confidence = 0.0);

    // Clear the voxel grid
    void clear();

    // Get the number of blocks
    size_t num_blocks() const;

    // Get the total number of voxels
    size_t size() const;

    // Check if the voxel grid is empty
    bool empty() const;

    // Get block size
    int get_block_size() const;

    // Helper function to count total voxels
    size_t get_total_voxel_count() const;

  protected:
    // Implementation with per-point parallelization and mutex protection for each block.
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void integrate_raw_baseline(const Tpos *pts_ptr, size_t num_points,
                                const Tcolor *cols_ptr = nullptr,
                                const Tclass *class_ids_ptr = nullptr,
                                const Tinstance *instance_ids_ptr = nullptr,
                                const Tdepth *depths_ptr = nullptr);

    // Optimized implementation with block-based grouping to minimize mutex contention
    // Groups points by block key using hash map (O(n)).
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void integrate_raw_preorder_no_block_mutex(const Tpos *pts_ptr, size_t num_points,
                                               const Tcolor *cols_ptr = nullptr,
                                               const Tclass *class_ids_ptr = nullptr,
                                               const Tinstance *instance_ids_ptr = nullptr,
                                               const Tdepth *depths_ptr = nullptr);

    // Helper function to update voxel (internal, with LockMutex template parameter)
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t,
              bool LockMutex = true>
    void update_voxel(const Tpos x, const Tpos y, const Tpos z, const Tcolor color_x = nullptr,
                      const Tcolor color_y = nullptr, const Tcolor color_z = nullptr,
                      const Tclass class_id = nullptr, const Tinstance object_id = nullptr,
                      const Tdepth depth = nullptr);

    // Convenience wrappers for update_voxel with explicit locking behavior
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void update_voxel_lock(const Tpos x, const Tpos y, const Tpos z, const Tcolor color_x = nullptr,
                           const Tcolor color_y = nullptr, const Tcolor color_z = nullptr,
                           const Tclass class_id = nullptr, const Tinstance object_id = nullptr,
                           const Tdepth depth = nullptr);

    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void update_voxel_no_lock(const Tpos x, const Tpos y, const Tpos z,
                              const Tcolor color_x = nullptr, const Tcolor color_y = nullptr,
                              const Tcolor color_z = nullptr, const Tclass class_id = nullptr,
                              const Tinstance object_id = nullptr, const Tdepth depth = nullptr);

  private:
    // Internal helper: update voxel directly using pre-computed Block& and LocalVoxelKey
    // This avoids recomputing keys and re-fetching blocks, used for optimized block-grouped paths
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void update_voxel_direct(Block &block, const LocalVoxelKey &local_key, const Tpos x,
                             const Tpos y, const Tpos z, const Tcolor color_x = nullptr,
                             const Tcolor color_y = nullptr, const Tcolor color_z = nullptr,
                             const Tclass class_id = nullptr, const Tinstance object_id = nullptr,
                             const Tdepth depth = nullptr);

  protected:
    float voxel_size_;
    float inv_voxel_size_;     // Precomputed 1.0f / voxel_size_ for faster division
    int block_size_;           // number of voxels per side of the block
    int num_voxels_per_block_; // number of voxels per block (block_size_^3)
#ifdef TBB_FOUND
    tbb::concurrent_unordered_map<BlockKey, Block, BlockKeyHash> blocks_;
#else
    std::unordered_map<BlockKey, Block, BlockKeyHash> blocks_;
#endif
};

using VoxelBlockGrid = VoxelBlockGridT<VoxelData>;

} // namespace volumetric

// Include implementation file
#include "voxel_block_grid.hpp"

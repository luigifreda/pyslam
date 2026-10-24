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

#include "bounding_boxes.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

// SIMD support
#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD 1
#define USE_AVX2 1
#define USE_DOUBLE_SIMD 1
#elif defined(__SSE4_1__) || defined(__SSE4_2__)
#include <emmintrin.h> // SSE2 for double SIMD (__m128d)
#include <smmintrin.h> // SSE4.1 includes floor_ps and floor_pd
#define USE_SIMD 1
#define USE_AVX2 0
#define USE_DOUBLE_SIMD 1
#else
#define USE_SIMD 0
#define USE_AVX2 0
#define USE_DOUBLE_SIMD 0
#endif

#include "camera_frustrum.h"
#include "voxel_data.h"
#include "voxel_data_semantic.h"
#include "voxel_grid_carving.h"
#include "voxel_grid_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

#include <opencv2/opencv.hpp>

namespace py = pybind11;

namespace volumetric {

// VoxelGrid class with direct voxel hashing
template <typename VoxelDataT> class VoxelGridT {

  public:
    using PosScalar = typename VoxelDataT::PosScalar;
    using ColorScalar = typename VoxelDataT::ColorScalar;
    using Pos3 = typename VoxelDataT::Pos3;
    using Color3 = typename VoxelDataT::Color3;

    using VoxelGridDataType = VoxelGridDataT<PosScalar, ColorScalar>;

  public:
    explicit VoxelGridT(float voxel_size = 0.05);

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
    template <typename Tp, typename Tc = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void integrate_raw(const Tp *pts_ptr, size_t num_points, const Tc *cols_ptr = nullptr,
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

    // Iterate over voxels in a camera frustrum with a callback function
    // The callback receives (voxel_key, voxel_data) for each voxel in the frustrum
    template <typename Callback>
    void iterate_voxels_in_camera_frustrum(const CameraFrustrum &camera_frustrum,
                                           Callback &&callback, int min_count = 1,
                                           float min_confidence = 0.0);

    // Clear the voxel grid
    void clear();

    // Get the size of the voxel grid
    size_t size() const;

    // Check if the voxel grid is empty
    bool empty() const;

  protected:
    // Semantic-aware integration implementation using optimized pre-partitioning
    template <typename Tp, typename Tc, typename Tinstance, typename Tclass, typename Tdepth>
    void integrate_raw_semantic_impl(const Tp *pts_ptr, size_t num_points, const Tc *cols_ptr,
                                     const Tclass *class_ids_ptr, const Tinstance *instance_ids_ptr,
                                     const Tdepth *depths_ptr);

    template <typename Tp, typename Tc>
    void integrate_raw_scalar(const Tp *pts_ptr, const Tc *cols_ptr, size_t num_points);

    template <typename Tp> void integrate_raw_scalar(const Tp *pts_ptr, size_t num_points);

    // Optimized scalar version using pre-partitioning (default implementation)
    // Groups points by voxel key first, then processes each voxel group serially in parallel.
    // This avoids race conditions without needing mutexes: each voxel is updated by exactly
    // one thread, eliminating concurrent access to the same VoxelDataT object.
    template <typename Tp, typename Tc, bool HasColors>
    void integrate_raw_scalar_impl_with_prepartitioning(const Tp *pts_ptr, const Tc *cols_ptr,
                                                        const size_t num_points);

    // Alternative scalar version using per-voxel mutexes (for comparison/fallback)
    // NOTE: This implementation fixes a race condition in the original code.
    // tbb::concurrent_unordered_map only protects the container structure (insertions/deletions),
    // NOT the contained VoxelDataT objects. Without mutexes, concurrent updates to the same voxel
    // (position_sum, color_sum, count) would race and silently corrupt data.
    template <typename Tp, typename Tc, bool HasColors>
    void integrate_raw_scalar_impl_with_mutexes(const Tp *pts_ptr, const Tc *cols_ptr,
                                                const size_t num_points);

#if USE_SIMD

    template <bool HasColors>
    void integrate_raw_simd_impl(const float *pts_ptr, const float *cols_ptr,
                                 const size_t num_points);

    // Wrapper functions that call the unified template implementation
    void integrate_raw_simd(const float *pts_ptr, const float *cols_ptr, size_t num_points);

    void integrate_raw_simd(const float *pts_ptr, size_t num_points);
#endif

#if USE_DOUBLE_SIMD

    template <bool HasColors>
    void integrate_raw_simd_impl_double(const double *pts_ptr, const float *cols_ptr,
                                        const size_t num_points);

    // Wrapper functions for double precision points
    void integrate_raw_simd(const double *pts_ptr, const float *cols_ptr, const size_t num_points);

    void integrate_raw_simd(const double *pts_ptr, const size_t num_points);

#endif

  private:
    float voxel_size_;
    float inv_voxel_size_; // Precomputed 1.0f / voxel_size_ for faster division
#ifdef TBB_FOUND
    tbb::concurrent_unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#else
    std::unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#endif
};

using VoxelGrid = VoxelGridT<VoxelData>;

} // namespace volumetric

// Include implementation file
#include "voxel_grid.hpp"

// Include SIMD implementations
#include "voxel_grid_simd.hpp"
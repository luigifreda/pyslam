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

#include "voxel_data.h"
#include "voxel_grid_data.h"
#include "voxel_hashing.h"

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

// VoxelGrid class with direct voxel hashing
template <typename VoxelDataT> class VoxelGridT {
  public:
    explicit VoxelGridT(float voxel_size = 0.05)
        : voxel_size_(voxel_size), inv_voxel_size_(1.0f / voxel_size) {
        static_assert(Voxel<VoxelDataT>, "VoxelDataT must satisfy the Voxel concept");
    }

    // Insert a point cloud into the voxel grid
    template <typename Tp, typename Tc>
    void integrate(py::array_t<Tp> points, py::array_t<Tc> colors) {
        auto pts_info = points.request();
        auto cols_info = colors.request();

        // Validate array shapes: points and colors should be (N, 3) or have 3*N elements
        if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
            throw std::runtime_error("points must be a 2D array with shape (N, 3)");
        }
        if (cols_info.ndim != 2 || cols_info.shape[1] != 3) {
            throw std::runtime_error("colors must be a 2D array with shape (N, 3)");
        }
        if (pts_info.shape[0] != cols_info.shape[0]) {
            throw std::runtime_error("points and colors must have the same number of rows");
        }
        if (!pts_info.ptr || !cols_info.ptr) {
            throw std::runtime_error("points and colors arrays must be contiguous");
        }

        integrate_raw<Tp, Tc>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0],
                              static_cast<const Tc *>(cols_info.ptr));
    }

    // Internal method that does the actual integration work (can be called without GIL)
    template <typename Tp, typename Tc>
    void integrate_raw(const Tp *pts_ptr, size_t num_points, const Tc *cols_ptr) {
        // Use SIMD for float types when available
        if constexpr (std::is_same_v<Tp, float> && std::is_same_v<Tc, float> && USE_SIMD) {
            integrate_raw_simd(pts_ptr, cols_ptr, num_points);
        } else if constexpr (std::is_same_v<Tp, double> && std::is_same_v<Tc, float> &&
                             USE_DOUBLE_SIMD) {
            integrate_raw_simd(pts_ptr, cols_ptr, num_points);
        } else {
            integrate_raw_scalar(pts_ptr, cols_ptr, num_points);
        }
    }

    // Internal method that does the actual integration work (can be called without GIL)
    template <typename Tp> void integrate_raw(const Tp *pts_ptr, size_t num_points) {
        // Use SIMD for float types when available
        if constexpr (std::is_same_v<Tp, float> && USE_SIMD) {
            integrate_raw_simd(pts_ptr, num_points);
        } else if constexpr (std::is_same_v<Tp, double> && USE_DOUBLE_SIMD) {
            integrate_raw_simd(pts_ptr, num_points);
        } else {
            integrate_raw_scalar(pts_ptr, num_points);
        }
    }

    template <typename Tp, typename Tc>
    void integrate_raw_scalar(const Tp *pts_ptr, const Tc *cols_ptr, size_t num_points) {
        // Use pre-partitioning approach by default (thread-safe, efficient)
        integrate_raw_scalar_impl_with_prepartitioning<Tp, Tc, std::is_same_v<Tc, float>>(
            pts_ptr, cols_ptr, num_points);
    }

    template <typename Tp> void integrate_raw_scalar(const Tp *pts_ptr, size_t num_points) {
        // Use pre-partitioning approach by default (thread-safe, efficient)
        integrate_raw_scalar_impl_with_prepartitioning<Tp, float, false>(
            pts_ptr, nullptr, num_points); // fake float type for colors
    }

    // Optimized scalar version using pre-partitioning (default implementation)
    // Groups points by voxel key first, then processes each voxel group serially in parallel.
    // This avoids race conditions without needing mutexes: each voxel is updated by exactly
    // one thread, eliminating concurrent access to the same VoxelDataT object.
    template <typename Tp, typename Tc, bool HasColors>
    void integrate_raw_scalar_impl_with_prepartitioning(const Tp *pts_ptr, const Tc *cols_ptr,
                                                        size_t num_points) {
#ifdef TBB_FOUND
        // Pre-partition points by voxel key to avoid race conditions
        struct PointInfo {
            Tp x, y, z;
            Tc color_x, color_y, color_z;
        };
        tbb::concurrent_unordered_map<VoxelKey, std::vector<PointInfo>, VoxelKeyHash> voxel_groups;

        // Phase 1: Group points by voxel key in parallel (thread-local accumulation)
        std::mutex merge_mutex;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_points),
            [&](const tbb::blocked_range<size_t> &range) {
                // Thread-local map for this range
                std::unordered_map<VoxelKey, std::vector<PointInfo>, VoxelKeyHash> local_groups;
                local_groups.reserve(64); // Pre-allocate for typical voxel count

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const size_t idx = i * 3;
                    const Tp x = pts_ptr[idx + 0];
                    const Tp y = pts_ptr[idx + 1];
                    const Tp z = pts_ptr[idx + 2];

                    const VoxelKey key = get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

                    PointInfo info;
                    info.x = x;
                    info.y = y;
                    info.z = z;
                    if constexpr (HasColors) {
                        info.color_x = cols_ptr[idx + 0];
                        info.color_y = cols_ptr[idx + 1];
                        info.color_z = cols_ptr[idx + 2];
                    }

                    local_groups[key].push_back(info);
                }

                // Merge local groups into global concurrent map
                std::lock_guard<std::mutex> lock(merge_mutex);
                for (auto &[key, points] : local_groups) {
                    auto it = voxel_groups.find(key);
                    if (it == voxel_groups.end()) {
                        auto result = voxel_groups.insert({key, std::vector<PointInfo>()});
                        it = result.first;
                    }
                    auto &global_vec = it->second;
                    global_vec.insert(global_vec.end(), points.begin(), points.end());
                }
            });

        // Phase 2: Process each voxel group in parallel (one thread per voxel)
        tbb::parallel_for_each(voxel_groups.begin(), voxel_groups.end(), [&](const auto &pair) {
            const VoxelKey &key = pair.first;
            const std::vector<PointInfo> &points = pair.second;

            // Get or create voxel (concurrent map is thread-safe)
            auto [it, inserted] = grid_.insert({key, VoxelDataT()});
            auto &v = it->second;

            // Update all points for this voxel serially (by this thread only)
            bool is_first = true;
            for (const auto &info : points) {
                if (inserted || (is_first && v.count == 0)) {
                    // New voxel or reset voxel: initialize and update
                    v.update_point(info.x, info.y, info.z);
                    if constexpr (HasColors) {
                        v.update_color(info.color_x, info.color_y, info.color_z);
                    }
                    v.count = 1;
                    inserted = false; // Only first point is "inserted"
                    is_first = false;
                } else {
                    // Existing voxel: accumulate
                    v.update_point(info.x, info.y, info.z);
                    if constexpr (HasColors) {
                        v.update_color(info.color_x, info.color_y, info.color_z);
                    }
                    ++v.count;
                }
            }
        });
#else
        // Sequential version
        for (size_t i = 0; i < num_points; ++i) {
            const size_t idx = i * 3;
            const Tp x = pts_ptr[idx + 0];
            const Tp y = pts_ptr[idx + 1];
            const Tp z = pts_ptr[idx + 2];

            Tc color_x, color_y, color_z;
            if constexpr (HasColors) {
                color_x = cols_ptr[idx + 0];
                color_y = cols_ptr[idx + 1];
                color_z = cols_ptr[idx + 2];
            }

            const VoxelKey key = get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

            // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
            auto [it, inserted] = grid_.try_emplace(key);
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize and update
                // If count==0, the voxel was reset and should be treated as new
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                v.count = 1;
            } else {
                // Existing voxel: just update
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                ++v.count;
            }
        }
#endif
    }

    // Alternative scalar version using per-voxel mutexes (for comparison/fallback)
    // NOTE: This implementation fixes a race condition in the original code.
    // tbb::concurrent_unordered_map only protects the container structure (insertions/deletions),
    // NOT the contained VoxelDataT objects. Without mutexes, concurrent updates to the same voxel
    // (position_sum, color_sum, count) would race and silently corrupt data.
    template <typename Tp, typename Tc, bool HasColors>
    void integrate_raw_scalar_impl_with_mutexes(const Tp *pts_ptr, const Tc *cols_ptr,
                                                size_t num_points) {
#ifdef TBB_FOUND
        // Hash map of mutexes (one per voxel key)
        tbb::concurrent_unordered_map<VoxelKey, std::unique_ptr<std::mutex>, VoxelKeyHash>
            voxel_mutexes;
        std::mutex mutex_map_mutex; // For creating new mutexes

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_points),
            [&](const tbb::blocked_range<size_t> &range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const size_t idx = i * 3;
                    const Tp x = pts_ptr[idx + 0];
                    const Tp y = pts_ptr[idx + 1];
                    const Tp z = pts_ptr[idx + 2];

                    Tc color_x, color_y, color_z;
                    if constexpr (HasColors) {
                        color_x = cols_ptr[idx + 0];
                        color_y = cols_ptr[idx + 1];
                        color_z = cols_ptr[idx + 2];
                    }

                    const VoxelKey key = get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

                    // Get or create mutex for this voxel
                    auto mutex_it = voxel_mutexes.find(key);
                    if (mutex_it == voxel_mutexes.end()) {
                        std::lock_guard<std::mutex> lock(mutex_map_mutex);
                        mutex_it = voxel_mutexes.find(key); // Double-check
                        if (mutex_it == voxel_mutexes.end()) {
                            mutex_it =
                                voxel_mutexes.insert({key, std::make_unique<std::mutex>()}).first;
                        }
                    }
                    std::mutex &voxel_mutex = *mutex_it->second;

                    // Lock and update voxel
                    std::lock_guard<std::mutex> lock(voxel_mutex);
                    auto [it, inserted] = grid_.insert({key, VoxelDataT()});
                    auto &v = it->second;

                    if (inserted || v.count == 0) {
                        // New voxel or reset voxel: initialize and update
                        v.update_point(x, y, z);
                        if constexpr (HasColors) {
                            v.update_color(color_x, color_y, color_z);
                        }
                        v.count = 1;
                    } else {
                        // Existing voxel: just update
                        v.update_point(x, y, z);
                        if constexpr (HasColors) {
                            v.update_color(color_x, color_y, color_z);
                        }
                        ++v.count;
                    }
                }
            });
#else
        // Sequential version
        for (size_t i = 0; i < num_points; ++i) {
            const size_t idx = i * 3;
            const Tp x = pts_ptr[idx + 0];
            const Tp y = pts_ptr[idx + 1];
            const Tp z = pts_ptr[idx + 2];

            Tc color_x, color_y, color_z;
            if constexpr (HasColors) {
                color_x = cols_ptr[idx + 0];
                color_y = cols_ptr[idx + 1];
                color_z = cols_ptr[idx + 2];
            }

            const VoxelKey key = get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

            // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
            auto [it, inserted] = grid_.try_emplace(key);
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize and update
                // If count==0, the voxel was reset and should be treated as new
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                v.count = 1;
            } else {
                // Existing voxel: just update
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                ++v.count;
            }
        }
#endif
    }

#if USE_SIMD

    template <bool HasColors>
    void integrate_raw_simd_impl(const float *pts_ptr, const float *cols_ptr, size_t num_points);

    // Wrapper functions that call the unified template implementation
    void integrate_raw_simd(const float *pts_ptr, const float *cols_ptr, size_t num_points) {
        integrate_raw_simd_impl<true>(pts_ptr, cols_ptr, num_points);
    }

    void integrate_raw_simd(const float *pts_ptr, size_t num_points) {
        integrate_raw_simd_impl<false>(pts_ptr, nullptr, num_points);
    }
#endif

#if USE_DOUBLE_SIMD

    template <bool HasColors>
    void integrate_raw_simd_impl_double(const double *pts_ptr, const float *cols_ptr,
                                        size_t num_points);

    // Wrapper functions for double precision points
    void integrate_raw_simd(const double *pts_ptr, const float *cols_ptr, size_t num_points) {
        integrate_raw_simd_impl_double<true>(pts_ptr, cols_ptr, num_points);
    }

    void integrate_raw_simd(const double *pts_ptr, size_t num_points) {
        integrate_raw_simd_impl_double<false>(pts_ptr, nullptr, num_points);
    }

#endif

    // Insert a point cloud into the voxel grid
    template <typename Tp> void integrate_points(py::array_t<Tp> points) {
        auto pts_info = points.request();

        // Validate array shape: points should be (N, 3)
        if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
            throw std::runtime_error("points must be a 2D array with shape (N, 3)");
        }
        if (!pts_info.ptr) {
            throw std::runtime_error("points array must be contiguous");
        }

        integrate_raw_scalar<Tp, float>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0]);
    }

    std::vector<std::array<double, 3>> get_points() const {
        std::vector<std::array<double, 3>> points;
        points.reserve(grid_.size());
#ifdef TBB_FOUND
        // Parallel version: collect keys first, then process in parallel
        // Use isolate() to prevent deadlock from nested parallelism
        std::vector<VoxelKey> keys;
        keys.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            keys.push_back(key);
        }
        points.resize(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      points[i] = v.get_position();
                                  }
                              });
        });
#else
        // Sequential version
        for (const auto &[key, v] : grid_) {
            points.push_back(v.get_position());
        }
#endif
        return points;
    }

    std::vector<std::array<float, 3>> get_colors() const {
        std::vector<std::array<float, 3>> colors;
        colors.reserve(grid_.size());
#ifdef TBB_FOUND
        // Parallel version: collect keys first, then process in parallel
        // Use isolate() to prevent deadlock from nested parallelism
        std::vector<VoxelKey> keys;
        keys.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            keys.push_back(key);
        }
        colors.resize(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      colors[i] = v.get_color();
                                  }
                              });
        });
#else
        // Sequential version
        for (const auto &[key, v] : grid_) {
            colors.push_back(v.get_color());
        }
#endif
        return colors;
    }

    std::pair<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>>
    get_voxel_data(int min_count = 1) const {
#ifdef TBB_FOUND
        // Parallel version: collect keys first, filter, then process in parallel
        // Use isolate() to prevent deadlock from nested parallelism
        std::vector<VoxelKey> keys;
        keys.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            if (v.count >= min_count) {
                keys.push_back(key);
            }
        }
        std::vector<std::array<double, 3>> points(keys.size());
        std::vector<std::array<float, 3>> colors(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      points[i] = v.get_position();
                                      colors[i] = v.get_color();
                                  }
                              });
        });
        return {points, colors};
#else
        // Sequential version
        std::vector<std::array<double, 3>> points;
        points.reserve(grid_.size());
        std::vector<std::array<float, 3>> colors;
        colors.reserve(grid_.size());

        // Always filter by min_count to exclude reset voxels (count=0) unless explicitly requested
        for (const auto &[key, v] : grid_) {
            if (v.count >= min_count) {
                points.push_back(v.get_position());
                colors.push_back(v.get_color());
            }
        }
        return {points, colors};
#endif
    }

    // Get voxels within a spatial interval (bounding box)
    // Returns points and colors for voxels whose centers fall within [min_xyz, max_xyz]
    // If IncludeSemantics is true and VoxelDataT is a SemanticVoxel, also returns semantic data
    template <typename T, bool IncludeSemantics = false>
    VoxelGridData get_voxels_in_interval(const T min_x, const T min_y, const T min_z, const T max_x,
                                         const T max_y, const T max_z,
                                         const int min_count = 1) const {
        // Convert spatial bounds to voxel key bounds
        const VoxelKey min_key = get_voxel_key_inv<T, float>(min_x, min_y, min_z, inv_voxel_size_);
        const VoxelKey max_key = get_voxel_key_inv<T, float>(max_x, max_y, max_z, inv_voxel_size_);

        const size_t estimated_num_voxels =
            (max_key.x - min_key.x + 1) * (max_key.y - min_key.y + 1) * (max_key.z - min_key.z + 1);

#ifdef TBB_FOUND
        // Parallel version: use thread-local storage to collect data directly
        // Each thread processes distinct voxels, so no contention on reads
        VoxelGridData result;
        std::mutex merge_mutex; // Single mutex for merging thread-local results

        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(min_key.x, max_key.x + 1, [&](KeyType kx) {
                // Thread-local storage for collecting data (reduces contention)
                VoxelGridData local_result;
                local_result.points.reserve((max_key.y - min_key.y + 1) *
                                            (max_key.z - min_key.z + 1));
                local_result.colors.reserve((max_key.y - min_key.y + 1) *
                                            (max_key.z - min_key.z + 1));
                if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                    local_result.instance_ids.reserve((max_key.y - min_key.y + 1) *
                                                      (max_key.z - min_key.z + 1));
                    local_result.class_ids.reserve((max_key.y - min_key.y + 1) *
                                                   (max_key.z - min_key.z + 1));
                    local_result.confidence_counters.reserve((max_key.y - min_key.y + 1) *
                                                             (max_key.z - min_key.z + 1));
                }

                for (KeyType ky = min_key.y; ky <= max_key.y; ++ky) {
                    for (KeyType kz = min_key.z; kz <= max_key.z; ++kz) {
                        VoxelKey key(kx, ky, kz);
                        auto it = grid_.find(key);
                        if (it != grid_.end()) {
                            const auto &v = it->second;
                            if (v.count >= min_count) {
                                // Check if voxel center is actually within bounds
                                auto pos = v.get_position();
                                if (pos[0] >= static_cast<double>(min_x) &&
                                    pos[0] <= static_cast<double>(max_x) &&
                                    pos[1] >= static_cast<double>(min_y) &&
                                    pos[1] <= static_cast<double>(max_y) &&
                                    pos[2] >= static_cast<double>(min_z) &&
                                    pos[2] <= static_cast<double>(max_z)) {
                                    local_result.points.push_back(pos);
                                    local_result.colors.push_back(v.get_color());

                                    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                        local_result.instance_ids.push_back(v.get_instance_id());
                                        local_result.class_ids.push_back(v.get_class_id());
                                        local_result.confidence_counters.push_back(
                                            v.get_confidence_counter());
                                    }
                                }
                            }
                        }
                    }
                }

                // Merge local results into global result (protected by mutex)
                if (!local_result.points.empty()) {
                    std::lock_guard<std::mutex> lock(merge_mutex);
                    result.points.insert(result.points.end(), local_result.points.begin(),
                                         local_result.points.end());
                    result.colors.insert(result.colors.end(), local_result.colors.begin(),
                                         local_result.colors.end());
                    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                        result.instance_ids.insert(result.instance_ids.end(),
                                                   local_result.instance_ids.begin(),
                                                   local_result.instance_ids.end());
                        result.class_ids.insert(result.class_ids.end(),
                                                local_result.class_ids.begin(),
                                                local_result.class_ids.end());
                        result.confidence_counters.insert(result.confidence_counters.end(),
                                                          local_result.confidence_counters.begin(),
                                                          local_result.confidence_counters.end());
                    }
                }
            });
        });
#else
        // Sequential version
        VoxelGridData result;
        result.points.reserve(estimated_num_voxels);
        result.colors.reserve(estimated_num_voxels);
        if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
            result.instance_ids.reserve(estimated_num_voxels);
            result.class_ids.reserve(estimated_num_voxels);
            result.confidence_counters.reserve(estimated_num_voxels);
        }

        // Iterate over all voxel keys in the interval
        for (KeyType kx = min_key.x; kx <= max_key.x; ++kx) {
            for (KeyType ky = min_key.y; ky <= max_key.y; ++ky) {
                for (KeyType kz = min_key.z; kz <= max_key.z; ++kz) {
                    VoxelKey key(kx, ky, kz);
                    auto it = grid_.find(key);
                    if (it != grid_.end()) {
                        const auto &v = it->second;
                        if (v.count >= min_count) {
                            // Check if voxel center is actually within bounds
                            // (voxel key bounds may be slightly larger than spatial bounds)
                            auto pos = v.get_position();
                            if (pos[0] >= static_cast<double>(min_x) &&
                                pos[0] <= static_cast<double>(max_x) &&
                                pos[1] >= static_cast<double>(min_y) &&
                                pos[1] <= static_cast<double>(max_y) &&
                                pos[2] >= static_cast<double>(min_z) &&
                                pos[2] <= static_cast<double>(max_z)) {
                                result.points.push_back(pos);
                                result.colors.push_back(v.get_color());

                                // Extract semantic data if requested and available
                                if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                    result.instance_ids.push_back(v.get_instance_id());
                                    result.class_ids.push_back(v.get_class_id());
                                    result.confidence_counters.push_back(
                                        v.get_confidence_counter());
                                }
                            }
                        }
                    }
                }
            }
        }
#endif

        return result;
    }

    // Iterate over voxels in a spatial interval with a callback function
    // The callback receives (voxel_key, voxel_data) for each voxel in the interval
    template <typename T, typename Callback>
    void iterate_voxels_in_interval(T min_x, T min_y, T min_z, T max_x, T max_y, T max_z,
                                    Callback &&callback, int min_count = 1) const {
        // Convert spatial bounds to voxel key bounds
        const VoxelKey min_key = get_voxel_key_inv<T, float>(min_x, min_y, min_z, inv_voxel_size_);
        const VoxelKey max_key = get_voxel_key_inv<T, float>(max_x, max_y, max_z, inv_voxel_size_);

        // Iterate over all voxel keys in the interval
        for (KeyType kx = min_key.x; kx <= max_key.x; ++kx) {
            for (KeyType ky = min_key.y; ky <= max_key.y; ++ky) {
                for (KeyType kz = min_key.z; kz <= max_key.z; ++kz) {
                    VoxelKey key(kx, ky, kz);
                    auto it = grid_.find(key);
                    if (it != grid_.end()) {
                        const auto &v = it->second;
                        if (v.count >= min_count) {
                            // Check if voxel center is actually within bounds
                            auto pos = v.get_position();
                            if (pos[0] >= static_cast<double>(min_x) &&
                                pos[0] <= static_cast<double>(max_x) &&
                                pos[1] >= static_cast<double>(min_y) &&
                                pos[1] <= static_cast<double>(max_y) &&
                                pos[2] >= static_cast<double>(min_z) &&
                                pos[2] <= static_cast<double>(max_z)) {
                                callback(key, v);
                            }
                        }
                    }
                }
            }
        }
    }

    // Get voxels within a spatial interval using array input (for Python interface)
    // bounds should be a 1D array with 6 elements: [min_x, min_y, min_z, max_x, max_y, max_z]
    template <typename T, bool IncludeSemantics = false>
    auto get_voxels_in_interval_array(py::array_t<T> bounds, int min_count = 1) const {
        auto bounds_info = bounds.request();

        // Validate array shape: bounds should be (6,) or have 6 elements
        if (bounds_info.ndim != 1 || bounds_info.shape[0] != 6) {
            throw std::runtime_error("bounds must be a 1D array with 6 elements: [min_x, min_y, "
                                     "min_z, max_x, max_y, max_z]");
        }
        if (!bounds_info.ptr) {
            throw std::runtime_error("bounds array must be contiguous");
        }

        const T *bounds_ptr = static_cast<const T *>(bounds_info.ptr);
        return get_voxels_in_interval<T, IncludeSemantics>(bounds_ptr[0], bounds_ptr[1],
                                                           bounds_ptr[2], bounds_ptr[3],
                                                           bounds_ptr[4], bounds_ptr[5], min_count);
    }

    // Clear the voxel grid
    void clear() { grid_.clear(); }

    // Get the size of the voxel grid
    size_t size() const { return grid_.size(); }

    // Check if the voxel grid is empty
    bool empty() const { return grid_.empty(); }

  private:
    float voxel_size_;
    float inv_voxel_size_; // Precomputed 1.0f / voxel_size_ for faster division
#ifdef TBB_FOUND
    tbb::concurrent_unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#else
    std::unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#endif
};

// Include SIMD implementations (out-of-class definitions)
#include "voxel_grid_simd.h"

using VoxelGrid = VoxelGridT<VoxelData>;

} // namespace volumetric
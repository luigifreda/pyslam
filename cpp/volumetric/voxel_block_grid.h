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

#include "voxel_block.h"
#include "voxel_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#endif

namespace py = pybind11;

namespace volumetric {

// VoxelBlockGridT class with indirect voxel hashing (block-based)
// The space is divided into blocks of contiguous voxels (NxNxN)
// First, hashing identifies the block, then coordinates are transformed into the final voxel
template <typename VoxelDataT> class VoxelBlockGridT {

    using Block = VoxelBlockT<VoxelDataT>;

  public:
    VoxelBlockGridT(float voxel_size = 0.05, int block_size = 8)
        : voxel_size_(voxel_size), inv_voxel_size_(1.0f / voxel_size), block_size_(block_size) {
        static_assert(Voxel<VoxelDataT>, "VoxelDataT must satisfy the Voxel concept");
        num_voxels_per_block_ = block_size_ * block_size_ * block_size_;
    }

    // Insert a point cloud into the voxel grid
    template <typename Tpos, typename Tcolor>
    void integrate(py::array_t<Tpos> points, py::array_t<Tcolor> colors) {
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

        integrate_raw<Tpos, Tcolor>(static_cast<const Tpos *>(pts_info.ptr), pts_info.shape[0],
                                    static_cast<const Tcolor *>(cols_info.ptr));
    }

    // Insert a point cloud into the voxel grid (points only, no colors)
    template <typename Tpos> void integrate(py::array_t<Tpos> points) {
        auto pts_info = points.request();

        // Validate array shape: points should be (N, 3)
        if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
            throw std::runtime_error("points must be a 2D array with shape (N, 3)");
        }
        if (!pts_info.ptr) {
            throw std::runtime_error("points array must be contiguous");
        }

        integrate_raw<Tpos>(static_cast<const Tpos *>(pts_info.ptr), pts_info.shape[0]);
    }

    // Implementation
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t>
    void integrate_raw(const Tpos *pts_ptr, size_t num_points, const Tcolor *cols_ptr = nullptr,
                       const Tinstance *instance_ids_ptr = nullptr,
                       const Tclass *class_ids_ptr = nullptr) {
        // Here we actually select the implementation of the integration function.
#if 1
        // Implementation with
        // 1) Preliminary parallelized block-based grouping to minimize mutex contention
        // 2) Per-block parallelized integration without block-level mutex protection
        integrate_raw_preorder_no_block_mutex<Tpos, Tcolor, Tinstance, Tclass>(
            pts_ptr, num_points, cols_ptr, instance_ids_ptr, class_ids_ptr);
#else
        // Implementation with parallelized per-point integration with block-level mutex protection
        integrate_raw_baseline<Tpos, Tcolor, Tinstance, Tclass>(pts_ptr, num_points, cols_ptr,
                                                                instance_ids_ptr, class_ids_ptr);
#endif
    }

    // Implementation with per-point parallelization and mutex protection for each block.
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t>
    void integrate_raw_baseline(const Tpos *pts_ptr, size_t num_points,
                                const Tcolor *cols_ptr = nullptr,
                                const Tinstance *instance_ids_ptr = nullptr,
                                const Tclass *class_ids_ptr = nullptr) {
#ifdef TBB_FOUND
        // Parallel version using TBB with concurrent_unordered_map (thread-safe, no mutex needed)
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_points),
            [&](const tbb::blocked_range<size_t> &range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
#else
        // Sequential version
        for (size_t i = 0; i < num_points; ++i) {
#endif
                    const size_t idx = i * 3;
                    const Tpos x = pts_ptr[idx + 0];
                    const Tpos y = pts_ptr[idx + 1];
                    const Tpos z = pts_ptr[idx + 2];

                    if constexpr (std::is_same_v<Tcolor, std::nullptr_t>) {
                        // No colors: call overload without color parameters
                        update_voxel<Tpos>(x, y, z);
                    } else {
                        // Colors provided: read colors and call with color parameters
                        const Tcolor color_x = cols_ptr[idx + 0];
                        const Tcolor color_y = cols_ptr[idx + 1];
                        const Tcolor color_z = cols_ptr[idx + 2];

                        if constexpr (std::is_same_v<Tinstance, std::nullptr_t> &&
                                      std::is_same_v<Tclass, std::nullptr_t>) {
                            // No semantics
                            update_voxel<Tpos, Tcolor>(x, y, z, color_x, color_y, color_z);
                        } else {
                            // With semantics
                            const Tinstance instance_id = instance_ids_ptr[i];
                            const Tclass class_id = class_ids_ptr[i];
                            update_voxel<Tpos, Tcolor, Tinstance, Tclass>(
                                x, y, z, color_x, color_y, color_z, instance_id, class_id);
                        }
                    }
                }
#ifdef TBB_FOUND
            });
#endif
    }

    // Optimized implementation with block-based grouping to minimize mutex contention
    // Groups points by block key using hash map (O(n)).
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t>
    void integrate_raw_preorder_no_block_mutex(const Tpos *pts_ptr, size_t num_points,
                                               const Tcolor *cols_ptr = nullptr,
                                               const Tinstance *instance_ids_ptr = nullptr,
                                               const Tclass *class_ids_ptr = nullptr) {
#ifdef TBB_FOUND
        // Group point indices by block key using concurrent hash map
        struct PointInfo {
            size_t point_idx;
            LocalVoxelKey local_key;
        };
        tbb::concurrent_unordered_map<BlockKey, std::vector<PointInfo>, BlockKeyHash> block_groups;

        // Precompute keys and group points by block in parallel
        // Use thread-local storage to minimize contention, then merge with mutex
        std::mutex merge_mutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points), [&](auto r) {
            // Thread-local map for this range
            std::unordered_map<BlockKey, std::vector<PointInfo>, BlockKeyHash> local_groups;
            local_groups.reserve(64); // Pre-allocate for typical block count

            for (size_t i = r.begin(); i < r.end(); ++i) {
                const size_t idx = i * 3;
                const Tpos x = pts_ptr[idx + 0], y = pts_ptr[idx + 1], z = pts_ptr[idx + 2];
                const VoxelKey vk = get_voxel_key_inv<Tpos, float>(x, y, z, inv_voxel_size_);
                const BlockKey block_key = get_block_key(vk, block_size_);
                const LocalVoxelKey local_key = get_local_voxel_key(vk, block_key, block_size_);
                local_groups[block_key].push_back({i, local_key});
            }

            // Merge local groups into global concurrent map with mutex protection
            // This ensures thread-safe vector operations
            std::lock_guard<std::mutex> lock(merge_mutex);
            for (auto &[key, points] : local_groups) {
                // Get or create the vector for this block key
                auto it = block_groups.find(key);
                if (it == block_groups.end()) {
                    // Insert a new entry
                    auto result = block_groups.insert({key, std::vector<PointInfo>()});
                    it = result.first;
                }
                // Append points to the vector (protected by mutex)
                auto &global_vec = it->second;
                global_vec.insert(global_vec.end(), points.begin(), points.end());
            }
        });

        // Process each block in parallel (one thread per block)
        tbb::parallel_for_each(block_groups.begin(), block_groups.end(), [&](auto &pair) {
            const BlockKey &block_key = pair.first;
            const std::vector<PointInfo> &points = pair.second;

            // Create/find the block once (concurrent map → no external lock needed)
            // Note: tbb::concurrent_unordered_map::insert() returns a pair<iterator, bool>
            auto [it, inserted] = blocks_.insert(std::make_pair(block_key, Block(block_size_)));
            Block &block = it->second;

            // Update all points in this block serially (by this thread only)
            for (const auto &info : points) {
                const size_t i = info.point_idx;
                const size_t idx = i * 3;
                const Tpos x = pts_ptr[idx + 0], y = pts_ptr[idx + 1], z = pts_ptr[idx + 2];

                auto &v = block.get_voxel(info.local_key);

                if (v.count == 0) {
                    v.update_point(x, y, z);
                    if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                        const Tcolor cx = cols_ptr[idx + 0], cy = cols_ptr[idx + 1],
                                     cz = cols_ptr[idx + 2];
                        v.update_color(cx, cy, cz);
                    }
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        if constexpr (!std::is_same_v<Tinstance, std::nullptr_t> &&
                                      !std::is_same_v<Tclass, std::nullptr_t>) {
                            v.initialize_semantics(instance_ids_ptr[i], class_ids_ptr[i]);
                        }
                    }
                    v.count = 1;
                } else {
                    v.update_point(x, y, z);
                    if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                        const Tcolor cx = cols_ptr[idx + 0], cy = cols_ptr[idx + 1],
                                     cz = cols_ptr[idx + 2];
                        v.update_color(cx, cy, cz);
                    }
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        if constexpr (!std::is_same_v<Tinstance, std::nullptr_t> &&
                                      !std::is_same_v<Tclass, std::nullptr_t>) {
                            v.update_semantics(instance_ids_ptr[i], class_ids_ptr[i]);
                        }
                    }
                    ++v.count;
                }
            }
        });
#else
        // Sequential version: fall back to integrate_raw_old
        integrate_raw_baseline<Tpos, Tcolor, Tinstance, Tclass>(pts_ptr, num_points, cols_ptr,
                                                                instance_ids_ptr, class_ids_ptr);
#endif
    }

    // Helper function to update voxel with colors
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t>
    void update_voxel(const Tpos x, const Tpos y, const Tpos z, const Tcolor color_x = nullptr,
                      const Tcolor color_y = nullptr, const Tcolor color_z = nullptr,
                      const Tinstance instance_id = nullptr, const Tclass class_id = nullptr) {
        // Compute voxel coordinates
        const VoxelKey voxel_key = get_voxel_key_inv<Tpos, float>(x, y, z, inv_voxel_size_);

        // Compute block coordinates using helper function from
        // voxel_hashing.h
        const BlockKey block_key = get_block_key(voxel_key, block_size_);
        const LocalVoxelKey local_key = get_local_voxel_key(voxel_key, block_key, block_size_);

        // Get or create the block (concurrent_unordered_map is thread-safe)
        // Note: tbb::concurrent_unordered_map::insert() returns a pair<iterator, bool>
        auto [block_it, inserted] = blocks_.insert(std::make_pair(block_key, Block(block_size_)));
        Block &block = block_it->second;

        // Acquire mutex BEFORE accessing voxel data to prevent race conditions
#ifdef TBB_FOUND
        std::lock_guard<std::mutex> lock(*block.mutex);
#endif
        // Get voxel using direct array access (thread-safe with mutex)
        auto &v = block.get_voxel(local_key);

        if (v.count == 0) {
            // New voxel: initialize and update
            v.update_point(x, y, z);
            if constexpr ((!std::is_same_v<Tcolor, std::nullptr_t>)) {
                v.update_color(color_x, color_y, color_z);
            }
            if constexpr (SemanticVoxel<VoxelDataT>) {
                if constexpr ((!std::is_same_v<Tinstance, std::nullptr_t>) &&
                              (!std::is_same_v<Tclass, std::nullptr_t>)) {
                    v.initialize_semantics(instance_id, class_id);
                }
            }
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            if constexpr ((!std::is_same_v<Tcolor, std::nullptr_t>)) {
                v.update_color(color_x, color_y, color_z);
            }
            if constexpr (SemanticVoxel<VoxelDataT>) {
                if constexpr ((!std::is_same_v<Tinstance, std::nullptr_t>) &&
                              (!std::is_same_v<Tclass, std::nullptr_t>)) {
                    v.update_semantics(instance_id, class_id);
                }
            }
            ++v.count;
        }
    }

    // Remove all voxels with low confidence counter
    void remove_low_count_voxels(const int min_count) {
#ifdef TBB_FOUND
        // Parallel version
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](auto &pair) {
            Block &block = pair.second;
            std::lock_guard<std::mutex> lock(*block.mutex);
            for (auto &v : block.data) {
                if (v.count < min_count) {
                    v.reset();
                }
            }
        });
#else
        // Sequential version
        for (auto &[block_key, block] : blocks_) {
            for (auto &v : block.data) {
                if (v.count < min_count) {
                    v.reset();
                }
            }
        }
#endif
    }

    std::vector<std::array<double, 3>> get_points() const {
        std::vector<std::array<double, 3>> points;
        points.reserve(get_total_voxel_count());

        for (const auto &[block_key, block] : blocks_) {
            for (const auto &v : block.data) {
                if (v.count > 0) {
                    points.push_back(v.get_position());
                }
            }
        }
        return points;
    }

    std::vector<std::array<float, 3>> get_colors() const {
        std::vector<std::array<float, 3>> colors;
        colors.reserve(get_total_voxel_count());

        for (const auto &[block_key, block] : blocks_) {
            for (const auto &v : block.data) {
                if (v.count > 0) {
                    colors.push_back(v.get_color());
                }
            }
        }
        return colors;
    }

    std::pair<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>>
    get_voxel_data(int min_count = 1) const {
        std::vector<std::array<double, 3>> points;
        std::vector<std::array<float, 3>> colors;
        const size_t upper_bound_num_voxels = num_voxels_per_block_ * blocks_.size();
        points.reserve(upper_bound_num_voxels);
        colors.reserve(upper_bound_num_voxels);

        for (const auto &[block_key, block] : blocks_) {
            for (const auto &v : block.data) {
                if (v.count >= min_count) {
                    points.push_back(v.get_position());
                    colors.push_back(v.get_color());
                }
            }
        }
        return {points, colors};
    }

    // Clear the voxel grid
    void clear() { blocks_.clear(); }

    // Get the number of blocks
    size_t num_blocks() const { return blocks_.size(); }

    // Get the total number of voxels
    size_t size() const { return get_total_voxel_count(); }

    // Check if the voxel grid is empty
    bool empty() const { return blocks_.empty(); }

    // Get block size
    int get_block_size() const { return block_size_; }

    // Helper function to count total voxels
    size_t get_total_voxel_count() const {
        size_t total = 0;
        for (const auto &[block_key, block] : blocks_) {
            total += block.count_active();
        }
        return total;
    }

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

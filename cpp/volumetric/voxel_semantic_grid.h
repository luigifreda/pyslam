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
#include "voxel_hashing.h"

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
    explicit VoxelSemanticGridT(double voxel_size = 0.05)
        : voxel_size_(voxel_size), inv_voxel_size_(1.0 / voxel_size) {
        static_assert(SemanticVoxel<VoxelDataT>,
                      "VoxelDataT must satisfy the SemanticVoxel concept");
    }

    void set_depth_threshold(float depth_threshold) {
        if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
            VoxelDataT::kDepthThreshold = depth_threshold;
        }
    }

    void set_depth_decay_rate(float depth_decay_rate) {
        if constexpr (std::is_same_v<VoxelDataT, VoxelSemanticDataProbabilistic>) {
            VoxelDataT::kDepthDecayRate = depth_decay_rate;
        }
    }

    // Insert a point cloud into the voxel grid
    template <typename Tp, typename Tc>
    void integrate(py::array_t<Tp> points, py::array_t<Tc> colors, py::array_t<int> instance_ids,
                   py::array_t<int> class_ids) {

        auto pts_info = points.request();
        auto cols_info = colors.request();
        auto instance_ids_info = instance_ids.request();
        auto class_ids_info = class_ids.request();

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

        // check all have same size
        if (pts_info.shape[0] != instance_ids_info.shape[0] ||
            pts_info.shape[0] != class_ids_info.shape[0]) {
            throw std::runtime_error(
                "points, instance_ids, and class_ids must have the same number of rows");
        }

        integrate_raw<Tp, Tc>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0],
                              static_cast<const Tc *>(cols_info.ptr),
                              static_cast<const int *>(instance_ids_info.ptr),
                              static_cast<const int *>(class_ids_info.ptr));
    }

    // Insert a point cloud into the voxel grid (with optional depth-based confidence)
    // Uses pre-partitioning to avoid race conditions: groups points by voxel key first,
    // then processes each voxel group serially in parallel. This ensures each voxel is
    // updated by exactly one thread, eliminating concurrent access to the same VoxelDataT object.
    template <typename Tp, typename Tcolor = std::nullptr_t, typename Tinstance = std::nullptr_t,
              typename Tclass = std::nullptr_t, typename Tdepth = std::nullptr_t>
    void integrate_raw(const Tp *pts_ptr, size_t num_points, const Tcolor *cols_ptr = nullptr,
                       const Tinstance *instance_ids_ptr = nullptr,
                       const Tclass *class_ids_ptr = nullptr, const Tdepth *depths_ptr = nullptr) {
#ifdef TBB_FOUND
        // Pre-partition points by voxel key to avoid race conditions
        // Helper types to handle nullptr_t template parameters
        using ColorType = std::conditional_t<std::is_same_v<Tcolor, std::nullptr_t>, float, Tcolor>;
        using InstanceType =
            std::conditional_t<std::is_same_v<Tinstance, std::nullptr_t>, int, Tinstance>;
        using ClassType = std::conditional_t<std::is_same_v<Tclass, std::nullptr_t>, int, Tclass>;
        using DepthType = std::conditional_t<std::is_same_v<Tdepth, std::nullptr_t>, float, Tdepth>;

        struct PointInfo {
            Tp x, y, z;
            ColorType color_x, color_y, color_z;
            InstanceType instance_id;
            ClassType class_id;
            DepthType depth;
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

                    const VoxelKey key = get_voxel_key_inv<Tp, double>(x, y, z, inv_voxel_size_);

                    PointInfo info;
                    info.x = x;
                    info.y = y;
                    info.z = z;
                    if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                        info.color_x = cols_ptr[idx + 0];
                        info.color_y = cols_ptr[idx + 1];
                        info.color_z = cols_ptr[idx + 2];
                    }
                    if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                        info.instance_id = instance_ids_ptr[i];
                    }
                    if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                        info.class_id = class_ids_ptr[i];
                    }
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        info.depth = depths_ptr[i];
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
                    if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                        v.update_color(info.color_x, info.color_y, info.color_z);
                    }
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                            // we have semantics/class id
                            if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                                // we have instance id
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                        v.initialize_semantics_with_depth(
                                            info.instance_id, info.class_id, info.depth);
                                    } else {
                                        v.initialize_semantics(info.instance_id, info.class_id);
                                    }
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    v.initialize_semantics(info.instance_id, info.class_id);
                                }
                            } else {
                                // we do not have instance id => use default instance id 1
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                        v.initialize_semantics_with_depth(1, info.class_id,
                                                                          info.depth);
                                    } else {
                                        v.initialize_semantics(1, info.class_id);
                                    }
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    v.initialize_semantics(1, info.class_id);
                                }
                            }
                        }
                    }
                    v.count = 1;
                    inserted = false; // Only first point is "inserted"
                    is_first = false;
                } else {
                    // Existing voxel: accumulate
                    v.update_point(info.x, info.y, info.z);
                    if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                        v.update_color(info.color_x, info.color_y, info.color_z);
                    }
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                            // we have semantics/class id
                            if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                                // we have instance id
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                        v.update_semantics_with_depth(info.instance_id,
                                                                      info.class_id, info.depth);
                                    } else {
                                        v.update_semantics(info.instance_id, info.class_id);
                                    }
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    v.update_semantics(info.instance_id, info.class_id);
                                }
                            } else {
                                // we do not have instance id => use default instance id 1
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                        v.update_semantics_with_depth(1, info.class_id, info.depth);
                                    } else {
                                        v.update_semantics(1, info.class_id);
                                    }
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    v.update_semantics(1, info.class_id);
                                }
                            }
                        }
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

            const VoxelKey key = get_voxel_key_inv<Tp, double>(x, y, z, inv_voxel_size_);

            // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
            auto [it, inserted] = grid_.try_emplace(key);
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize and update
                // If count==0, the voxel was reset and should be treated as new
                v.update_point(x, y, z);
                if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                    const Tcolor color_x = cols_ptr[idx + 0];
                    const Tcolor color_y = cols_ptr[idx + 1];
                    const Tcolor color_z = cols_ptr[idx + 2];
                    v.update_color(color_x, color_y, color_z);
                }
                if constexpr (SemanticVoxel<VoxelDataT>) {
                    if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                        // we have semantics/class id
                        if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                            // we have instance id
                            const Tinstance instance_id = instance_ids_ptr[i];
                            const Tclass class_id = class_ids_ptr[i];
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.initialize_semantics_with_depth(instance_id, class_id,
                                                                      depths_ptr[i]);
                                } else {
                                    v.initialize_semantics(instance_id, class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.initialize_semantics(instance_id, class_id);
                            }
                        } else {
                            // we do not have instance id => use default instance id 1
                            const Tclass class_id = class_ids_ptr[i];
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.initialize_semantics_with_depth(1, class_id, depths_ptr[i]);
                                } else {
                                    v.initialize_semantics(1, class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.initialize_semantics(1, class_id);
                            }
                        }
                    }
                }
                v.count = 1;
            } else {
                // Existing voxel: just update
                v.update_point(x, y, z);
                if constexpr (!std::is_same_v<Tcolor, std::nullptr_t>) {
                    const Tcolor color_x = cols_ptr[idx + 0];
                    const Tcolor color_y = cols_ptr[idx + 1];
                    const Tcolor color_z = cols_ptr[idx + 2];
                    v.update_color(color_x, color_y, color_z);
                }
                if constexpr (SemanticVoxel<VoxelDataT>) {
                    if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                        // we have semantics/class id
                        if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                            // we have instance id
                            const Tinstance instance_id = instance_ids_ptr[i];
                            const Tclass class_id = class_ids_ptr[i];
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.update_semantics_with_depth(instance_id, class_id,
                                                                  depths_ptr[i]);
                                } else {
                                    v.update_semantics(instance_id, class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.update_semantics(instance_id, class_id);
                            }
                        } else {
                            // we do not have instance id => use default instance id 1
                            const Tclass class_id = class_ids_ptr[i];
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.update_semantics_with_depth(1, class_id, depths_ptr[i]);
                                } else {
                                    v.update_semantics(1, class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.update_semantics(1, class_id);
                            }
                        }
                    }
                }
                ++v.count;
            }
        }
#endif
    }

    // Insert a segment of points (same instance and class IDs) into the voxel grid
    template <typename Tp, typename Tc>
    void integrate_segment(py::array_t<Tp> points, py::array_t<Tc> colors, const int instance_id,
                           const int class_id) {
        auto pts_info = points.request();
        auto cols_info = colors.request();
        integrate_segment_raw<Tp, Tc>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0],
                                      static_cast<const Tc *>(cols_info.ptr), instance_id,
                                      class_id);
    }

    // Insert a cluster of points (same instance and class IDs) into the voxel grid
    // Uses pre-partitioning to avoid race conditions: groups points by voxel key first,
    // then processes each voxel group serially in parallel. This ensures each voxel is
    // updated by exactly one thread, eliminating concurrent access to the same VoxelDataT object.
    template <typename Tp, typename Tc>
    void integrate_segment_raw(const Tp *pts_ptr, const size_t num_points, const Tc *cols_ptr,
                               const int instance_id, const int class_id) {

        // Skip points with invalid instance or class IDs
        if (instance_id < 0 || class_id < 0) {
            return;
        }

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

                    const VoxelKey key = get_voxel_key_inv<Tp, double>(x, y, z, inv_voxel_size_);

                    PointInfo info;
                    info.x = x;
                    info.y = y;
                    info.z = z;
                    info.color_x = cols_ptr[idx + 0];
                    info.color_y = cols_ptr[idx + 1];
                    info.color_z = cols_ptr[idx + 2];

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
                    v.update_color(info.color_x, info.color_y, info.color_z);
                    v.initialize_semantics(instance_id, class_id);
                    v.count = 1;
                    inserted = false; // Only first point is "inserted"
                    is_first = false;
                } else {
                    // Existing voxel: accumulate
                    v.update_point(info.x, info.y, info.z);
                    v.update_color(info.color_x, info.color_y, info.color_z);
                    v.update_semantics(instance_id, class_id);
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

            const Tc color_x = cols_ptr[idx + 0];
            const Tc color_y = cols_ptr[idx + 1];
            const Tc color_z = cols_ptr[idx + 2];

            const VoxelKey key = get_voxel_key_inv<Tp, double>(x, y, z, inv_voxel_size_);

            // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
            auto [it, inserted] = grid_.try_emplace(key);
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize and update
                // If count==0, the voxel was reset and should be treated as new
                v.update_point(x, y, z);
                v.update_color(color_x, color_y, color_z);
                v.initialize_semantics(instance_id, class_id);
                v.count = 1;
            } else {
                // Existing voxel: just update
                v.update_point(x, y, z);
                v.update_color(color_x, color_y, color_z);
                v.update_semantics(instance_id, class_id);
                ++v.count;
            }
        }
#endif
    }

    // Merge two segments of voxels (different instance IDs) into a single segment of voxels with
    // the same instance ID instance_id1
    void merge_segments(const int instance_id1, const int instance_id2) {
#ifdef TBB_FOUND
        // Parallel version using TBB - concurrent_unordered_map is thread-safe
        // Use isolate() to prevent deadlock from nested parallelism
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for_each(grid_.begin(), grid_.end(), [&](auto &pair) {
                if (pair.second.get_instance_id() == instance_id2) {
                    // concurrent_unordered_map allows safe concurrent modification
                    pair.second.set_instance_id(instance_id1);
                }
            });
        });
#else
        // Sequential version
        for (auto &[key, v] : grid_) {
            if (v.get_instance_id() == instance_id2) {
                v.set_instance_id(instance_id1);
            }
        }
#endif
    }

    // Remove all voxels with the specified instance ID
    void remove_segment(const int instance_id) {
#ifdef TBB_FOUND
        // For concurrent_unordered_map, collect keys first, then erase
        std::vector<VoxelKey> keys_to_remove;
        keys_to_remove.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            if (v.get_instance_id() == instance_id) {
                keys_to_remove.push_back(key);
            }
        }
        for (const auto &key : keys_to_remove) {
            grid_.unsafe_erase(key);
        }
#else
        // Sequential version for std::unordered_map
        for (auto it = grid_.begin(); it != grid_.end();) {
            if (it->second.get_instance_id() == instance_id) {
                it = grid_.erase(it);
            } else {
                ++it;
            }
        }
#endif
    }

    // Remove all voxels with low confidence counter
    void remove_low_confidence_segments(const int min_confidence_counter) {
#ifdef TBB_FOUND
        // For concurrent_unordered_map, collect keys first, then erase
        std::vector<VoxelKey> keys_to_remove;
        keys_to_remove.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            // Use getter method if available (for probabilistic), otherwise direct member access
            if (v.get_confidence_counter() < min_confidence_counter) {
                keys_to_remove.push_back(key);
            }
        }
        for (const auto &key : keys_to_remove) {
            grid_.unsafe_erase(key);
        }
#else
        // Sequential version for std::unordered_map
        for (auto it = grid_.begin(); it != grid_.end();) {
            // Use getter method if available (for probabilistic), otherwise direct member access
            if (it->second.get_confidence_counter() < min_confidence_counter) {
                // Remove the voxel with low confidence_counter
                it = grid_.erase(it);
            } else {
                ++it;
            }
        }
#endif
    }

    // Get the points of the voxel grid
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

    // Get the colors of the voxel grid
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

    // Get the class and instance IDs of the voxel grid
    std::pair<std::vector<int>, std::vector<int>> get_ids() const {
        std::vector<int> class_ids;
        class_ids.reserve(grid_.size());
        std::vector<int> instance_ids;
        instance_ids.reserve(grid_.size());
#ifdef TBB_FOUND
        // Parallel version: collect keys first, then process in parallel
        // Use isolate() to prevent deadlock from nested parallelism
        std::vector<VoxelKey> keys;
        keys.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            keys.push_back(key);
        }
        class_ids.resize(keys.size());
        instance_ids.resize(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      class_ids[i] = v.get_class_id();
                                      instance_ids[i] = v.get_instance_id();
                                  }
                              });
        });
#else
        // Sequential version
        for (const auto &[key, v] : grid_) {
            class_ids.push_back(v.get_class_id());
            instance_ids.push_back(v.get_instance_id());
        }
#endif
        return {class_ids, instance_ids};
    }

    std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>,
               std::vector<int>, std::vector<int>>
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
        std::vector<int> class_ids(keys.size());
        std::vector<int> instance_ids(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      points[i] = v.get_position();
                                      colors[i] = v.get_color();
                                      class_ids[i] = v.get_class_id();
                                      instance_ids[i] = v.get_instance_id();
                                  }
                              });
        });
        return {points, colors, class_ids, instance_ids};
#else
        // Sequential version
        std::vector<std::array<double, 3>> points;
        points.reserve(grid_.size());
        std::vector<std::array<float, 3>> colors;
        colors.reserve(grid_.size());
        std::vector<int> class_ids;
        class_ids.reserve(grid_.size());
        std::vector<int> instance_ids;
        instance_ids.reserve(grid_.size());
        // Always filter by min_count to exclude reset voxels (count=0) unless explicitly requested
        for (const auto &[key, v] : grid_) {
            if (v.count >= min_count) {
                points.push_back(v.get_position());
                colors.push_back(v.get_color());
                class_ids.push_back(v.get_class_id());
                instance_ids.push_back(v.get_instance_id());
            }
        }
        return {points, colors, class_ids, instance_ids};
#endif
    }

    // Get clusters of voxels based on instance IDs
    std::unordered_map<int, std::vector<std::array<double, 3>>> get_segments() const {
        std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
        for (const auto &[key, v] : grid_) {
            segments[v.get_instance_id()].push_back(v.get_position());
        }
        return segments;
    }

    // Clear the voxel grid
    void clear() { grid_.clear(); }

    // Get the size of the voxel grid
    size_t size() const { return grid_.size(); }

    // Check if the voxel grid is empty
    bool empty() const { return grid_.empty(); }

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
using VoxelSemanticGridProbabilistic = VoxelSemanticGridT<VoxelSemanticDataProbabilistic>;

} // namespace volumetric
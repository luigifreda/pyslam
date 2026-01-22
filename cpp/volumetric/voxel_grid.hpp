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

// Implementation file for VoxelGridT template class
// This file should be included at the end of voxel_grid.h

namespace volumetric {

// Constructor implementation
template <typename VoxelDataT>
VoxelGridT<VoxelDataT>::VoxelGridT(float voxel_size)
    : voxel_size_(voxel_size), inv_voxel_size_(1.0f / voxel_size) {
    static_assert(Voxel<VoxelDataT>, "VoxelDataT must satisfy the Voxel concept");
}

// integrate with std::vector inputs
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelGridT<VoxelDataT>::integrate(const std::vector<Tpos> &points,
                                       const std::vector<Tcolor> &colors,
                                       const std::vector<Tclass> &class_ids,
                                       const std::vector<Tinstance> &instance_ids,
                                       const std::vector<Tdepth> &depths) {

    if (points.empty()) {
        return;
    }
    // check if we have colors
    const bool has_colors = !colors.empty();
    if (has_colors && points.size() != colors.size()) {
        throw std::runtime_error("points and colors must have the same size");
    }
    // check if we have depths
    const bool has_depths = !depths.empty();
    if (has_depths && points.size() != depths.size()) {
        throw std::runtime_error("points and depths must have the same size");
    }
    // Check if we have semantic data
    const bool has_semantics = !class_ids.empty();
    const bool has_instance_ids = !instance_ids.empty();
    if (has_semantics) {
        if (points.size() != class_ids.size()) {
            throw std::runtime_error("points and class_ids must have the same size");
        }
        if (has_instance_ids && points.size() != instance_ids.size()) {
            throw std::runtime_error("points and instance_ids must have the same size");
        }
    } else {
        if (has_instance_ids) {
            throw std::runtime_error("instance_ids but no class_ids is not supported");
        }
    }

    if (has_colors) {
        // we have colors
        if (has_semantics) {
            // we have semantics
            if (has_instance_ids) {
                // we have instance ids
                if (has_depths) {
                    integrate_raw(points.data(), points.size(), colors.data(), class_ids.data(),
                                  instance_ids.data(), depths.data());
                } else {
                    integrate_raw(points.data(), points.size(), colors.data(), class_ids.data(),
                                  instance_ids.data());
                }
            } else {
                // we do not have instance ids
                if (has_depths) {
                    integrate_raw(points.data(), points.size(), colors.data(), class_ids.data(),
                                  nullptr, depths.data());
                } else {
                    integrate_raw(points.data(), points.size(), colors.data(), class_ids.data());
                }
            }
        } else {
            // with colors, no semantics
            if (has_depths) {
                integrate_raw(points.data(), points.size(), colors.data(), nullptr, nullptr,
                              depths.data());
            } else {
                integrate_raw(points.data(), points.size(), colors.data());
            }
        }
    } else {
        // No colors
        if (has_semantics) {
            // we have semantics
            if (has_instance_ids) {
                // we have instance ids
                if (has_depths) {
                    integrate_raw(points.data(), points.size(), nullptr, class_ids.data(),
                                  instance_ids.data(), depths.data());
                } else {
                    integrate_raw(points.data(), points.size(), nullptr, class_ids.data(),
                                  instance_ids.data());
                }
            } else {
                // we do not have instance ids
                if (has_depths) {
                    integrate_raw(points.data(), points.size(), nullptr, class_ids.data(), nullptr,
                                  depths.data());
                } else {
                    integrate_raw(points.data(), points.size(), nullptr, class_ids.data());
                }
            }
        } else {
            // no colors, no semantics
            if (has_depths) {
                integrate_raw(points.data(), points.size(), nullptr, nullptr, nullptr,
                              depths.data());
            } else {
                integrate_raw(points.data(), points.size());
            }
        }
    }
}

// Internal method that does the actual integration work (can be called without GIL)
template <typename VoxelDataT>
template <typename Tp, typename Tc, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelGridT<VoxelDataT>::integrate_raw(const Tp *pts_ptr, size_t num_points, const Tc *cols_ptr,
                                           const Tclass *class_ids_ptr,
                                           const Tinstance *instance_ids_ptr,
                                           const Tdepth *depths_ptr) {
    // Route to semantic-aware implementation if semantic parameters are provided
    constexpr bool has_semantics = !std::is_same_v<Tinstance, std::nullptr_t> ||
                                   !std::is_same_v<Tclass, std::nullptr_t> ||
                                   !std::is_same_v<Tdepth, std::nullptr_t>;

    if constexpr (has_semantics) {
        // Use pre-partitioning approach for semantic-aware integration
        integrate_raw_semantic_impl<Tp, Tc, Tinstance, Tclass, Tdepth>(
            pts_ptr, num_points, cols_ptr, class_ids_ptr, instance_ids_ptr, depths_ptr);
    } else {
        // Use SIMD for float types when available (non-semantic path)
        if constexpr (std::is_same_v<Tp, float> && std::is_same_v<Tc, float> && USE_SIMD) {
            if constexpr (std::is_same_v<Tc, std::nullptr_t>) {
                integrate_raw_simd(pts_ptr, num_points);
            } else {
                integrate_raw_simd(pts_ptr, cols_ptr, num_points);
            }
        } else if constexpr (std::is_same_v<Tp, double> && std::is_same_v<Tc, float> &&
                             USE_DOUBLE_SIMD) {
            if constexpr (std::is_same_v<Tc, std::nullptr_t>) {
                integrate_raw_simd(pts_ptr, num_points);
            } else {
                integrate_raw_simd(pts_ptr, cols_ptr, num_points);
            }
        } else {
            if constexpr (std::is_same_v<Tc, std::nullptr_t>) {
                integrate_raw_scalar(pts_ptr, num_points);
            } else {
                integrate_raw_scalar(pts_ptr, cols_ptr, num_points);
            }
        }
    }
}

// Semantic-aware integration implementation using optimized pre-partitioning
template <typename VoxelDataT>
template <typename Tp, typename Tc, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelGridT<VoxelDataT>::integrate_raw_semantic_impl(const Tp *pts_ptr, size_t num_points,
                                                         const Tc *cols_ptr,
                                                         const Tclass *class_ids_ptr,
                                                         const Tinstance *instance_ids_ptr,
                                                         const Tdepth *depths_ptr) {
#ifdef TBB_FOUND
    // Use enumerable_thread_specific to collect thread-local maps without contention
    // Each thread builds its own map independently, then we merge sequentially after parallel phase
    // Helper types to handle nullptr_t template parameters
    using ColorScalar = std::conditional_t<std::is_same_v<Tc, std::nullptr_t>, float, Tc>;
    using InstanceType =
        std::conditional_t<std::is_same_v<Tinstance, std::nullptr_t>, int, Tinstance>;
    using ClassType = std::conditional_t<std::is_same_v<Tclass, std::nullptr_t>, int, Tclass>;
    using DepthType = std::conditional_t<std::is_same_v<Tdepth, std::nullptr_t>, float, Tdepth>;

    struct PointInfo {
        Tp x, y, z;
        ColorScalar color_x, color_y, color_z;
        InstanceType object_id;
        ClassType class_id;
        DepthType depth;
    };

    using LocalGroupsMap = std::unordered_map<VoxelKey, std::vector<PointInfo>, VoxelKeyHash>;
    tbb::enumerable_thread_specific<LocalGroupsMap> thread_local_groups([]() {
        LocalGroupsMap map;
        map.reserve(64); // Pre-allocate for typical voxel count
        return map;
    });

    // Precompute keys and group points by voxel in parallel
    // Each thread builds its own local map without any synchronization
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points), [&](auto r) {
        // Get thread-local map (created lazily if needed)
        auto &local_groups = thread_local_groups.local();

        for (size_t i = r.begin(); i < r.end(); ++i) {
            const size_t idx = i * 3;
            const Tp x = pts_ptr[idx + 0];
            const Tp y = pts_ptr[idx + 1];
            const Tp z = pts_ptr[idx + 2];

            const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

            PointInfo info{x, y, z};
            if constexpr (!std::is_same_v<Tc, std::nullptr_t>) {
                info.color_x = cols_ptr[idx + 0];
                info.color_y = cols_ptr[idx + 1];
                info.color_z = cols_ptr[idx + 2];
            }
            if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                info.object_id = instance_ids_ptr[i];
            }
            if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                info.class_id = class_ids_ptr[i];
            }
            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                info.depth = depths_ptr[i];
            }

            local_groups[key].push_back(info);
        }
    });

    // Merge thread-local maps into final result sequentially (no contention)
    // Use std::unordered_map since merge is sequential and subsequent parallel processing is
    // read-only
    std::unordered_map<VoxelKey, std::vector<PointInfo>, VoxelKeyHash> voxel_groups;

    // First pass: count total points per key to reserve appropriate vector sizes
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> key_point_counts;
    for (const auto &local_groups : thread_local_groups) {
        for (const auto &[key, points] : local_groups) {
            key_point_counts[key] += points.size();
        }
    }

    // Reserve map capacity based on unique keys (avoids rehashing during insert)
    voxel_groups.reserve(key_point_counts.size());

    // Second pass: merge with pre-reserved vectors to avoid reallocation
    for (auto &local_groups : thread_local_groups) {
        for (auto &[key, points] : local_groups) {
            auto it = voxel_groups.find(key);
            if (it == voxel_groups.end()) {
                // Insert a new entry with reserved capacity based on total points for this key
                std::vector<PointInfo> vec;
                vec.reserve(key_point_counts[key]);
                vec.insert(vec.end(), points.begin(), points.end());
                voxel_groups.emplace(key, std::move(vec));
            } else {
                // Append points to the existing vector (sequential merge, no synchronization
                // needed)
                // Ensure vector has enough capacity to avoid reallocation during append
                auto &global_vec = it->second;
                const size_t total_capacity_needed = key_point_counts[key];
                if (global_vec.capacity() < total_capacity_needed) {
                    global_vec.reserve(total_capacity_needed);
                }
                global_vec.insert(global_vec.end(), points.begin(), points.end());
            }
        }
    }

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
                if constexpr (!std::is_same_v<Tc, std::nullptr_t>) {
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
                                    v.initialize_semantics_with_depth(info.object_id, info.class_id,
                                                                      info.depth);
                                } else {
                                    v.initialize_semantics(info.object_id, info.class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.initialize_semantics(info.object_id, info.class_id);
                            }
                        } else {
                            // we do not have instance id => use default instance id 0
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.initialize_semantics_with_depth(0, info.class_id, info.depth);
                                } else {
                                    v.initialize_semantics(0, info.class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.initialize_semantics(0, info.class_id);
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
                if constexpr (!std::is_same_v<Tc, std::nullptr_t>) {
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
                                    v.update_semantics_with_depth(info.object_id, info.class_id,
                                                                  info.depth);
                                } else {
                                    v.update_semantics(info.object_id, info.class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.update_semantics(info.object_id, info.class_id);
                            }
                        } else {
                            // we do not have instance id => use default instance id 0
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                    v.update_semantics_with_depth(0, info.class_id, info.depth);
                                } else {
                                    v.update_semantics(0, info.class_id);
                                }
                            } else {
                                // we do not have depth => use confidence = 1.0
                                v.update_semantics(0, info.class_id);
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

        const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

        // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
        auto [it, inserted] = grid_.try_emplace(key);
        auto &v = it->second;

        if (inserted || v.count == 0) {
            // New voxel or reset voxel: initialize and update
            // If count==0, the voxel was reset and should be treated as new
            v.update_point(x, y, z);
            if constexpr (!std::is_same_v<Tc, std::nullptr_t>) {
                const Tc color_x = cols_ptr[idx + 0];
                const Tc color_y = cols_ptr[idx + 1];
                const Tc color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            if constexpr (SemanticVoxel<VoxelDataT>) {
                if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                    // we have semantics/class id
                    if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                        // we have instance id
                        const Tinstance object_id = instance_ids_ptr[i];
                        const Tclass class_id = class_ids_ptr[i];
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                v.initialize_semantics_with_depth(object_id, class_id,
                                                                  depths_ptr[i]);
                            } else {
                                v.initialize_semantics(object_id, class_id);
                            }
                        } else {
                            // we do not have depth => use confidence = 1.0
                            v.initialize_semantics(object_id, class_id);
                        }
                    } else {
                        // we do not have instance id => use default instance id 0
                        const Tclass class_id = class_ids_ptr[i];
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                v.initialize_semantics_with_depth(0, class_id, depths_ptr[i]);
                            } else {
                                v.initialize_semantics(0, class_id);
                            }
                        } else {
                            // we do not have depth => use confidence = 1.0
                            v.initialize_semantics(0, class_id);
                        }
                    }
                }
            }
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            if constexpr (!std::is_same_v<Tc, std::nullptr_t>) {
                const Tc color_x = cols_ptr[idx + 0];
                const Tc color_y = cols_ptr[idx + 1];
                const Tc color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            if constexpr (SemanticVoxel<VoxelDataT>) {
                if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                    // we have semantics/class id
                    if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                        // we have instance id
                        const Tinstance object_id = instance_ids_ptr[i];
                        const Tclass class_id = class_ids_ptr[i];
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                v.update_semantics_with_depth(object_id, class_id, depths_ptr[i]);
                            } else {
                                v.update_semantics(object_id, class_id);
                            }
                        } else {
                            // we do not have depth => use confidence = 1.0
                            v.update_semantics(object_id, class_id);
                        }
                    } else {
                        // we do not have instance id => use default instance id 0
                        const Tclass class_id = class_ids_ptr[i];
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                                v.update_semantics_with_depth(0, class_id, depths_ptr[i]);
                            } else {
                                v.update_semantics(0, class_id);
                            }
                        } else {
                            // we do not have depth => use confidence = 1.0
                            v.update_semantics(0, class_id);
                        }
                    }
                }
            }
            ++v.count;
        }
    }
#endif
}

template <typename VoxelDataT>
template <typename Tp, typename Tc>
void VoxelGridT<VoxelDataT>::integrate_raw_scalar(const Tp *pts_ptr, const Tc *cols_ptr,
                                                  const size_t num_points) {
    // Use pre-partitioning approach by default (thread-safe, efficient)
    integrate_raw_scalar_impl_with_prepartitioning<Tp, Tc, std::is_same_v<Tc, float>>(
        pts_ptr, cols_ptr, num_points);
}

template <typename VoxelDataT>
template <typename Tp>
void VoxelGridT<VoxelDataT>::integrate_raw_scalar(const Tp *pts_ptr, const size_t num_points) {
    // Use pre-partitioning approach by default (thread-safe, efficient)
    integrate_raw_scalar_impl_with_prepartitioning<Tp, float, false>(
        pts_ptr, nullptr, num_points); // fake float type for colors
}

// Optimized scalar version using pre-partitioning (default implementation)
// Groups points by voxel key first, then processes each voxel group serially in parallel.
// This avoids race conditions without needing mutexes: each voxel is updated by exactly
// one thread, eliminating concurrent access to the same VoxelDataT object.
template <typename VoxelDataT>
template <typename Tp, typename Tc, bool HasColors>
void VoxelGridT<VoxelDataT>::integrate_raw_scalar_impl_with_prepartitioning(const Tp *pts_ptr,
                                                                            const Tc *cols_ptr,
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
        tbb::blocked_range<size_t>(0, num_points), [&](const tbb::blocked_range<size_t> &range) {
            // Thread-local map for this range
            std::unordered_map<VoxelKey, std::vector<PointInfo>, VoxelKeyHash> local_groups;
            local_groups.reserve(64); // Pre-allocate for typical voxel count

            for (size_t i = range.begin(); i < range.end(); ++i) {
                const size_t idx = i * 3;
                const Tp x = pts_ptr[idx + 0];
                const Tp y = pts_ptr[idx + 1];
                const Tp z = pts_ptr[idx + 2];

                const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

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

        const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

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
template <typename VoxelDataT>
template <typename Tp, typename Tc, bool HasColors>
void VoxelGridT<VoxelDataT>::integrate_raw_scalar_impl_with_mutexes(const Tp *pts_ptr,
                                                                    const Tc *cols_ptr,
                                                                    size_t num_points) {
#ifdef TBB_FOUND
    // Hash map of mutexes (one per voxel key)
    tbb::concurrent_unordered_map<VoxelKey, std::unique_ptr<std::mutex>, VoxelKeyHash>
        voxel_mutexes;
    std::mutex mutex_map_mutex; // For creating new mutexes

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_points), [&](const tbb::blocked_range<size_t> &range) {
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

                const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

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

        const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

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

// Wrapper functions that call the unified template implementation
template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::integrate_raw_simd(const float *pts_ptr, const float *cols_ptr,
                                                const size_t num_points) {
    integrate_raw_simd_impl<true>(pts_ptr, cols_ptr, num_points);
}

template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::integrate_raw_simd(const float *pts_ptr, const size_t num_points) {
    integrate_raw_simd_impl<false>(pts_ptr, nullptr, num_points);
}

#endif

#if USE_DOUBLE_SIMD

// Wrapper functions for double precision points
template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::integrate_raw_simd(const double *pts_ptr, const float *cols_ptr,
                                                const size_t num_points) {
    integrate_raw_simd_impl_double<true>(pts_ptr, cols_ptr, num_points);
}

template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::integrate_raw_simd(const double *pts_ptr, const size_t num_points) {
    integrate_raw_simd_impl_double<false>(pts_ptr, nullptr, num_points);
}

#endif

template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::carve(const CameraFrustrum &camera_frustrum,
                                   const cv::Mat &depth_image, const float depth_threshold) {

    using ThisType = VoxelGridT<VoxelDataT>;
    volumetric::carve<ThisType, VoxelDataT>(*this, camera_frustrum, depth_image, depth_threshold);
};

// remove_low_count_voxels implementation
template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::remove_low_count_voxels(const int min_count) {
#ifdef TBB_FOUND
    // Parallel version
    tbb::parallel_for_each(grid_.begin(), grid_.end(), [&](auto &pair) {
        auto &v = pair.second;
        if (v.count < min_count) {
            v.reset();
        }
    });
#else
    // Sequential version
    for (auto &[key, v] : grid_) {
        if (v.count < min_count) {
            v.reset();
        }
    }
#endif
}

// remove_low_confidence_voxels implementation
template <typename VoxelDataT>
void VoxelGridT<VoxelDataT>::remove_low_confidence_voxels(const float min_confidence) {
    if constexpr (SemanticVoxel<VoxelDataT>) {
#ifdef TBB_FOUND
        // Parallel version
        tbb::parallel_for_each(grid_.begin(), grid_.end(), [&](auto &pair) {
            auto &v = pair.second;
            if (v.get_confidence() < min_confidence) {
                v.reset();
            }
        });
#else
        // Sequential version
        for (auto &[key, v] : grid_) {
            if (v.get_confidence() < min_confidence) {
                v.reset();
            }
        }
#endif
    }
}

template <typename VoxelDataT>
std::vector<typename VoxelGridT<VoxelDataT>::Pos3> VoxelGridT<VoxelDataT>::get_points() const {
    std::vector<typename VoxelGridT<VoxelDataT>::Pos3> points;
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

template <typename VoxelDataT>
std::vector<typename VoxelGridT<VoxelDataT>::Color3> VoxelGridT<VoxelDataT>::get_colors() const {
    std::vector<typename VoxelGridT<VoxelDataT>::Color3> colors;
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

template <typename VoxelDataT>
typename VoxelGridT<VoxelDataT>::VoxelGridDataType
VoxelGridT<VoxelDataT>::get_voxels(int min_count, float min_confidence) const {
    typename VoxelGridT<VoxelDataT>::VoxelGridDataType result;
#ifdef TBB_FOUND
    // Parallel version: collect keys first, filter, then process in parallel
    // Use isolate() to prevent deadlock from nested parallelism
    std::vector<VoxelKey> keys;
    keys.reserve(grid_.size());
    bool check_point;
    for (const auto &[key, v] : grid_) {
        if constexpr (SemanticVoxel<VoxelDataT>) {
            check_point = v.count >= min_count && v.get_confidence() >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (check_point) {
            keys.push_back(key);
        }
    }
    // Resize vectors to exact size to enable thread-safe indexed writes
    result.points.resize(keys.size());
    result.colors.resize(keys.size());
    if constexpr (SemanticVoxel<VoxelDataT>) {
        result.object_ids.resize(keys.size());
        result.class_ids.resize(keys.size());
        result.confidences.resize(keys.size());
    }
    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                          [&](const tbb::blocked_range<size_t> &range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  const auto &v = grid_.at(keys[i]);
                                  result.points[i] = v.get_position();
                                  result.colors[i] = v.get_color();
                                  if constexpr (SemanticVoxel<VoxelDataT>) {
                                      result.object_ids[i] = v.get_object_id();
                                      result.class_ids[i] = v.get_class_id();
                                      result.confidences[i] = v.get_confidence();
                                  }
                              }
                          });
    });
#else
    // Sequential version
    result.points.reserve(grid_.size());
    result.colors.reserve(grid_.size());
    if constexpr (SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(grid_.size());
        result.class_ids.reserve(grid_.size());
        result.confidences.reserve(grid_.size());
    }

    // Always filter by min_count to exclude reset voxels (count=0) unless explicitly requested
    bool check_point;
    for (const auto &[key, v] : grid_) {
        if constexpr (SemanticVoxel<VoxelDataT>) {
            check_point = v.count >= min_count && v.get_confidence() >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (check_point) {
            result.points.push_back(v.get_position());
            result.colors.push_back(v.get_color());
            if constexpr (SemanticVoxel<VoxelDataT>) {
                result.object_ids.push_back(v.get_object_id());
                result.class_ids.push_back(v.get_class_id());
                result.confidences.push_back(v.get_confidence());
            }
        }
    }
#endif
    return result;
}

// Get voxels within a spatial interval (bounding box)
// Returns points and colors for voxels whose centers fall within [min_xyz, max_xyz]
// If IncludeSemantics is true and VoxelDataT is a SemanticVoxel, also returns semantic data
template <typename VoxelDataT>
template <bool IncludeSemantics>
typename VoxelGridT<VoxelDataT>::VoxelGridDataType
VoxelGridT<VoxelDataT>::get_voxels_in_bb(const BoundingBox3D &bbox, const int min_count,
                                         float min_confidence) const {
    // Convert bounding box to voxel key bounds for fast filtering
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    typename VoxelGridT<VoxelDataT>::VoxelGridDataType result;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing voxels, filter by key range, then check
    // position Collect matching keys first, then process in parallel
    std::vector<VoxelKey> keys;
    keys.reserve(grid_.size() / 4); // Heuristic: assume ~25% of voxels might be in range
    bool check_point;
    float confidence;

    // First pass: iterate existing voxels and filter by key range (fast integer comparison)
    // This avoids O(volume) hash lookups and skips most voxels outside the bounding box
    for (const auto &[key, v] : grid_) {
        // Fast key-range check (6 integer comparisons) - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue; // Skip voxels outside key range
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained position check (more expensive, but only for candidates)
        auto pos = v.get_position();
        if (bbox.contains(pos[0], pos[1], pos[2])) {
            keys.push_back(key);
        }
    }

    // Resize vectors to exact size to enable thread-safe indexed writes
    result.points.resize(keys.size());
    result.colors.resize(keys.size());
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.resize(keys.size());
        result.class_ids.resize(keys.size());
        result.confidences.resize(keys.size());
    }

    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                          [&](const tbb::blocked_range<size_t> &range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  const auto &v = grid_.at(keys[i]);
                                  result.points[i] = v.get_position();
                                  result.colors[i] = v.get_color();
                                  if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                      result.object_ids[i] = v.get_object_id();
                                      result.class_ids[i] = v.get_class_id();
                                      result.confidences[i] = v.get_confidence();
                                  }
                              }
                          });
    });
#else
    // Sequential version: iterate existing voxels with key-range filtering
    result.points.reserve(grid_.size() / 4);
    result.colors.reserve(grid_.size() / 4);
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(grid_.size() / 4);
        result.class_ids.reserve(grid_.size() / 4);
        result.confidences.reserve(grid_.size() / 4);
    }

    bool check_point;
    float confidence;
    for (const auto &[key, v] : grid_) {
        // Fast key-range check - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue;
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained position check
        auto pos = v.get_position();
        if (bbox.contains(pos[0], pos[1], pos[2])) {
            result.points.push_back(pos);
            result.colors.push_back(v.get_color());
            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                result.object_ids.push_back(v.get_object_id());
                result.class_ids.push_back(v.get_class_id());
                result.confidences.push_back(confidence);
            }
        }
    }
#endif

    return result;
}

// Get voxels within a camera frustrum
// Returns points and colors for voxels whose centers fall within the camera frustrum
// If IncludeSemantics is true and VoxelDataT is a SemanticVoxel, also returns semantic data
template <typename VoxelDataT>
template <bool IncludeSemantics>
typename VoxelGridT<VoxelDataT>::VoxelGridDataType
VoxelGridT<VoxelDataT>::get_voxels_in_camera_frustrum(const CameraFrustrum &camera_frustrum,
                                                      const int min_count,
                                                      float min_confidence) const {

    const BoundingBox3D bbox = camera_frustrum.get_bbox();

    // Convert bounding box to voxel key bounds for fast filtering
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    typename VoxelGridT<VoxelDataT>::VoxelGridDataType result;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing voxels, filter by key range, then check
    // frustum Collect matching keys first, then process in parallel
    std::vector<VoxelKey> keys;
    keys.reserve(grid_.size() / 4); // Heuristic: assume ~25% of voxels might be in range
    bool check_point;
    float confidence;

    // First pass: iterate existing voxels and filter by key range (fast integer comparison)
    // This avoids O(volume) hash lookups and skips most voxels outside the bounding box
    for (const auto &[key, v] : grid_) {
        // Fast key-range check (6 integer comparisons) - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue; // Skip voxels outside key range
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained frustum check (more expensive, but only for candidates)
        const auto pos = v.get_position(); // world coordinates
        const auto [is_in_frustum, image_point] = camera_frustrum.contains(pos[0], pos[1], pos[2]);
        if (is_in_frustum) {
            keys.push_back(key);
        }
    }

    // Resize vectors to exact size to enable thread-safe indexed writes
    result.points.resize(keys.size());
    result.colors.resize(keys.size());
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.resize(keys.size());
        result.class_ids.resize(keys.size());
        result.confidences.resize(keys.size());
    }

    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                          [&](const tbb::blocked_range<size_t> &range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  const auto &v = grid_.at(keys[i]);
                                  result.points[i] = v.get_position();
                                  result.colors[i] = v.get_color();
                                  if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                      result.object_ids[i] = v.get_object_id();
                                      result.class_ids[i] = v.get_class_id();
                                      result.confidences[i] = v.get_confidence();
                                  }
                              }
                          });
    });
#else
    // Sequential version: iterate existing voxels with key-range filtering
    result.points.reserve(grid_.size() / 4);
    result.colors.reserve(grid_.size() / 4);
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(grid_.size() / 4);
        result.class_ids.reserve(grid_.size() / 4);
        result.confidences.reserve(grid_.size() / 4);
    }

    bool check_point;
    float confidence;
    for (const auto &[key, v] : grid_) {
        // Fast key-range check - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue;
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained frustum check
        const auto pos = v.get_position(); // world coordinates
        const auto [is_in_frustum, image_point] = camera_frustrum.contains(pos[0], pos[1], pos[2]);
        if (is_in_frustum) {
            result.points.push_back(pos);
            result.colors.push_back(v.get_color());
            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                result.object_ids.push_back(v.get_object_id());
                result.class_ids.push_back(v.get_class_id());
                result.confidences.push_back(confidence);
            }
        }
    }
#endif

    return result;
}

// Iterate over voxels in a spatial interval with a callback function
// The callback receives (voxel_key, voxel_data) for each voxel in the interval
template <typename VoxelDataT>
template <typename Callback>
void VoxelGridT<VoxelDataT>::iterate_voxels_in_bb(const BoundingBox3D &bbox, Callback &&callback,
                                                  int min_count, float min_confidence) const {
    // Convert spatial bounds to voxel key bounds
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    bool check_point;
    float confidence;
    // Optimized: iterate existing voxels and filter by key range (fast integer comparison)
    // This avoids O(volume) hash lookups and skips most voxels outside the bounding box
    for (const auto &[key, v] : grid_) {
        // Fast key-range check - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue;
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained position check
        const auto pos = v.get_position();
        if (bbox.contains(pos[0], pos[1], pos[2])) {
            using Pos3d = std::array<double, 3>;
            using Pos3f = std::array<float, 3>;
            using Pos3Type = typename VoxelGridT<VoxelDataT>::Pos3;
            if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const Pos3Type &,
                                              const VoxelKey &>) {
                callback(v, pos, key);
            } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3Type &>) {
                callback(v, key, pos);
            } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3d &>) {
                const Pos3d pos_d = {
                    static_cast<double>(pos[0]),
                    static_cast<double>(pos[1]),
                    static_cast<double>(pos[2]),
                };
                callback(v, key, pos_d);
            } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3f &>) {
                const Pos3f pos_f = {
                    static_cast<float>(pos[0]),
                    static_cast<float>(pos[1]),
                    static_cast<float>(pos[2]),
                };
                callback(v, key, pos_f);
            } else {
                struct UnsupportedCallbackSignature : std::false_type {};
                static_assert(UnsupportedCallbackSignature::value,
                              "Callback must accept one of: "
                              "(VoxelDataT&, Pos3, VoxelKey), "
                              "(VoxelDataT&, VoxelKey, Pos3), "
                              "(VoxelDataT&, VoxelKey, std::array<double,3>), "
                              "(VoxelDataT&, VoxelKey, std::array<float,3>).");
            }
        }
    }
}

// Iterate over voxels in camera frustrum
template <typename VoxelDataT>
template <typename Callback>
void VoxelGridT<VoxelDataT>::iterate_voxels_in_camera_frustrum(
    const CameraFrustrum &camera_frustrum, Callback &&callback, int min_count,
    float min_confidence) {
    const BoundingBox3D bbox = camera_frustrum.get_bbox();
    // Convert bounding box to voxel key bounds for fast filtering
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    bool check_point;
    float confidence;
    // Optimized: iterate existing voxels and filter by key range (fast integer comparison)
    // This avoids O(volume) hash lookups and skips most voxels outside the bounding box
    for (auto &[key, v] : grid_) {
        // Fast key-range check - rejects most voxels immediately
        if (key.x < min_key.x || key.x > max_key.x || key.y < min_key.y || key.y > max_key.y ||
            key.z < min_key.z || key.z > max_key.z) {
            continue;
        }

        // Check count/confidence criteria
        if constexpr (SemanticVoxel<VoxelDataT>) {
            confidence = v.get_confidence();
            check_point = v.count >= min_count && confidence >= min_confidence;
        } else {
            check_point = v.count >= min_count;
        }
        if (!check_point) {
            continue;
        }

        // Fine-grained position check with camera frustrum
        const auto pos = v.get_position();
        const auto [is_in_frustum, image_point] = camera_frustrum.contains(pos[0], pos[1], pos[2]);
        if (is_in_frustum) {
            using Pos3d = std::array<double, 3>;
            using Pos3f = std::array<float, 3>;
            using Pos3Type = typename VoxelGridT<VoxelDataT>::Pos3;
            if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                              const Pos3Type &, const ImagePoint &>) {
                callback(v, key, pos, image_point);
            } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3d &, const ImagePoint &>) {
                const Pos3d pos_d = {
                    static_cast<double>(pos[0]),
                    static_cast<double>(pos[1]),
                    static_cast<double>(pos[2]),
                };
                callback(v, key, pos_d, image_point);
            } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3f &, const ImagePoint &>) {
                const Pos3f pos_f = {
                    static_cast<float>(pos[0]),
                    static_cast<float>(pos[1]),
                    static_cast<float>(pos[2]),
                };
                callback(v, key, pos_f, image_point);
            } else {
                struct UnsupportedCallbackSignature : std::false_type {};
                static_assert(UnsupportedCallbackSignature::value,
                              "Callback must accept one of: "
                              "(VoxelDataT&, VoxelKey, Pos3, ImagePoint), "
                              "(VoxelDataT&, VoxelKey, std::array<double,3>, ImagePoint), "
                              "(VoxelDataT&, VoxelKey, std::array<float,3>, ImagePoint).");
            }
        }
    }
}

// Clear the voxel grid
template <typename VoxelDataT> void VoxelGridT<VoxelDataT>::clear() { grid_.clear(); }

// Get the size of the voxel grid
template <typename VoxelDataT> size_t VoxelGridT<VoxelDataT>::size() const { return grid_.size(); }

// Check if the voxel grid is empty
template <typename VoxelDataT> bool VoxelGridT<VoxelDataT>::empty() const { return grid_.empty(); }

} // namespace volumetric

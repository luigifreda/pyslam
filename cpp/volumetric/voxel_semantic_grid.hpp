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

namespace volumetric {

// Constructor implementation
template <typename VoxelDataT>
VoxelSemanticGridT<VoxelDataT>::VoxelSemanticGridT(double voxel_size)
    : voxel_size_(voxel_size), inv_voxel_size_(1.0 / voxel_size) {
    static_assert(SemanticVoxel<VoxelDataT>, "VoxelDataT must satisfy the SemanticVoxel concept");
}

// set_depth_threshold implementation
template <typename VoxelDataT>
void VoxelSemanticGridT<VoxelDataT>::set_depth_threshold(float depth_threshold) {
    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
        VoxelDataT::kDepthThreshold = depth_threshold;
    }
}

// set_depth_decay_rate implementation
template <typename VoxelDataT>
void VoxelSemanticGridT<VoxelDataT>::set_depth_decay_rate(float depth_decay_rate) {
    if constexpr (std::is_same_v<VoxelDataT, VoxelSemanticDataProbabilistic>) {
        VoxelDataT::kDepthDecayRate = depth_decay_rate;
    }
}

// integrate_segment_raw implementation
template <typename VoxelDataT>
template <typename Tp, typename Tc>
void VoxelSemanticGridT<VoxelDataT>::integrate_segment_raw(const Tp *pts_ptr,
                                                           const size_t num_points,
                                                           const Tc *cols_ptr, const int class_id,
                                                           const int object_id) {
    // Skip points with invalid instance or class IDs
    if (object_id < 0 || class_id < 0) {
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
                v.initialize_semantics(object_id, class_id);
                v.count = 1;
                inserted = false; // Only first point is "inserted"
                is_first = false;
            } else {
                // Existing voxel: accumulate
                v.update_point(info.x, info.y, info.z);
                v.update_color(info.color_x, info.color_y, info.color_z);
                v.update_semantics(object_id, class_id);
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

        const VoxelKey key = get_voxel_key_inv<Tp, Tp>(x, y, z, inv_voxel_size_);

        // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
        auto [it, inserted] = grid_.try_emplace(key);
        auto &v = it->second;

        if (inserted || v.count == 0) {
            // New voxel or reset voxel: initialize and update
            // If count==0, the voxel was reset and should be treated as new
            v.update_point(x, y, z);
            v.update_color(color_x, color_y, color_z);
            v.initialize_semantics(object_id, class_id);
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            v.update_color(color_x, color_y, color_z);
            v.update_semantics(object_id, class_id);
            ++v.count;
        }
    }
#endif
}

// merge_segments implementation
template <typename VoxelDataT>
void VoxelSemanticGridT<VoxelDataT>::merge_segments(const int instance_id1,
                                                    const int instance_id2) {
#ifdef TBB_FOUND
    // Parallel version using TBB - concurrent_unordered_map is thread-safe
    // Use isolate() to prevent deadlock from nested parallelism
    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for_each(grid_.begin(), grid_.end(), [&](auto &pair) {
            if (pair.second.get_object_id() == instance_id2) {
                // concurrent_unordered_map allows safe concurrent modification
                pair.second.set_object_id(instance_id1);
            }
        });
    });
#else
    // Sequential version
    for (auto &[key, v] : grid_) {
        if (v.get_object_id() == instance_id2) {
            v.set_object_id(instance_id1);
        }
    }
#endif
}

// remove_segment implementation
template <typename VoxelDataT>
void VoxelSemanticGridT<VoxelDataT>::remove_segment(const int object_id) {
#ifdef TBB_FOUND
    // For concurrent_unordered_map, collect keys first, then erase
    std::vector<VoxelKey> keys_to_remove;
    keys_to_remove.reserve(grid_.size());
    for (const auto &[key, v] : grid_) {
        if (v.get_object_id() == object_id) {
            keys_to_remove.push_back(key);
        }
    }
    for (const auto &key : keys_to_remove) {
        grid_.unsafe_erase(key);
    }
#else
    // Sequential version for std::unordered_map
    for (auto it = grid_.begin(); it != grid_.end();) {
        if (it->second.get_object_id() == object_id) {
            it = grid_.erase(it);
        } else {
            ++it;
        }
    }
#endif
}

// remove_low_confidence_segments implementation
template <typename VoxelDataT>
void VoxelSemanticGridT<VoxelDataT>::remove_low_confidence_segments(const int min_confidence) {
#ifdef TBB_FOUND
    // For concurrent_unordered_map, collect keys first, then erase
    std::vector<VoxelKey> keys_to_remove;
    keys_to_remove.reserve(grid_.size());
    for (const auto &[key, v] : grid_) {
        // Use getter method if available (for probabilistic), otherwise direct member access
        if (v.get_confidence() < min_confidence) {
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
        if (it->second.get_confidence() < min_confidence) {
            // Remove the voxel with low confidence_counter
            it = grid_.erase(it);
        } else {
            ++it;
        }
    }
#endif
}

// get_points implementation
template <typename VoxelDataT>
std::vector<std::array<double, 3>> VoxelSemanticGridT<VoxelDataT>::get_points() const {
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

// get_colors implementation
template <typename VoxelDataT>
std::vector<std::array<float, 3>> VoxelSemanticGridT<VoxelDataT>::get_colors() const {
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

// get_ids implementation
template <typename VoxelDataT>
std::pair<std::vector<int>, std::vector<int>> VoxelSemanticGridT<VoxelDataT>::get_ids() const {
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
                                  instance_ids[i] = v.get_object_id();
                              }
                          });
    });
#else
    // Sequential version
    for (const auto &[key, v] : grid_) {
        class_ids.push_back(v.get_class_id());
        instance_ids.push_back(v.get_object_id());
    }
#endif
    return {class_ids, instance_ids};
}

// get_instance_segments implementation
template <typename VoxelDataT>
std::unordered_map<int, std::vector<std::array<double, 3>>>
VoxelSemanticGridT<VoxelDataT>::get_instance_segments() const {
    std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
    for (const auto &[key, v] : grid_) {
        segments[v.get_object_id()].push_back(v.get_position());
    }
    return segments;
}

// get_class_segments implementation
template <typename VoxelDataT>
std::unordered_map<int, std::vector<std::array<double, 3>>>
VoxelSemanticGridT<VoxelDataT>::get_class_segments() const {
    std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
    for (const auto &[key, v] : grid_) {
        segments[v.get_class_id()].push_back(v.get_position());
    }
    return segments;
}

// clear implementation
template <typename VoxelDataT> void VoxelSemanticGridT<VoxelDataT>::clear() { grid_.clear(); }

// size implementation
template <typename VoxelDataT> size_t VoxelSemanticGridT<VoxelDataT>::size() const {
    return grid_.size();
}

// empty implementation
template <typename VoxelDataT> bool VoxelSemanticGridT<VoxelDataT>::empty() const {
    return grid_.empty();
}

} // namespace volumetric

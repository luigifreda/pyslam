namespace volumetric {

// Constructor implementation
template <typename VoxelDataT>
VoxelBlockGridT<VoxelDataT>::VoxelBlockGridT(float voxel_size, int block_size)
    : voxel_size_(voxel_size), inv_voxel_size_(1.0f / voxel_size), block_size_(block_size) {
    static_assert(Voxel<VoxelDataT>, "VoxelDataT must satisfy the Voxel concept");
    num_voxels_per_block_ = block_size_ * block_size_ * block_size_;
}

// integrate with std::vector inputs
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::integrate(const std::vector<Tpos> &points,
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

// integrate_raw implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::integrate_raw(const Tpos *pts_ptr, size_t num_points,
                                                const Tcolor *cols_ptr, const Tclass *class_ids_ptr,
                                                const Tinstance *instance_ids_ptr,
                                                const Tdepth *depths_ptr) {
    if (num_points == 0) {
        return;
    }
    // Here we actually select the implementation of the integration function.
#if 1
    // Implementation with
    // 1) Preliminary parallelized block-based grouping to minimize mutex contention
    // 2) Per-block parallelized integration without block-level mutex protection
    integrate_raw_preorder_no_block_mutex<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
        pts_ptr, num_points, cols_ptr, class_ids_ptr, instance_ids_ptr, depths_ptr);
#else
    // Implementation with parallelized per-point integration with block-level mutex protection
    integrate_raw_baseline<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
        pts_ptr, num_points, cols_ptr, class_ids_ptr, instance_ids_ptr, depths_ptr);
#endif
}

// integrate_raw_baseline implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::integrate_raw_baseline(const Tpos *pts_ptr, size_t num_points,
                                                         const Tcolor *cols_ptr,
                                                         const Tclass *class_ids_ptr,
                                                         const Tinstance *instance_ids_ptr,
                                                         const Tdepth *depths_ptr) {
#ifdef TBB_FOUND
    // Parallel version using TBB with concurrent_unordered_map (thread-safe, no mutex needed)
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_points), [&](const tbb::blocked_range<size_t> &range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                const size_t idx = i * 3;
                const Tpos x = pts_ptr[idx + 0];
                const Tpos y = pts_ptr[idx + 1];
                const Tpos z = pts_ptr[idx + 2];

                if constexpr (std::is_same_v<Tcolor, std::nullptr_t>) {
                    // No colors: call overload without color parameters
                    update_voxel_lock<Tpos>(x, y, z);
                } else {
                    // Colors provided: read colors and call with color parameters
                    const Tcolor color_x = cols_ptr[idx + 0];
                    const Tcolor color_y = cols_ptr[idx + 1];
                    const Tcolor color_z = cols_ptr[idx + 2];

                    if constexpr (std::is_same_v<Tclass, std::nullptr_t>) {
                        // No semantics
                        update_voxel_lock<Tpos, Tcolor>(x, y, z, color_x, color_y, color_z);
                    } else {
                        // With semantics
                        const Tclass class_id = class_ids_ptr[i];
                        if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                            const Tinstance object_id = instance_ids_ptr[i];
                            // we have instance id
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                const Tdepth depth = depths_ptr[i];
                                update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                                    x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
                            } else {
                                // we do not have depth => use confidence = 1.0
                                update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass>(
                                    x, y, z, color_x, color_y, color_z, class_id, object_id);
                            }
                        } else {
                            // we do not have instance id => use default instance id 0
                            if constexpr (std::is_same_v<Tinstance, std::nullptr_t>) {
                                // Tinstance is nullptr_t, so we can't create a variable, use
                                // literal 0 directly
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    const Tdepth depth = depths_ptr[i];
                                    update_voxel_lock<Tpos, Tcolor, std::nullptr_t, Tclass, Tdepth>(
                                        x, y, z, color_x, color_y, color_z, class_id, nullptr,
                                        depth);
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    update_voxel_lock<Tpos, Tcolor, std::nullptr_t, Tclass>(
                                        x, y, z, color_x, color_y, color_z, class_id, nullptr);
                                }
                            } else {
                                // Tinstance is a real type, use default value 0
                                const Tinstance object_id = 0;
                                if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                    // we have depth
                                    const Tdepth depth = depths_ptr[i];
                                    update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                                        x, y, z, color_x, color_y, color_z, class_id, object_id,
                                        depth);
                                } else {
                                    // we do not have depth => use confidence = 1.0
                                    update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass>(
                                        x, y, z, color_x, color_y, color_z, class_id, object_id);
                                }
                            }
                        }
                    }
                }
            }
        });
#else
    // Sequential version
    for (size_t i = 0; i < num_points; ++i) {
        const size_t idx = i * 3;
        const Tpos x = pts_ptr[idx + 0];
        const Tpos y = pts_ptr[idx + 1];
        const Tpos z = pts_ptr[idx + 2];

        if constexpr (std::is_same_v<Tcolor, std::nullptr_t>) {
            // No colors: call overload without color parameters
            update_voxel_lock<Tpos>(x, y, z);
        } else {
            // Colors provided: read colors and call with color parameters
            const Tcolor color_x = cols_ptr[idx + 0];
            const Tcolor color_y = cols_ptr[idx + 1];
            const Tcolor color_z = cols_ptr[idx + 2];

            if constexpr (std::is_same_v<Tclass, std::nullptr_t>) {
                // No semantics
                update_voxel_lock<Tpos, Tcolor>(x, y, z, color_x, color_y, color_z);
            } else {
                // With semantics
                const Tclass class_id = class_ids_ptr[i];
                if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                    const Tinstance object_id = instance_ids_ptr[i];
                    // we have instance id
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        // we have depth
                        const Tdepth depth = depths_ptr[i];
                        update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                            x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
                    } else {
                        // we do not have depth => use confidence = 1.0
                        update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass>(
                            x, y, z, color_x, color_y, color_z, class_id, object_id);
                    }
                } else {
                    // we do not have instance id => use default instance id 0
                    if constexpr (std::is_same_v<Tinstance, std::nullptr_t>) {
                        // Tinstance is nullptr_t, so we can't create a variable, use nullptr
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            const Tdepth depth = depths_ptr[i];
                            update_voxel_lock<Tpos, Tcolor, std::nullptr_t, Tclass, Tdepth>(
                                x, y, z, color_x, color_y, color_z, class_id, nullptr, depth);
                        } else {
                            // we do not have depth => use confidence = 1.0
                            update_voxel_lock<Tpos, Tcolor, std::nullptr_t, Tclass>(
                                x, y, z, color_x, color_y, color_z, class_id, nullptr);
                        }
                    } else {
                        // Tinstance is a real type, use default value 0
                        const Tinstance object_id = 0;
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            const Tdepth depth = depths_ptr[i];
                            update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                                x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
                        } else {
                            // we do not have depth => use confidence = 1.0
                            update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass>(
                                x, y, z, color_x, color_y, color_z, class_id, object_id);
                        }
                    }
                }
            }
        }
    }
#endif
}

// integrate_raw_preorder_no_block_mutex implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::integrate_raw_preorder_no_block_mutex(
    const Tpos *pts_ptr, size_t num_points, const Tcolor *cols_ptr, const Tclass *class_ids_ptr,
    const Tinstance *instance_ids_ptr, const Tdepth *depths_ptr) {
#ifdef TBB_FOUND
    // Group point indices by block key using thread-local storage
    struct PointInfo {
        size_t point_idx;
        LocalVoxelKey local_key;
    };

    // Use enumerable_thread_specific to collect thread-local maps without contention
    // Each thread builds its own map independently, then we merge sequentially after parallel
    // phase
    using LocalGroupsMap = std::unordered_map<BlockKey, std::vector<PointInfo>, BlockKeyHash>;
    tbb::enumerable_thread_specific<LocalGroupsMap> thread_local_groups([]() {
        LocalGroupsMap map;
        map.reserve(64); // Pre-allocate for typical block count
        return map;
    });

    // Precompute keys and group points by block in parallel
    // Each thread builds its own local map without any synchronization
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points), [&](auto r) {
        // Get thread-local map (created lazily if needed)
        auto &local_groups = thread_local_groups.local();

        for (size_t i = r.begin(); i < r.end(); ++i) {
            const size_t idx = i * 3;
            const Tpos x = pts_ptr[idx + 0], y = pts_ptr[idx + 1], z = pts_ptr[idx + 2];
            const VoxelKey vk = get_voxel_key_inv<Tpos, float>(x, y, z, inv_voxel_size_);
            const BlockKey block_key = get_block_key(vk, block_size_);
            const LocalVoxelKey local_key = get_local_voxel_key(vk, block_key, block_size_);
            local_groups[block_key].push_back({i, local_key});
        }
    });

    // Merge thread-local maps into final result sequentially (no contention)
    // Use std::unordered_map since merge is sequential and subsequent parallel processing is
    // read-only
    std::unordered_map<BlockKey, std::vector<PointInfo>, BlockKeyHash> block_groups;

    // First pass: count total points per key to reserve appropriate vector sizes
    std::unordered_map<BlockKey, size_t, BlockKeyHash> key_point_counts;
    for (const auto &local_groups : thread_local_groups) {
        for (const auto &[key, points] : local_groups) {
            key_point_counts[key] += points.size();
        }
    }

    // Reserve map capacity based on unique keys (avoids rehashing during insert)
    block_groups.reserve(key_point_counts.size());

    // Second pass: merge with pre-reserved vectors to avoid reallocation
    for (auto &local_groups : thread_local_groups) {
        for (auto &[key, points] : local_groups) {
            auto it = block_groups.find(key);
            if (it == block_groups.end()) {
                // Insert a new entry with reserved capacity based on total points for this key
                std::vector<PointInfo> vec;
                vec.reserve(key_point_counts[key]);
                vec.insert(vec.end(), points.begin(), points.end());
                block_groups.emplace(key, std::move(vec));
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

    // Process each block in parallel (one thread per block)
    tbb::parallel_for_each(block_groups.begin(), block_groups.end(), [&](auto &pair) {
        const BlockKey &block_key = pair.first;
        const std::vector<PointInfo> &points = pair.second;

        // Create/find the block once (concurrent map → no external lock needed)
        // Note: tbb::concurrent_unordered_map::insert() returns a pair<iterator, bool>
        auto [it, inserted] = blocks_.insert(std::make_pair(block_key, Block(block_size_)));
        Block &block = it->second;

        // Update all points in this block serially (by this thread only)
        // Use update_voxel_direct to avoid recomputing keys and re-fetching blocks
        // since we already have the block and local_key precomputed
        for (const auto &info : points) {
            const size_t i = info.point_idx;
            const size_t idx = i * 3;
            const Tpos x = pts_ptr[idx + 0], y = pts_ptr[idx + 1], z = pts_ptr[idx + 2];

            if constexpr (std::is_same_v<Tcolor, std::nullptr_t>) {
                // No colors: call overload without color parameters
                update_voxel_direct<Tpos>(block, info.local_key, x, y, z);
            } else {
                // Colors provided: read colors and call with color parameters
                const Tcolor color_x = cols_ptr[idx + 0];
                const Tcolor color_y = cols_ptr[idx + 1];
                const Tcolor color_z = cols_ptr[idx + 2];

                if constexpr (std::is_same_v<Tclass, std::nullptr_t>) {
                    // No semantics
                    update_voxel_direct<Tpos, Tcolor>(block, info.local_key, x, y, z, color_x,
                                                      color_y, color_z);
                } else {
                    // With semantics
                    const Tclass class_id = class_ids_ptr[i];
                    if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                        const Tinstance object_id = instance_ids_ptr[i];
                        // we have instance id
                        if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                            // we have depth
                            const Tdepth depth = depths_ptr[i];
                            update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                                block, info.local_key, x, y, z, color_x, color_y, color_z, class_id,
                                object_id, depth);
                        } else {
                            // we do not have depth => use confidence = 1.0
                            update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass>(
                                block, info.local_key, x, y, z, color_x, color_y, color_z, class_id,
                                object_id);
                        }
                    } else {
                        // we do not have instance id => use default instance id 0
                        if constexpr (std::is_same_v<Tinstance, std::nullptr_t>) {
                            // Tinstance is nullptr_t, so we can't create a variable, use
                            // literal 0 directly
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                const Tdepth depth = depths_ptr[i];
                                update_voxel_direct<Tpos, Tcolor, std::nullptr_t, Tclass, Tdepth>(
                                    block, info.local_key, x, y, z, color_x, color_y, color_z,
                                    class_id, nullptr, depth);
                            } else {
                                // we do not have depth => use confidence = 1.0
                                update_voxel_direct<Tpos, Tcolor, std::nullptr_t, Tclass>(
                                    block, info.local_key, x, y, z, color_x, color_y, color_z,
                                    class_id, nullptr);
                            }
                        } else {
                            // Tinstance is a real type, use default value 0
                            const Tinstance object_id = 0;
                            if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                                // we have depth
                                const Tdepth depth = depths_ptr[i];
                                update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
                                    block, info.local_key, x, y, z, color_x, color_y, color_z,
                                    class_id, object_id, depth);
                            } else {
                                // we do not have depth => use confidence = 1.0
                                update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass>(
                                    block, info.local_key, x, y, z, color_x, color_y, color_z,
                                    class_id, object_id);
                            }
                        }
                    }
                }
            }
        }
    });
#else
    // Sequential version: fall back to integrate_raw_baseline
    integrate_raw_baseline<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
        pts_ptr, num_points, cols_ptr, class_ids_ptr, instance_ids_ptr, depths_ptr);
#endif
}

// update_voxel implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth,
          bool LockMutex>
void VoxelBlockGridT<VoxelDataT>::update_voxel(const Tpos x, const Tpos y, const Tpos z,
                                               const Tcolor color_x, const Tcolor color_y,
                                               const Tcolor color_z, const Tclass class_id,
                                               const Tinstance object_id, const Tdepth depth) {
    // Compute voxel coordinates
    const VoxelKey voxel_key = get_voxel_key_inv<Tpos, Tpos>(x, y, z, inv_voxel_size_);

    // Compute block coordinates using helper function from voxel_hashing.h
    const BlockKey block_key = get_block_key(voxel_key, block_size_);
    const LocalVoxelKey local_key = get_local_voxel_key(voxel_key, block_key, block_size_);

    // Get or create the block (concurrent_unordered_map is thread-safe)
    // Note: tbb::concurrent_unordered_map::insert() returns a pair<iterator, bool>
    auto [block_it, inserted] = blocks_.insert(std::make_pair(block_key, Block(block_size_)));
    Block &block = block_it->second;

    // Conditionally acquire mutex BEFORE accessing voxel data to prevent race conditions
#ifdef TBB_FOUND
    if constexpr (LockMutex) {
        std::lock_guard<std::mutex> lock(*block.mutex);
        update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
            block, local_key, x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
    } else {
        update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
            block, local_key, x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
    }
#else
    update_voxel_direct<Tpos, Tcolor, Tinstance, Tclass, Tdepth>(
        block, local_key, x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
#endif
}

// update_voxel_lock implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::update_voxel_lock(const Tpos x, const Tpos y, const Tpos z,
                                                    const Tcolor color_x, const Tcolor color_y,
                                                    const Tcolor color_z, const Tclass class_id,
                                                    const Tinstance object_id, const Tdepth depth) {
    this->template update_voxel<Tpos, Tcolor, Tinstance, Tclass, Tdepth, true>(
        x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
}

// update_voxel_no_lock implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::update_voxel_no_lock(const Tpos x, const Tpos y, const Tpos z,
                                                       const Tcolor color_x, const Tcolor color_y,
                                                       const Tcolor color_z, const Tclass class_id,
                                                       const Tinstance object_id,
                                                       const Tdepth depth) {
    this->template update_voxel<Tpos, Tcolor, Tinstance, Tclass, Tdepth, false>(
        x, y, z, color_x, color_y, color_z, class_id, object_id, depth);
}

// update_voxel_direct implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass, typename Tdepth>
void VoxelBlockGridT<VoxelDataT>::update_voxel_direct(Block &block, const LocalVoxelKey &local_key,
                                                      const Tpos x, const Tpos y, const Tpos z,
                                                      const Tcolor color_x, const Tcolor color_y,
                                                      const Tcolor color_z, const Tclass class_id,
                                                      const Tinstance object_id,
                                                      const Tdepth depth) {
    // Get voxel using direct array access
    auto &v = block.get_voxel(local_key);

    if (v.count == 0) {
        // New voxel: initialize and update
        v.update_point(x, y, z);
        if constexpr ((!std::is_same_v<Tcolor, std::nullptr_t>)) {
            v.update_color(color_x, color_y, color_z);
        }
        if constexpr (SemanticVoxel<VoxelDataT>) {
            if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                // we have semantics/class id
                if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                    // we have instance id
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        // we have depth
                        if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                            v.initialize_semantics_with_depth(object_id, class_id, depth);
                        } else {
                            v.initialize_semantics(object_id, class_id);
                        }
                    } else {
                        // we do not have depth => use confidence = 1.0
                        v.initialize_semantics(object_id, class_id);
                    }
                } else {
                    // we do not have instance id => use default instance id 0
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        // we have depth
                        if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                            v.initialize_semantics_with_depth(0, class_id, depth);
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
        if constexpr ((!std::is_same_v<Tcolor, std::nullptr_t>)) {
            v.update_color(color_x, color_y, color_z);
        }
        if constexpr (SemanticVoxel<VoxelDataT>) {
            if constexpr (!std::is_same_v<Tclass, std::nullptr_t>) {
                // we have semantics/class id
                if constexpr (!std::is_same_v<Tinstance, std::nullptr_t>) {
                    // we have instance id
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        // we have depth
                        if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                            v.update_semantics_with_depth(object_id, class_id, depth);
                        } else {
                            v.update_semantics(object_id, class_id);
                        }
                    } else {
                        // we do not have depth => use confidence = 1.0
                        v.update_semantics(object_id, class_id);
                    }
                } else {
                    // we do not have instance id => use default instance id 0
                    if constexpr (!std::is_same_v<Tdepth, std::nullptr_t>) {
                        // we have depth
                        if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
                            v.update_semantics_with_depth(0, class_id, depth);
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

template <typename VoxelDataT>
void VoxelBlockGridT<VoxelDataT>::carve(const CameraFrustrum &camera_frustrum,
                                        const cv::Mat &depth_image, const float depth_threshold) {

    using ThisType = VoxelBlockGridT<VoxelDataT>;
    volumetric::carve<ThisType, VoxelDataT>(*this, camera_frustrum, depth_image, depth_threshold);
};

// remove_low_count_voxels implementation
template <typename VoxelDataT>
void VoxelBlockGridT<VoxelDataT>::remove_low_count_voxels(const int min_count) {
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

// remove_low_confidence_voxels implementation
template <typename VoxelDataT>
void VoxelBlockGridT<VoxelDataT>::remove_low_confidence_voxels(const float min_confidence) {
    if constexpr (SemanticVoxel<VoxelDataT>) {
#ifdef TBB_FOUND
        // Parallel version
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](auto &pair) {
            Block &block = pair.second;
            std::lock_guard<std::mutex> lock(*block.mutex);
            for (auto &v : block.data) {
                if (v.get_confidence() < min_confidence) {
                    v.reset();
                }
            }
        });
#else
        // Sequential version
        for (auto &[block_key, block] : blocks_) {
            for (auto &v : block.data) {
                if (v.get_confidence() < min_confidence) {
                    v.reset();
                }
            }
        }
#endif
    }
}

// get_points implementation
template <typename VoxelDataT>
std::vector<typename VoxelBlockGridT<VoxelDataT>::Pos3>
VoxelBlockGridT<VoxelDataT>::get_points() const {
    std::vector<Pos3> points;
    // Conservative reserve: upper bound to avoid double traversal
    // (may slightly over-allocate, but avoids expensive get_total_voxel_count() call)
    points.reserve(num_voxels_per_block_ * blocks_.size());

    for (const auto &[block_key, block] : blocks_) {
        for (const auto &v : block.data) {
            if (v.count > 0) {
                points.push_back(v.get_position());
            }
        }
    }
    return points;
}

// get_colors implementation
template <typename VoxelDataT>
std::vector<typename VoxelBlockGridT<VoxelDataT>::Color3>
VoxelBlockGridT<VoxelDataT>::get_colors() const {
    std::vector<Color3> colors;
    // Conservative reserve: upper bound to avoid double traversal
    // (may slightly over-allocate, but avoids expensive get_total_voxel_count() call)
    colors.reserve(num_voxels_per_block_ * blocks_.size());

    for (const auto &[block_key, block] : blocks_) {
        for (const auto &v : block.data) {
            if (v.count > 0) {
                colors.push_back(v.get_color());
            }
        }
    }
    return colors;
}

// get_voxels implementation
template <typename VoxelDataT>
typename VoxelBlockGridT<VoxelDataT>::VoxelGridDataType
VoxelBlockGridT<VoxelDataT>::get_voxels(int min_count, float min_confidence) const {
#ifdef TBB_FOUND
    // Parallel version: use enumerable_thread_specific to collect data without contention
    // Each thread processes distinct blocks and collects data independently
    // Use member variable for reservation size (capture by reference in lambda)
    const size_t reserve_size = num_voxels_per_block_;
    tbb::enumerable_thread_specific<VoxelGridDataType> thread_local_results([reserve_size]() {
        VoxelGridDataType result;
        // Reserve space for typical block (may be overestimate, but better than reallocation)
        result.points.reserve(reserve_size);
        result.colors.reserve(reserve_size);
        if constexpr (SemanticVoxel<VoxelDataT>) {
            result.object_ids.reserve(reserve_size);
            result.class_ids.reserve(reserve_size);
            result.confidences.reserve(reserve_size);
        }
        return result;
    });

    tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](const auto &pair) {
        // Get thread-local result (created lazily if needed)
        auto &local_result = thread_local_results.local();
        const auto &block = pair.second;

        bool check_point;
        float confidence;
        // Process all voxels in this block
        for (const auto &v : block.data) {
            if constexpr (SemanticVoxel<VoxelDataT>) {
                confidence = v.get_confidence(); // compute confidence only once
                check_point = v.count >= min_count && confidence >= min_confidence;
            } else {
                check_point = v.count >= min_count;
            }
            if (check_point) {
                local_result.points.push_back(v.get_position());
                local_result.colors.push_back(v.get_color());
                if constexpr (SemanticVoxel<VoxelDataT>) {
                    local_result.object_ids.push_back(v.get_object_id());
                    local_result.class_ids.push_back(v.get_class_id());
                    local_result.confidences.push_back(confidence);
                }
            }
        }
    });

    // Merge thread-local results sequentially (no contention)
    VoxelGridDataType result;
    for (auto &local_result : thread_local_results) {
        if (!local_result.points.empty()) {
            result.points.insert(result.points.end(), local_result.points.begin(),
                                 local_result.points.end());
            result.colors.insert(result.colors.end(), local_result.colors.begin(),
                                 local_result.colors.end());
            if constexpr (SemanticVoxel<VoxelDataT>) {
                result.object_ids.insert(result.object_ids.end(), local_result.object_ids.begin(),
                                         local_result.object_ids.end());
                result.class_ids.insert(result.class_ids.end(), local_result.class_ids.begin(),
                                        local_result.class_ids.end());
                result.confidences.insert(result.confidences.end(),
                                          local_result.confidences.begin(),
                                          local_result.confidences.end());
            }
        }
    }
#else
    // Sequential version
    VoxelGridDataType result;
    const size_t upper_bound_num_voxels = num_voxels_per_block_ * blocks_.size();
    result.points.reserve(upper_bound_num_voxels);
    result.colors.reserve(upper_bound_num_voxels);
    if constexpr (SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(upper_bound_num_voxels);
        result.class_ids.reserve(upper_bound_num_voxels);
        result.confidences.reserve(upper_bound_num_voxels);
    }

    bool check_point;
    float confidence;
    for (const auto &[block_key, block] : blocks_) {
        for (const auto &v : block.data) {
            if constexpr (SemanticVoxel<VoxelDataT>) {
                confidence = v.get_confidence(); // compute confidence only once
                check_point = v.count >= min_count && confidence >= min_confidence;
            } else {
                check_point = v.count >= min_count;
            }
            if (check_point) {
                result.points.push_back(v.get_position());
                result.colors.push_back(v.get_color());
                if constexpr (SemanticVoxel<VoxelDataT>) {
                    result.object_ids.push_back(v.get_object_id());
                    result.class_ids.push_back(v.get_class_id());
                    result.confidences.push_back(confidence);
                }
            }
        }
    }
#endif
    return result;
}

// get_voxels_in_bb implementation
template <typename VoxelDataT>
template <bool IncludeSemantics>
typename VoxelBlockGridT<VoxelDataT>::VoxelGridDataType
VoxelBlockGridT<VoxelDataT>::get_voxels_in_bb(const BoundingBox3D &bbox, const int min_count,
                                              float min_confidence) const {
    // Convert spatial bounds to voxel key bounds
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    // Convert voxel key bounds to block key bounds
    const BlockKey min_block_key = get_block_key(min_key, block_size_);
    const BlockKey max_block_key = get_block_key(max_key, block_size_);

    VoxelGridDataType result;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing blocks, filter by key range, then check
    // voxels Use enumerable_thread_specific to collect data without contention
    const size_t reserve_size = num_voxels_per_block_;
    tbb::enumerable_thread_specific<VoxelGridDataType> thread_local_results([reserve_size]() {
        VoxelGridDataType local_result;
        local_result.points.reserve(reserve_size);
        local_result.colors.reserve(reserve_size);
        if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
            local_result.object_ids.reserve(reserve_size);
            local_result.class_ids.reserve(reserve_size);
            local_result.confidences.reserve(reserve_size);
        }
        return local_result;
    });

    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](const auto &pair) {
            const BlockKey &block_key = pair.first;
            const auto &block = pair.second;

            // Fast block key-range check - rejects most blocks immediately
            // Note: min_block_key and max_block_key are computed from voxel key bounds using
            // floor_div, so this range includes all blocks that could contain voxels in the
            // bounding box, including blocks that are partially overlapping (some voxels
            // inside, some outside)
            if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
                block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
                block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
                return; // Skip blocks outside key range
            }

            // Get thread-local result (created lazily if needed)
            auto &local_result = thread_local_results.local();

            bool check_point;
            float confidence;
            // Iterate through voxels in this block
            for (int lx = 0; lx < block_size_; ++lx) {
                for (int ly = 0; ly < block_size_; ++ly) {
                    for (int lz = 0; lz < block_size_; ++lz) {
                        LocalVoxelKey local_key(lx, ly, lz);
                        const size_t idx = block.get_index(local_key);
                        const auto &v = block.data[idx];

                        // Check count/confidence criteria early
                        if constexpr (SemanticVoxel<VoxelDataT>) {
                            confidence = v.get_confidence();
                            check_point = v.count >= min_count && confidence >= min_confidence;
                        } else {
                            check_point = v.count >= min_count;
                        }
                        if (!check_point) {
                            continue;
                        }

                        // Convert local voxel key to global voxel key
                        VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                           block_key.y * block_size_ + ly,
                                           block_key.z * block_size_ + lz);

                        // Fast voxel key-range check
                        if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                            voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                            voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                            continue;
                        }

                        // Fine-grained position check
                        const auto pos = v.get_position();
                        if (bbox.contains(pos[0], pos[1], pos[2])) {
                            local_result.points.push_back(pos);
                            local_result.colors.push_back(v.get_color());
                            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                local_result.object_ids.push_back(v.get_object_id());
                                local_result.class_ids.push_back(v.get_class_id());
                                local_result.confidences.push_back(confidence);
                            }
                        }
                    }
                }
            }
        });
    });

    // Merge thread-local results sequentially (no contention)
    for (auto &local_result : thread_local_results) {
        if (!local_result.points.empty()) {
            result.points.insert(result.points.end(), local_result.points.begin(),
                                 local_result.points.end());
            result.colors.insert(result.colors.end(), local_result.colors.begin(),
                                 local_result.colors.end());
            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                result.object_ids.insert(result.object_ids.end(), local_result.object_ids.begin(),
                                         local_result.object_ids.end());
                result.class_ids.insert(result.class_ids.end(), local_result.class_ids.begin(),
                                        local_result.class_ids.end());
                result.confidences.insert(result.confidences.end(),
                                          local_result.confidences.begin(),
                                          local_result.confidences.end());
            }
        }
    }
#else
    // Sequential version: iterate existing blocks with key-range filtering
    result.points.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    result.colors.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(num_voxels_per_block_ * blocks_.size() / 4);
        result.class_ids.reserve(num_voxels_per_block_ * blocks_.size() / 4);
        result.confidences.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    }

    bool check_point;
    float confidence;
    // Iterate through existing blocks and filter by block key range (fast integer comparison)
    // This avoids O(block_volume) hash lookups and skips most blocks outside the bounding box
    for (const auto &[block_key, block] : blocks_) {
        // Fast block key-range check - rejects most blocks immediately
        // Note: min_block_key and max_block_key are computed from voxel key bounds using
        // floor_div, so this range includes all blocks that could contain voxels in the
        // bounding box, including blocks that are partially overlapping (some voxels inside,
        // some outside)
        if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
            block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
            block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
            continue; // Skip blocks outside key range
        }

        // Iterate through voxels in this block
        for (int lx = 0; lx < block_size_; ++lx) {
            for (int ly = 0; ly < block_size_; ++ly) {
                for (int lz = 0; lz < block_size_; ++lz) {
                    LocalVoxelKey local_key(lx, ly, lz);
                    const size_t idx = block.get_index(local_key);
                    const auto &v = block.data[idx];

                    // Check count/confidence criteria early
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        confidence = v.get_confidence();
                        check_point = v.count >= min_count && confidence >= min_confidence;
                    } else {
                        check_point = v.count >= min_count;
                    }
                    if (!check_point) {
                        continue;
                    }

                    // Convert local voxel key to global voxel key
                    VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                       block_key.y * block_size_ + ly,
                                       block_key.z * block_size_ + lz);

                    // Fast voxel key-range check
                    if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                        voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                        voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                        continue;
                    }

                    // Fine-grained position check
                    const auto pos = v.get_position();
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
            }
        }
    }
#endif

    return result;
}

// get_voxels_in_bb implementation
template <typename VoxelDataT>
template <bool IncludeSemantics>
typename VoxelBlockGridT<VoxelDataT>::VoxelGridDataType
VoxelBlockGridT<VoxelDataT>::get_voxels_in_camera_frustrum(const CameraFrustrum &camera_frustrum,
                                                           const int min_count,
                                                           float min_confidence) const {
    const BoundingBox3D bbox = camera_frustrum.get_bbox();
    // Convert spatial bounds to voxel key bounds
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    // Convert voxel key bounds to block key bounds
    const BlockKey min_block_key = get_block_key(min_key, block_size_);
    const BlockKey max_block_key = get_block_key(max_key, block_size_);

    VoxelGridDataType result;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing blocks, filter by key range, then check
    // frustum Use enumerable_thread_specific to collect data without contention
    const size_t reserve_size = num_voxels_per_block_;
    tbb::enumerable_thread_specific<VoxelGridDataType> thread_local_results([reserve_size]() {
        VoxelGridDataType local_result;
        local_result.points.reserve(reserve_size);
        local_result.colors.reserve(reserve_size);
        if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
            local_result.object_ids.reserve(reserve_size);
            local_result.class_ids.reserve(reserve_size);
            local_result.confidences.reserve(reserve_size);
        }
        return local_result;
    });

    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](const auto &pair) {
            const BlockKey &block_key = pair.first;
            const auto &block = pair.second;

            // Fast block key-range check - rejects most blocks immediately
            // Note: min_block_key and max_block_key are computed from voxel key bounds using
            // floor_div, so this range includes all blocks that could contain voxels in the
            // bounding box, including blocks that are partially overlapping (some voxels
            // inside, some outside)
            if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
                block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
                block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
                return; // Skip blocks outside key range
            }

            // Get thread-local result (created lazily if needed)
            auto &local_result = thread_local_results.local();

            bool check_point;
            float confidence;
            // Iterate through voxels in this block
            for (int lx = 0; lx < block_size_; ++lx) {
                for (int ly = 0; ly < block_size_; ++ly) {
                    for (int lz = 0; lz < block_size_; ++lz) {
                        LocalVoxelKey local_key(lx, ly, lz);
                        const size_t idx = block.get_index(local_key);
                        const auto &v = block.data[idx];

                        // Check count/confidence criteria early
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
                        const auto [is_in_frustum, image_point] =
                            camera_frustrum.contains(pos[0], pos[1], pos[2]);
                        if (is_in_frustum) {
                            local_result.points.push_back(pos);
                            local_result.colors.push_back(v.get_color());
                            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                                local_result.object_ids.push_back(v.get_object_id());
                                local_result.class_ids.push_back(v.get_class_id());
                                local_result.confidences.push_back(confidence);
                            }
                        }
                    }
                }
            }
        });
    });

    // Merge thread-local results sequentially (no contention)
    for (auto &local_result : thread_local_results) {
        if (!local_result.points.empty()) {
            result.points.insert(result.points.end(), local_result.points.begin(),
                                 local_result.points.end());
            result.colors.insert(result.colors.end(), local_result.colors.begin(),
                                 local_result.colors.end());
            if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
                result.object_ids.insert(result.object_ids.end(), local_result.object_ids.begin(),
                                         local_result.object_ids.end());
                result.class_ids.insert(result.class_ids.end(), local_result.class_ids.begin(),
                                        local_result.class_ids.end());
                result.confidences.insert(result.confidences.end(),
                                          local_result.confidences.begin(),
                                          local_result.confidences.end());
            }
        }
    }
#else
    // Sequential version: iterate existing blocks with key-range filtering
    result.points.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    result.colors.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    if constexpr (IncludeSemantics && SemanticVoxel<VoxelDataT>) {
        result.object_ids.reserve(num_voxels_per_block_ * blocks_.size() / 4);
        result.class_ids.reserve(num_voxels_per_block_ * blocks_.size() / 4);
        result.confidences.reserve(num_voxels_per_block_ * blocks_.size() / 4);
    }

    bool check_point;
    float confidence;
    // Iterate through existing blocks and filter by block key range (fast integer comparison)
    // This avoids O(block_volume) hash lookups and skips most blocks outside the bounding box
    for (const auto &[block_key, block] : blocks_) {
        // Fast block key-range check - rejects most blocks immediately
        // Note: min_block_key and max_block_key are computed from voxel key bounds using
        // floor_div, so this range includes all blocks that could contain voxels in the
        // bounding box, including blocks that are partially overlapping (some voxels inside,
        // some outside)
        if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
            block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
            block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
            continue; // Skip blocks outside key range
        }

        // Iterate through voxels in this block
        for (int lx = 0; lx < block_size_; ++lx) {
            for (int ly = 0; ly < block_size_; ++ly) {
                for (int lz = 0; lz < block_size_; ++lz) {
                    LocalVoxelKey local_key(lx, ly, lz);
                    const size_t idx = block.get_index(local_key);
                    const auto &v = block.data[idx];

                    // Check count/confidence criteria early
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
                    const auto [is_in_frustum, image_point] =
                        camera_frustrum.contains(pos[0], pos[1], pos[2]);
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
            }
        }
    }
#endif

    return result;
}

// iterate_voxels_in_bb implementation
template <typename VoxelDataT>
template <typename Callback>
void VoxelBlockGridT<VoxelDataT>::iterate_voxels_in_bb(const BoundingBox3D &bbox,
                                                       Callback &&callback, int min_count,
                                                       float min_confidence) const {
    // Convert spatial bounds to voxel key bounds
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    // Convert voxel key bounds to block key bounds
    const BlockKey min_block_key = get_block_key(min_key, block_size_);
    const BlockKey max_block_key = get_block_key(max_key, block_size_);

    bool check_point;
    float confidence;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing blocks, filter by key range, then check
    // voxels. The callback must be thread-safe.
    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](const auto &pair) {
            const BlockKey &block_key = pair.first;
            const auto &block = pair.second;

            // Fast block key-range check - rejects most blocks immediately
            // Note: min_block_key and max_block_key are computed from voxel key bounds using
            // floor_div, so this range includes all blocks that could contain voxels in the
            // bounding box, including blocks that are partially overlapping (some voxels
            // inside, some outside)
            if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
                block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
                block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
                return; // Skip blocks outside key range
            }

            // Iterate through voxels in this block
            for (int lx = 0; lx < block_size_; ++lx) {
                for (int ly = 0; ly < block_size_; ++ly) {
                    for (int lz = 0; lz < block_size_; ++lz) {
                        LocalVoxelKey local_key(lx, ly, lz);
                        const size_t idx = block.get_index(local_key);
                        const auto &v = block.data[idx];

                        // Check count/confidence criteria early
                        if constexpr (SemanticVoxel<VoxelDataT>) {
                            confidence = v.get_confidence();
                            check_point = v.count >= min_count && confidence >= min_confidence;
                        } else {
                            check_point = v.count >= min_count;
                        }
                        if (!check_point) {
                            continue;
                        }

                        // Convert local voxel key to global voxel key
                        VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                           block_key.y * block_size_ + ly,
                                           block_key.z * block_size_ + lz);

                        // Fast voxel key-range check
                        if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                            voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                            voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                            continue;
                        }

                        // Fine-grained position check
                        const auto pos = v.get_position();
                        if (bbox.contains(pos[0], pos[1], pos[2])) {
                            callback(v, pos, voxel_key);
                        }
                    }
                }
            }
        });
    });
#else
    // Sequential version: iterate existing blocks and filter by block key range (fast integer
    // comparison) This avoids O(block_volume) hash lookups and skips most blocks outside the
    // bounding box
    for (const auto &[block_key, block] : blocks_) {
        // Fast block key-range check - rejects most blocks immediately
        // Note: min_block_key and max_block_key are computed from voxel key bounds using
        // floor_div, so this range includes all blocks that could contain voxels in the
        // bounding box, including blocks that are partially overlapping (some voxels inside,
        // some outside)
        if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
            block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
            block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
            continue; // Skip blocks outside key range
        }

        // Iterate through voxels in this block
        for (int lx = 0; lx < block_size_; ++lx) {
            for (int ly = 0; ly < block_size_; ++ly) {
                for (int lz = 0; lz < block_size_; ++lz) {
                    LocalVoxelKey local_key(lx, ly, lz);
                    const size_t idx = block.get_index(local_key);
                    const auto &v = block.data[idx];

                    // Check count/confidence criteria early
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        confidence = v.get_confidence();
                        check_point = v.count >= min_count && confidence >= min_confidence;
                    } else {
                        check_point = v.count >= min_count;
                    }
                    if (!check_point) {
                        continue;
                    }

                    // Convert local voxel key to global voxel key
                    VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                       block_key.y * block_size_ + ly,
                                       block_key.z * block_size_ + lz);

                    // Fast voxel key-range check
                    if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                        voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                        voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                        continue;
                    }

                    // Fine-grained position check
                    const auto pos = v.get_position();
                    if (bbox.contains(pos[0], pos[1], pos[2])) {
                        callback(v, pos, voxel_key);
                    }
                }
            }
        }
    }
#endif
}

// iterate_voxels_in_camera_frustrum implementation
template <typename VoxelDataT>
template <typename Callback>
void VoxelBlockGridT<VoxelDataT>::iterate_voxels_in_camera_frustrum(
    const CameraFrustrum &camera_frustrum, Callback &&callback, int min_count,
    float min_confidence) {
    const BoundingBox3D bbox = camera_frustrum.get_bbox();
    // Convert bounding box to voxel key bounds for fast filtering
    const VoxelKey min_key =
        get_voxel_key_inv<double, double>(bbox.min_x, bbox.min_y, bbox.min_z, inv_voxel_size_);
    const VoxelKey max_key =
        get_voxel_key_inv<double, double>(bbox.max_x, bbox.max_y, bbox.max_z, inv_voxel_size_);

    // Convert voxel key bounds to block key bounds
    const BlockKey min_block_key = get_block_key(min_key, block_size_);
    const BlockKey max_block_key = get_block_key(max_key, block_size_);

    bool check_point;
    float confidence;
#ifdef TBB_FOUND
    // Optimized parallel version: iterate existing blocks, filter by key range, then check
    // voxels. The callback must be thread-safe.
    tbb::this_task_arena::isolate([&]() {
        tbb::parallel_for_each(blocks_.begin(), blocks_.end(), [&](auto &pair) {
            const BlockKey &block_key = pair.first;
            auto &block = pair.second;

            // Fast block key-range check - rejects most blocks immediately
            // Note: min_block_key and max_block_key are computed from voxel key bounds using
            // floor_div, so this range includes all blocks that could contain voxels in the
            // bounding box, including blocks that are partially overlapping (some voxels
            // inside, some outside)
            if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
                block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
                block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
                return; // Skip blocks outside key range
            }

            // Iterate through voxels in this block
            for (int lx = 0; lx < block_size_; ++lx) {
                for (int ly = 0; ly < block_size_; ++ly) {
                    for (int lz = 0; lz < block_size_; ++lz) {
                        LocalVoxelKey local_key(lx, ly, lz);
                        const size_t idx = block.get_index(local_key);
                        auto &v = block.data[idx];

                        // Check count/confidence criteria early
                        if constexpr (SemanticVoxel<VoxelDataT>) {
                            confidence = v.get_confidence();
                            check_point = v.count >= min_count && confidence >= min_confidence;
                        } else {
                            check_point = v.count >= min_count;
                        }
                        if (!check_point) {
                            continue;
                        }

                        // Convert local voxel key to global voxel key
                        VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                           block_key.y * block_size_ + ly,
                                           block_key.z * block_size_ + lz);

                        // Fast voxel key-range check
                        if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                            voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                            voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                            continue;
                        }

                        // Fine-grained position check
                        const auto pos = v.get_position();
                        const auto [is_in_frustum, image_point] =
                            camera_frustrum.contains(pos[0], pos[1], pos[2]);
                        if (is_in_frustum) {
                            using Pos3d = std::array<double, 3>;
                            using Pos3f = std::array<float, 3>;
                            using Pos3Type = typename VoxelBlockGridT<VoxelDataT>::Pos3;
                            if constexpr (std::is_invocable_v<Callback, VoxelDataT &,
                                                              const VoxelKey &, const Pos3Type &,
                                                              const ImagePoint &>) {
                                callback(v, voxel_key, pos, image_point);
                            } else if constexpr (std::is_invocable_v<
                                                     Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3d &, const ImagePoint &>) {
                                const Pos3d pos_d = {
                                    static_cast<double>(pos[0]),
                                    static_cast<double>(pos[1]),
                                    static_cast<double>(pos[2]),
                                };
                                callback(v, voxel_key, pos_d, image_point);
                            } else if constexpr (std::is_invocable_v<
                                                     Callback, VoxelDataT &, const VoxelKey &,
                                                     const Pos3f &, const ImagePoint &>) {
                                const Pos3f pos_f = {
                                    static_cast<float>(pos[0]),
                                    static_cast<float>(pos[1]),
                                    static_cast<float>(pos[2]),
                                };
                                callback(v, voxel_key, pos_f, image_point);
                            } else {
                                struct UnsupportedCallbackSignature : std::false_type {};
                                static_assert(UnsupportedCallbackSignature::value,
                                              "Callback must accept one of: "
                                              "(VoxelDataT&, VoxelKey, Pos3, ImagePoint), "
                                              "(VoxelDataT&, VoxelKey, std::array<double,3>, "
                                              "ImagePoint), "
                                              "(VoxelDataT&, VoxelKey, std::array<float,3>, "
                                              "ImagePoint).");
                            }
                        }
                    }
                }
            }
        });
    });
#else
    // Sequential version: iterate existing blocks and filter by block key range (fast integer
    // comparison) This avoids O(block_volume) hash lookups and skips most blocks outside the
    // bounding box
    for (auto &[block_key, block] : blocks_) {
        // Fast block key-range check - rejects most blocks immediately
        // Note: min_block_key and max_block_key are computed from voxel key bounds using
        // floor_div, so this range includes all blocks that could contain voxels in the
        // bounding box, including blocks that are partially overlapping (some voxels inside,
        // some outside)
        if (block_key.x < min_block_key.x || block_key.x > max_block_key.x ||
            block_key.y < min_block_key.y || block_key.y > max_block_key.y ||
            block_key.z < min_block_key.z || block_key.z > max_block_key.z) {
            continue; // Skip blocks outside key range
        }

        // Iterate through voxels in this block
        for (int lx = 0; lx < block_size_; ++lx) {
            for (int ly = 0; ly < block_size_; ++ly) {
                for (int lz = 0; lz < block_size_; ++lz) {
                    LocalVoxelKey local_key(lx, ly, lz);
                    const size_t idx = block.get_index(local_key);
                    auto &v = block.data[idx];

                    // Check count/confidence criteria early
                    if constexpr (SemanticVoxel<VoxelDataT>) {
                        confidence = v.get_confidence();
                        check_point = v.count >= min_count && confidence >= min_confidence;
                    } else {
                        check_point = v.count >= min_count;
                    }
                    if (!check_point) {
                        continue;
                    }

                    // Convert local voxel key to global voxel key
                    VoxelKey voxel_key(block_key.x * block_size_ + lx,
                                       block_key.y * block_size_ + ly,
                                       block_key.z * block_size_ + lz);

                    // Fast voxel key-range check
                    if (voxel_key.x < min_key.x || voxel_key.x > max_key.x ||
                        voxel_key.y < min_key.y || voxel_key.y > max_key.y ||
                        voxel_key.z < min_key.z || voxel_key.z > max_key.z) {
                        continue;
                    }

                    // Fine-grained position check
                    const auto pos = v.get_position();
                    const auto [is_in_frustum, image_point] =
                        camera_frustrum.contains(pos[0], pos[1], pos[2]);
                    if (is_in_frustum) {
                        using Pos3d = std::array<double, 3>;
                        using Pos3f = std::array<float, 3>;
                        using Pos3Type = typename VoxelBlockGridT<VoxelDataT>::Pos3;
                        if constexpr (std::is_invocable_v<Callback, VoxelDataT &, const VoxelKey &,
                                                          const Pos3Type &, const ImagePoint &>) {
                            callback(v, voxel_key, pos, image_point);
                        } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &,
                                                                 const Pos3d &,
                                                                 const ImagePoint &>) {
                            const Pos3d pos_d = {
                                static_cast<double>(pos[0]),
                                static_cast<double>(pos[1]),
                                static_cast<double>(pos[2]),
                            };
                            callback(v, voxel_key, pos_d, image_point);
                        } else if constexpr (std::is_invocable_v<Callback, VoxelDataT &,
                                                                 const VoxelKey &, const Pos3f &,
                                                                 const ImagePoint &>) {
                            const Pos3f pos_f = {
                                static_cast<float>(pos[0]),
                                static_cast<float>(pos[1]),
                                static_cast<float>(pos[2]),
                            };
                            callback(v, voxel_key, pos_f, image_point);
                        } else {
                            struct UnsupportedCallbackSignature : std::false_type {};
                            static_assert(
                                UnsupportedCallbackSignature::value,
                                "Callback must accept one of: "
                                "(VoxelDataT&, VoxelKey, Pos3, ImagePoint), "
                                "(VoxelDataT&, VoxelKey, std::array<double,3>, ImagePoint), "
                                "(VoxelDataT&, VoxelKey, std::array<float,3>, ImagePoint).");
                        }
                    }
                }
            }
        }
    }
#endif
}

// clear implementation
template <typename VoxelDataT> void VoxelBlockGridT<VoxelDataT>::clear() { blocks_.clear(); }

// num_blocks implementation
template <typename VoxelDataT> size_t VoxelBlockGridT<VoxelDataT>::num_blocks() const {
    return blocks_.size();
}

// size implementation
template <typename VoxelDataT> size_t VoxelBlockGridT<VoxelDataT>::size() const {
    return get_total_voxel_count();
}

// empty implementation
template <typename VoxelDataT> bool VoxelBlockGridT<VoxelDataT>::empty() const {
    return blocks_.empty();
}

// get_block_size implementation
template <typename VoxelDataT> int VoxelBlockGridT<VoxelDataT>::get_block_size() const {
    return block_size_;
}

// get_total_voxel_count implementation
template <typename VoxelDataT> size_t VoxelBlockGridT<VoxelDataT>::get_total_voxel_count() const {
    size_t total = 0;
    for (const auto &[block_key, block] : blocks_) {
        total += block.count_active();
    }
    return total;
}

} // namespace volumetric

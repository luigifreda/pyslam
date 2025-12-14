namespace volumetric {

// Constructor implementation
template <typename VoxelDataT>
VoxelBlockSemanticGridT<VoxelDataT>::VoxelBlockSemanticGridT(double voxel_size, int block_size)
    : VoxelBlockGridT<VoxelDataT>(voxel_size, block_size) {
    static_assert(SemanticVoxel<VoxelDataT>, "VoxelDataT must satisfy the SemanticVoxel concept");
}

template <typename VoxelDataT>
MapInstanceIdToObjectId VoxelBlockSemanticGridT<VoxelDataT>::assign_object_ids_to_instance_ids(
    const CameraFrustrum &camera_frustrum, const cv::Mat &class_ids_image,
    const cv::Mat &semantic_instances_image, const cv::Mat &depth_image,
    const float depth_threshold, bool do_carving, const float min_vote_ratio, const int min_votes) {

    using ThisType = VoxelBlockSemanticGridT<VoxelDataT>;
    return ::volumetric::assign_object_ids_to_instance_ids<ThisType, VoxelDataT>(
        *this, camera_frustrum, class_ids_image, semantic_instances_image, depth_image,
        depth_threshold, do_carving, min_vote_ratio, min_votes);
}

// set_depth_threshold implementation
template <typename VoxelDataT>
void VoxelBlockSemanticGridT<VoxelDataT>::set_depth_threshold(float depth_threshold) {
    if constexpr (SemanticVoxelWithDepth<VoxelDataT>) {
        VoxelDataT::kDepthThreshold = depth_threshold;
    }
}

// set_depth_decay_rate implementation
template <typename VoxelDataT>
void VoxelBlockSemanticGridT<VoxelDataT>::set_depth_decay_rate(float depth_decay_rate) {
    if constexpr (std::is_same_v<VoxelDataT, VoxelSemanticDataProbabilistic>) {
        VoxelDataT::kDepthDecayRate = depth_decay_rate;
    }
}

// integrate_segment implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass>
void VoxelBlockSemanticGridT<VoxelDataT>::integrate_segment(py::array_t<Tpos> points,
                                                            py::array_t<Tcolor> colors,
                                                            const Tinstance &object_id,
                                                            const Tclass &class_id) {
    auto pts_info = points.request();
    auto cols_info = colors.request();
    integrate_segment_raw<Tpos, Tcolor, Tinstance, Tclass>(
        static_cast<const Tpos *>(pts_info.ptr), pts_info.shape[0],
        static_cast<const Tcolor *>(cols_info.ptr), class_id, object_id);
}

// integrate_segment_raw implementation
template <typename VoxelDataT>
template <typename Tpos, typename Tcolor, typename Tinstance, typename Tclass>
void VoxelBlockSemanticGridT<VoxelDataT>::integrate_segment_raw(const Tpos *pts_ptr,
                                                                const size_t num_points,
                                                                const Tcolor *cols_ptr,
                                                                const Tclass &class_id,
                                                                const Tinstance &object_id) {
    // Skip points with invalid instance or class IDs
    if (object_id < 0 || class_id < 0) {
        return;
    }
#ifdef TBB_FOUND
    // Parallel version using TBB with concurrent_unordered_map (thread-safe, no mutex needed)
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_points), [&](const tbb::blocked_range<size_t> &range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
#else
    // Sequential version
    for (size_t i = 0; i < num_points; ++i) {
#endif
                const size_t idx = i * 3;
                const Tpos x = pts_ptr[idx + 0];
                const Tpos y = pts_ptr[idx + 1];
                const Tpos z = pts_ptr[idx + 2];

                if constexpr ((std::is_same_v<Tcolor, std::nullptr_t>)) {
                    this->template update_voxel_lock<Tpos>(x, y, z);
                } else {
                    const Tcolor color_x = cols_ptr[idx + 0];
                    const Tcolor color_y = cols_ptr[idx + 1];
                    const Tcolor color_z = cols_ptr[idx + 2];

                    if constexpr ((std::is_same_v<Tinstance, std::nullptr_t>) &&
                                  (std::is_same_v<Tclass, std::nullptr_t>)) {
                        this->template update_voxel_lock<Tpos, Tcolor>(x, y, z, color_x, color_y,
                                                                       color_z);
                    } else {
                        // With semantics
                        this->template update_voxel_lock<Tpos, Tcolor, Tinstance, Tclass>(
                            x, y, z, color_x, color_y, color_z, class_id, object_id);
                    }
                }
            }
#ifdef TBB_FOUND
        });
#endif
}

// merge_segments implementation
template <typename VoxelDataT>
void VoxelBlockSemanticGridT<VoxelDataT>::merge_segments(const int instance_id1,
                                                         const int instance_id2) {
#ifdef TBB_FOUND
    // Parallel version using TBB - concurrent_unordered_map is thread-safe
    tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
        auto &block = pair.second;
        std::lock_guard<std::mutex> lock(*block.mutex);
        for (auto &v : block.data) {
            if (v.get_object_id() == instance_id2) {
                v.set_object_id(instance_id1);
            }
        }
    });
#else
    // Sequential version
    for (auto &[block_key, block] : this->blocks_) {
        for (auto &v : block.data) {
            if (v.get_object_id() == instance_id2) {
                v.set_object_id(instance_id1);
            }
        }
    }
#endif
}

// remove_segment implementation
template <typename VoxelDataT>
void VoxelBlockSemanticGridT<VoxelDataT>::remove_segment(const int object_id) {
#ifdef TBB_FOUND
    // For concurrent_unordered_map, we need to iterate through blocks and mark voxels
    // Since we can't efficiently remove individual voxels from blocks, we mark them as inactive
    tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
        auto &block = pair.second;
        std::lock_guard<std::mutex> lock(*block.mutex);
        for (auto &v : block.data) {
            if (v.get_object_id() == object_id) {
                v.reset();
            }
        }
    });
#else
    // Sequential version
    for (auto &[block_key, block] : this->blocks_) {
        for (auto &v : block.data) {
            if (v.get_object_id() == object_id) {
                v.reset();
            }
        }
    }
#endif
}

// remove_low_confidence_segments implementation
template <typename VoxelDataT>
void VoxelBlockSemanticGridT<VoxelDataT>::remove_low_confidence_segments(const int min_confidence) {
#ifdef TBB_FOUND
    // Parallel version
    tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
        auto &block = pair.second;
        std::lock_guard<std::mutex> lock(*block.mutex);
        for (auto &v : block.data) {
            // Use getter method if available (for probabilistic), otherwise direct member
            // access
            if (v.get_confidence() < min_confidence) {
                v.reset();
            }
        }
    });
#else
    // Sequential version
    for (auto &[block_key, block] : this->blocks_) {
        for (auto &v : block.data) {
            // Use getter method if available (for probabilistic), otherwise direct member
            // access
            if (v.get_confidence() < min_confidence) {
                v.reset();
            }
        }
    }
#endif
}

// get_ids implementation
template <typename VoxelDataT>
std::pair<std::vector<int>, std::vector<int>> VoxelBlockSemanticGridT<VoxelDataT>::get_ids() const {
    std::vector<int> class_ids;
    class_ids.reserve(this->get_total_voxel_count());
    std::vector<int> instance_ids;
    instance_ids.reserve(this->get_total_voxel_count());

    for (const auto &[block_key, block] : this->blocks_) {
        for (const auto &v : block.data) {
            if (v.count > 0) {
                class_ids.push_back(v.get_class_id());
                instance_ids.push_back(v.get_object_id());
            }
        }
    }
    return {class_ids, instance_ids};
}

// get_instance_segments implementation
template <typename VoxelDataT>
std::unordered_map<int, std::vector<std::array<double, 3>>>
VoxelBlockSemanticGridT<VoxelDataT>::get_instance_segments() const {
    std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
    for (const auto &[block_key, block] : this->blocks_) {
        for (const auto &v : block.data) {
            if (v.count > 0) {
                segments[v.get_object_id()].push_back(v.get_position());
            }
        }
    }
    return segments;
}

// get_class_segments implementation
template <typename VoxelDataT>
std::unordered_map<int, std::vector<std::array<double, 3>>>
VoxelBlockSemanticGridT<VoxelDataT>::get_class_segments() const {
    std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
    for (const auto &[block_key, block] : this->blocks_) {
        for (const auto &v : block.data) {
            if (v.count > 0) {
                segments[v.get_class_id()].push_back(v.get_position());
            }
        }
    }
    return segments;
}

} // namespace volumetric

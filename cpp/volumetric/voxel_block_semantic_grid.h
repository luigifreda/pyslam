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
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "voxel_block.h"
#include "voxel_block_grid.h"
#include "voxel_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#endif

namespace py = pybind11;

namespace volumetric {

// VoxelBlockSemanticGrid class with indirect voxel hashing (block-based) and semantic segmentation
// The space is divided into blocks of contiguous voxels (NxNxN)
// First, hashing identifies the block, then coordinates are transformed into the final voxel
template <typename VoxelDataT> class VoxelBlockSemanticGridT : public VoxelBlockGridT<VoxelDataT> {

    using Block = VoxelBlockT<VoxelDataT>;

  public:
    explicit VoxelBlockSemanticGridT(double voxel_size = 0.05, int block_size = 8)
        : VoxelBlockGridT<VoxelDataT>(voxel_size, block_size) {
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
    template <typename Tpos, typename Tcolor, typename Tinstance = int, typename Tclass = int,
              typename Tdepth = float>
    void integrate(py::array_t<Tpos> points, py::array_t<Tcolor> colors,
                   py::array_t<Tinstance> instance_ids, py::array_t<Tclass> class_ids,
                   py::array_t<Tdepth> depths) {

        auto pts_info = points.request();
        auto cols_info = colors.request();
        auto instance_ids_info = instance_ids.request();
        auto class_ids_info = class_ids.request();
        auto depths_info = depths.request();

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

        // if depths is not None, check if it has the same number of rows as points
        if (depths_info.ptr) {
            if (pts_info.shape[0] != depths_info.shape[0]) {
                throw std::runtime_error("points and depths must have the same number of rows");
            }
        }

        VoxelBlockGridT<VoxelDataT>::template integrate_raw<Tpos, Tcolor, Tinstance, Tclass>(
            static_cast<const Tpos *>(pts_info.ptr), pts_info.shape[0],
            static_cast<const Tcolor *>(cols_info.ptr),
            static_cast<const Tinstance *>(instance_ids_info.ptr),
            static_cast<const Tclass *>(class_ids_info.ptr),
            static_cast<const Tdepth *>(depths_info.ptr));
    }

    // Insert a segment of points (same instance and class IDs) into the voxel grid
    template <typename Tpos, typename Tcolor, typename Tinstance = int, typename Tclass = int>
    void integrate_segment(py::array_t<Tpos> points, py::array_t<Tcolor> colors,
                           const Tinstance &instance_id, const Tclass &class_id) {
        auto pts_info = points.request();
        auto cols_info = colors.request();
        integrate_segment_raw<Tpos, Tcolor, Tinstance, Tclass>(
            static_cast<const Tpos *>(pts_info.ptr), pts_info.shape[0],
            static_cast<const Tcolor *>(cols_info.ptr), instance_id, class_id);
    }

    // Insert a cluster of points (same instance and class IDs) into the voxel grid
    template <typename Tpos, typename Tcolor = std::nullptr_t, typename Tinstance = int,
              typename Tclass = int>
    void integrate_segment_raw(const Tpos *pts_ptr, const size_t num_points, const Tcolor *cols_ptr,
                               const Tinstance &instance_id, const Tclass &class_id) {
        // Skip points with invalid instance or class IDs
        if (instance_id < 0 || class_id < 0) {
            return;
        }
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

                    if constexpr ((std::is_same_v<Tcolor, std::nullptr_t>)) {
                        this->template update_voxel<Tpos>(x, y, z);
                    } else {
                        const Tcolor color_x = cols_ptr[idx + 0];
                        const Tcolor color_y = cols_ptr[idx + 1];
                        const Tcolor color_z = cols_ptr[idx + 2];

                        if constexpr ((std::is_same_v<Tinstance, std::nullptr_t>) &&
                                      (std::is_same_v<Tclass, std::nullptr_t>)) {
                            this->template update_voxel<Tpos, Tcolor>(x, y, z, color_x, color_y,
                                                                      color_z);
                        } else {
                            // With semantics
                            this->template update_voxel<Tpos, Tcolor, Tinstance, Tclass>(
                                x, y, z, color_x, color_y, color_z, instance_id, class_id);
                        }
                    }
                }
#ifdef TBB_FOUND
            });
#endif
    }

    // Merge two segments of voxels (different instance IDs) into a single segment of voxels with
    // the same instance ID instance_id1
    void merge_segments(const int instance_id1, const int instance_id2) {
#ifdef TBB_FOUND
        // Parallel version using TBB - concurrent_unordered_map is thread-safe
        tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
            auto &block = pair.second;
            std::lock_guard<std::mutex> lock(*block.mutex);
            for (auto &v : block.data) {
                if (v.get_instance_id() == instance_id2) {
                    v.set_instance_id(instance_id1);
                }
            }
        });
#else
        // Sequential version
        for (auto &[block_key, block] : this->blocks_) {
            for (auto &v : block.data) {
                if (v.get_instance_id() == instance_id2) {
                    v.set_instance_id(instance_id1);
                }
            }
        }
#endif
    }

    // Remove all voxels with the specified instance ID
    void remove_segment(const int instance_id) {
#ifdef TBB_FOUND
        // For concurrent_unordered_map, we need to iterate through blocks and mark voxels
        // Since we can't efficiently remove individual voxels from blocks, we mark them as inactive
        tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
            auto &block = pair.second;
            std::lock_guard<std::mutex> lock(*block.mutex);
            for (auto &v : block.data) {
                if (v.get_instance_id() == instance_id) {
                    v.reset();
                }
            }
        });
#else
        // Sequential version
        for (auto &[block_key, block] : this->blocks_) {
            for (auto &v : block.data) {
                if (v.get_instance_id() == instance_id) {
                    v.reset();
                }
            }
        }
#endif
    }

    // Remove all voxels with low confidence counter
    void remove_low_confidence_segments(const int min_confidence_counter) {
#ifdef TBB_FOUND
        // Parallel version
        tbb::parallel_for_each(this->blocks_.begin(), this->blocks_.end(), [&](auto &pair) {
            auto &block = pair.second;
            std::lock_guard<std::mutex> lock(*block.mutex);
            for (auto &v : block.data) {
                // Use getter method if available (for probabilistic), otherwise direct member
                // access
                if (v.get_confidence_counter() < min_confidence_counter) {
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
                if (v.get_confidence_counter() < min_confidence_counter) {
                    v.reset();
                }
            }
        }
#endif
    }

    // Get the class and instance IDs of the voxel grid
    std::pair<std::vector<int>, std::vector<int>> get_ids() const {
        std::vector<int> class_ids;
        class_ids.reserve(this->get_total_voxel_count());
        std::vector<int> instance_ids;
        instance_ids.reserve(this->get_total_voxel_count());

        for (const auto &[block_key, block] : this->blocks_) {
            for (const auto &v : block.data) {
                if (v.count > 0) {
                    class_ids.push_back(v.get_class_id());
                    instance_ids.push_back(v.get_instance_id());
                }
            }
        }
        return {class_ids, instance_ids};
    }

    std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>,
               std::vector<int>, std::vector<int>>
    get_voxel_data(int min_count = 1) const {
        std::vector<std::array<double, 3>> points;
        std::vector<std::array<float, 3>> colors;
        std::vector<int> class_ids;
        std::vector<int> instance_ids;
        const size_t upper_bound_num_voxels = this->num_voxels_per_block_ * this->blocks_.size();
        points.reserve(upper_bound_num_voxels);
        colors.reserve(upper_bound_num_voxels);
        class_ids.reserve(upper_bound_num_voxels);
        instance_ids.reserve(upper_bound_num_voxels);

        for (const auto &[block_key, block] : this->blocks_) {
            for (const auto &v : block.data) {
                if (v.count >= min_count) {
                    points.push_back(v.get_position());
                    colors.push_back(v.get_color());
                    class_ids.push_back(v.get_class_id());
                    instance_ids.push_back(v.get_instance_id());
                }
            }
        }
        return {points, colors, class_ids, instance_ids};
    }

    // Get clusters of voxels based on instance IDs
    std::unordered_map<int, std::vector<std::array<double, 3>>> get_segments() const {
        std::unordered_map<int, std::vector<std::array<double, 3>>> segments;
        for (const auto &[block_key, block] : this->blocks_) {
            for (const auto &v : block.data) {
                if (v.count > 0) {
                    segments[v.get_instance_id()].push_back(v.get_position());
                }
            }
        }
        return segments;
    }
};

using VoxelBlockSemanticGrid = VoxelBlockSemanticGridT<VoxelSemanticData>;
using VoxelBlockSemanticProbabilisticGrid = VoxelBlockSemanticGridT<VoxelSemanticDataProbabilistic>;

} // namespace volumetric

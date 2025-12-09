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

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "voxel_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#endif

namespace volumetric {

// VoxelBlockT structure: 3D array (flattened to 1D vector) for efficient access
// Uses a mutex for thread-safe access when TBB is enabled
template <typename VoxelDataT> struct VoxelBlockT {
    // Flattened 3D array: data[lx + ly * block_size + lz * block_size * block_size]
    std::vector<VoxelDataT> data;
    const int block_size; // number of voxels per side of the block
    const int block_size_squared;

#ifdef TBB_FOUND
    // Mutex for thread-safe access in parallel mode, depending on the used approach.
    // NOTE: When used, it avoids accessing a voxel data from different threads.
    mutable std::unique_ptr<std::mutex> mutex;
#endif

    VoxelBlockT(int size)
        : block_size(size), block_size_squared(size * size), data(size * size * size, VoxelDataT{})
#ifdef TBB_FOUND
          ,
          mutex(std::make_unique<std::mutex>())
#endif
    {
    }

    // Convert 3D local key coordinates to flat index
    size_t get_index(const LocalVoxelKey &local_key) const {
        return static_cast<size_t>(local_key.x + local_key.y * block_size +
                                   local_key.z * block_size_squared);
    }

    // Get voxel data at local coordinates (thread-safe)
    VoxelDataT &get_voxel(const LocalVoxelKey &local_key) { return data[get_index(local_key)]; }

    const VoxelDataT &get_voxel(const LocalVoxelKey &local_key) const {
        return data[get_index(local_key)];
    }

    // Count active voxels (count > 0)
    size_t count_active() const {
        size_t count = 0;
        for (const auto &v : data) {
            if (v.count > 0) {
                ++count;
            }
        }
        return count;
    }
};

} // namespace volumetric
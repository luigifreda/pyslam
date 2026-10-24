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

#include <atomic>
#include <cstdint>

namespace volumetric {

class VoxelSemanticSharedData {
  public:
    inline static std::atomic<int32_t> next_object_id{
        1}; // next object ID (starts at 1, since 0 is reserved for "no specific object")

    VoxelSemanticSharedData() = default;

    static int32_t get_next_object_id() { return next_object_id.fetch_add(1); }
};

} // namespace volumetric
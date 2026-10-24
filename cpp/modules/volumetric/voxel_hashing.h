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
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace volumetric {

// ----------------------------------------
// Voxel hashing
// ----------------------------------------

// For int32_t (range: ±2,147,483,647):
// With voxel_size = 0.015m: world coordinates up to ±32,212km
// With voxel_size = 0.001m (1mm): world coordinates up to ±2,147km
// With voxel_size = 0.0001m (0.1mm): world coordinates up to ±214km
using KeyType = int32_t;

struct VoxelKey {
    KeyType x, y, z;

    VoxelKey() = default;
    VoxelKey(KeyType x, KeyType y, KeyType z) : x(x), y(y), z(z) {}

    bool operator==(const VoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey &key) const {
        const std::size_t h1 = std::hash<KeyType>{}(key.x);
        const std::size_t h2 = std::hash<KeyType>{}(key.y);
        const std::size_t h3 = std::hash<KeyType>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

template <typename Tp, typename Tv>
inline VoxelKey get_voxel_key(const Tp x, const Tp y, const Tp z, const Tv voxel_size) {
    const KeyType vx = static_cast<KeyType>(std::floor(x / voxel_size));
    const KeyType vy = static_cast<KeyType>(std::floor(y / voxel_size));
    const KeyType vz = static_cast<KeyType>(std::floor(z / voxel_size));
    return VoxelKey(vx, vy, vz);
}

// Optimized version that uses multiplication instead of division (for inverse voxel size)
template <typename Tp, typename Tv>
inline VoxelKey get_voxel_key_inv(const Tp x, const Tp y, const Tp z, const Tv inv_voxel_size) {
    const KeyType vx = static_cast<KeyType>(std::floor(x * inv_voxel_size));
    const KeyType vy = static_cast<KeyType>(std::floor(y * inv_voxel_size));
    const KeyType vz = static_cast<KeyType>(std::floor(z * inv_voxel_size));
    return VoxelKey(vx, vy, vz);
}

// ----------------------------------------
// Block hashing
// ----------------------------------------

// Block key: identifies a block in the grid (for indirect voxel hashing)
struct BlockKey {
    KeyType x, y, z;

    BlockKey() = default;
    BlockKey(KeyType x, KeyType y, KeyType z) : x(x), y(y), z(z) {}

    bool operator==(const BlockKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Local voxel key: identifies a voxel within a block (0 to block_size-1)
struct LocalVoxelKey {
    KeyType x, y, z;

    LocalVoxelKey() = default;
    LocalVoxelKey(KeyType x, KeyType y, KeyType z) : x(x), y(y), z(z) {}

    bool operator==(const LocalVoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Hash function for BlockKey
struct BlockKeyHash {
    std::size_t operator()(const BlockKey &key) const {
        const std::size_t h1 = std::hash<KeyType>{}(key.x);
        const std::size_t h2 = std::hash<KeyType>{}(key.y);
        const std::size_t h3 = std::hash<KeyType>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Hash function for LocalVoxelKey
struct LocalVoxelKeyHash {
    std::size_t operator()(const LocalVoxelKey &key) const {
        // Since local coordinates are in range [0, block_size), we can use a simple hash
        const std::size_t h1 = std::hash<KeyType>{}(key.x);
        const std::size_t h2 = std::hash<KeyType>{}(key.y);
        const std::size_t h3 = std::hash<KeyType>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Helper function for floor division (handles negative numbers correctly)
// The floor_div function ensures integer division rounds toward negative infinity (floor), not
// toward zero (which is the default behavior of integer division in C++).
// World voxels (x):
// ...  -9  -8  -7  -6  -5  -4  -3  -2  -1  |  0   1   2   3   4   5   6   7   8   9  ...
// Block size B = 4

// Block index b_x = floor(x / B):
// ...  -3  -3  -2  -2  -2  -2  -1  -1  -1  |  0   0   0   0   1   1   1   1   2   2  ...
// Local index l_x = x - b_x * B:
// ...   3   0   3   2   1   0   3   2   1  |  0   1   2   3   0   1   2   3   0   1  ...
//            Block -2        Block -1        Block 0         Block 1         Block 2
//            l_x:0..3        l_x:0..3        l_x:0..3        l_x:0..3        l_x:0..3
inline int64_t floor_div(const int64_t a, const int64_t b) {
    // assert(b>0)
    return (a >= 0) ? (a / b) : ((a - b + 1) / b);
}

// Compute block coordinates from voxel coordinates
inline BlockKey get_block_key(const VoxelKey &voxel_key, const size_t block_size) {
    const int64_t bs = static_cast<int64_t>(block_size);
    const KeyType bx = static_cast<KeyType>(floor_div(voxel_key.x, bs));
    const KeyType by = static_cast<KeyType>(floor_div(voxel_key.y, bs));
    const KeyType bz = static_cast<KeyType>(floor_div(voxel_key.z, bs));
    return BlockKey(bx, by, bz);
}

// Compute local voxel coordinates within a block from voxel and block coordinates
inline LocalVoxelKey get_local_voxel_key(const VoxelKey &voxel_key, const BlockKey &block_key,
                                         const size_t block_size) {
    const int64_t bs = static_cast<int64_t>(block_size);
    const KeyType lx = static_cast<KeyType>(int64_t(voxel_key.x) - int64_t(block_key.x) * bs);
    const KeyType ly = static_cast<KeyType>(int64_t(voxel_key.y) - int64_t(block_key.y) * bs);
    const KeyType lz = static_cast<KeyType>(int64_t(voxel_key.z) - int64_t(block_key.z) * bs);
    return LocalVoxelKey(lx, ly, lz);
}

// Tuple-like get functions in volumetric namespace for ADL support in structured bindings
template <size_t I> constexpr KeyType get(const VoxelKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}

template <size_t I> constexpr KeyType get(const BlockKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}

template <size_t I> constexpr KeyType get(const LocalVoxelKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}

} // namespace volumetric

// Tuple-like support for structured bindings (must be in std namespace)
namespace std {
template <> struct tuple_size<volumetric::VoxelKey> : integral_constant<size_t, 3> {};

template <size_t I> struct tuple_element<I, volumetric::VoxelKey> {
    using type = volumetric::KeyType;
};

template <size_t I> constexpr volumetric::KeyType get(const volumetric::VoxelKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}

template <> struct tuple_size<volumetric::BlockKey> : integral_constant<size_t, 3> {};

template <size_t I> struct tuple_element<I, volumetric::BlockKey> {
    using type = volumetric::KeyType;
};

template <size_t I> constexpr volumetric::KeyType get(const volumetric::BlockKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}

template <> struct tuple_size<volumetric::LocalVoxelKey> : integral_constant<size_t, 3> {};

template <size_t I> struct tuple_element<I, volumetric::LocalVoxelKey> {
    using type = volumetric::KeyType;
};

template <size_t I> constexpr volumetric::KeyType get(const volumetric::LocalVoxelKey &key) {
    if constexpr (I == 0)
        return key.x;
    if constexpr (I == 1)
        return key.y;
    if constexpr (I == 2)
        return key.z;
}
} // namespace std
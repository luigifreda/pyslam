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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>

#include <atomic>

// NOTE: This file contains the definitions of the voxel data types and the concepts that they must
// satisfy.

namespace volumetric {

#ifndef NDEBUG
constexpr bool kCheckZeroCount = true; // Enable the check for zero count (debug)
                                       // In theory, it is not needed since we only process voxels
                                       // where  the count is always > 0
#else
constexpr bool kCheckZeroCount = false; // Disable the check for zero count (release)
#endif

// Macro for position-related members
#define VOXEL_POSITION_MEMBERS()                                                                   \
    std::array<double, 3> position_sum = {0.0, 0.0, 0.0}; // sum of positions

// Macro for position-related methods
#define VOXEL_POSITION_METHODS()                                                                   \
    template <typename Tpos> void update_point(const Tpos x, const Tpos y, const Tpos z) {         \
        position_sum[0] += static_cast<double>(x);                                                 \
        position_sum[1] += static_cast<double>(y);                                                 \
        position_sum[2] += static_cast<double>(z);                                                 \
    }                                                                                              \
    std::array<double, 3> get_position() const {                                                   \
        if constexpr (kCheckZeroCount) {                                                           \
            if (count == 0) {                                                                      \
                return {0.0, 0.0, 0.0};                                                            \
            }                                                                                      \
        }                                                                                          \
        const double count_d = static_cast<double>(count);                                         \
        std::array<double, 3> avg_coord = {position_sum[0] / count_d, position_sum[1] / count_d,   \
                                           position_sum[2] / count_d};                             \
        return avg_coord;                                                                          \
    }

// Macro for color-related members
#define VOXEL_COLOR_MEMBERS() std::array<float, 3> color_sum = {0.0f, 0.0f, 0.0f}; // sum of colors

// Macro for color-related methods
#define VOXEL_COLOR_METHODS()                                                                      \
    template <typename Tcolor>                                                                     \
    void update_color(const Tcolor color_x, const Tcolor color_y, const Tcolor color_z) {          \
        if constexpr (std::is_same_v<Tcolor, uint8_t>) {                                           \
            color_sum[0] += static_cast<float>(color_x) / 255.0f;                                  \
            color_sum[1] += static_cast<float>(color_y) / 255.0f;                                  \
            color_sum[2] += static_cast<float>(color_z) / 255.0f;                                  \
        } else if constexpr (std::is_same_v<Tcolor, float>) {                                      \
            color_sum[0] += static_cast<float>(color_x);                                           \
            color_sum[1] += static_cast<float>(color_y);                                           \
            color_sum[2] += static_cast<float>(color_z);                                           \
        } else if constexpr (std::is_same_v<Tcolor, double>) {                                     \
            color_sum[0] += static_cast<float>(color_x);                                           \
            color_sum[1] += static_cast<float>(color_y);                                           \
            color_sum[2] += static_cast<float>(color_z);                                           \
        } else {                                                                                   \
            static_assert(!std::is_same_v<Tcolor, uint8_t> && !std::is_same_v<Tcolor, float> &&    \
                              !std::is_same_v<Tcolor, double>,                                     \
                          "Unsupported color type");                                               \
        }                                                                                          \
    }                                                                                              \
    std::array<float, 3> get_color() const {                                                       \
        if constexpr (kCheckZeroCount) {                                                           \
            if (count == 0) {                                                                      \
                return {0.0f, 0.0f, 0.0f};                                                         \
            }                                                                                      \
        }                                                                                          \
        const float count_f = static_cast<float>(count);                                           \
        std::array<float, 3> avg_color = {color_sum[0] / count_f, color_sum[1] / count_f,          \
                                          color_sum[2] / count_f};                                 \
        return avg_color;                                                                          \
    }

//=================================================================================================
// Voxel Data
//=================================================================================================

// Simple voxel data structure for just storing and managing points and colors.
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
struct VoxelData {
    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
    }
};

//=================================================================================================
// Voxel Concepts
//=================================================================================================

// Helper concept to check if update_color works with a specific color type
template <typename T, typename ColorT>
concept HasUpdateColor =
    requires(T v, ColorT c1, ColorT c2, ColorT c3) { v.update_color(c1, c2, c3); };

// Concept for basic voxel data types
// Checks that a type has all the required members and methods for basic voxel functionality
// Note: update_color is checked to work with at least one of the supported types (uint8_t, float,
// or double) to support template-based implementations that accept different color types
template <typename T>
concept Voxel = requires(T v, double x, double y, double z) {
    // Basic voxel members (check they exist and are accessible)
    v.count;
    v.position_sum;
    v.color_sum;

    // Basic voxel methods
    v.update_point(x, y, z);
    { v.get_position() };
    { v.get_color() };
    v.reset();
} && (HasUpdateColor<T, float> || HasUpdateColor<T, uint8_t> || HasUpdateColor<T, double>);

// Static assertions to verify the concepts work with the actual types
static_assert(Voxel<VoxelData>, "VoxelData must satisfy the Voxel concept");

} // namespace volumetric

// #undef VOXEL_POSITION_MEMBERS
// #undef VOXEL_POSITION_METHODS
// #undef VOXEL_COLOR_MEMBERS
// #undef VOXEL_COLOR_METHODS

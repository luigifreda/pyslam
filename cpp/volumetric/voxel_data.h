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
#include <type_traits>

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

#define VOXEL_DATA_USING_TYPE(Tpos, Tcolor)                                                        \
    using PosScalar = Tpos;                                                                        \
    using ColorScalar = Tcolor;                                                                    \
    using Pos3 = std::array<PosScalar, 3>;                                                         \
    using Color3 = std::array<ColorScalar, 3>;

// Macro for position-related members
#define VOXEL_POSITION_MEMBERS(Tpos)                                                               \
    std::array<Tpos, 3> position_sum = {static_cast<Tpos>(0), static_cast<Tpos>(0),                \
                                        static_cast<Tpos>(0)}; // sum of positions

// Macro for position-related methods
#define VOXEL_POSITION_METHODS(Tpos)                                                               \
    template <typename PosScalarIn>                                                                \
    void update_point(const PosScalarIn x, const PosScalarIn y, const PosScalarIn z) {             \
        position_sum[0] += static_cast<Tpos>(x);                                                   \
        position_sum[1] += static_cast<Tpos>(y);                                                   \
        position_sum[2] += static_cast<Tpos>(z);                                                   \
    }                                                                                              \
    std::array<Tpos, 3> get_position() const {                                                     \
        if constexpr (kCheckZeroCount) {                                                           \
            if (count == 0) {                                                                      \
                return {static_cast<Tpos>(0), static_cast<Tpos>(0), static_cast<Tpos>(0)};         \
            }                                                                                      \
        }                                                                                          \
        const Tpos count_casted = static_cast<Tpos>(count);                                        \
        std::array<Tpos, 3> avg_coord = {position_sum[0] / count_casted,                           \
                                         position_sum[1] / count_casted,                           \
                                         position_sum[2] / count_casted};                          \
        return avg_coord;                                                                          \
    }

// Macro for color-related members
#define VOXEL_COLOR_MEMBERS(Tcolor)                                                                \
    std::array<Tcolor, 3> color_sum = {static_cast<Tcolor>(0), static_cast<Tcolor>(0),             \
                                       static_cast<Tcolor>(0)}; // sum of colors

// Macro for color-related methods
#define VOXEL_COLOR_METHODS(Tcolor)                                                                \
    template <typename ColorScalarIn>                                                              \
    void update_color(const ColorScalarIn color_x, const ColorScalarIn color_y,                    \
                      const ColorScalarIn color_z) {                                               \
        if constexpr (std::is_same_v<ColorScalarIn, uint8_t>) {                                    \
            constexpr Tcolor inv_255 = static_cast<Tcolor>(1.0) / static_cast<Tcolor>(255.0);      \
            color_sum[0] += static_cast<Tcolor>(color_x) * inv_255;                                \
            color_sum[1] += static_cast<Tcolor>(color_y) * inv_255;                                \
            color_sum[2] += static_cast<Tcolor>(color_z) * inv_255;                                \
        } else if constexpr (std::is_same_v<ColorScalarIn, double> ||                              \
                             std::is_same_v<ColorScalarIn, float>) {                               \
            color_sum[0] += static_cast<Tcolor>(color_x);                                          \
            color_sum[1] += static_cast<Tcolor>(color_y);                                          \
            color_sum[2] += static_cast<Tcolor>(color_z);                                          \
        } else {                                                                                   \
            static_assert(!std::is_same_v<ColorScalarIn, uint8_t> &&                               \
                              !std::is_same_v<ColorScalarIn, float> &&                             \
                              !std::is_same_v<ColorScalarIn, double>,                              \
                          "Unsupported color type");                                               \
        }                                                                                          \
    }                                                                                              \
    std::array<Tcolor, 3> get_color() const {                                                      \
        if constexpr (kCheckZeroCount) {                                                           \
            if (count == 0) {                                                                      \
                return {static_cast<Tcolor>(0), static_cast<Tcolor>(0), static_cast<Tcolor>(0)};   \
            }                                                                                      \
        }                                                                                          \
        const Tcolor count_casted = static_cast<Tcolor>(count);                                    \
        std::array<Tcolor, 3> avg_color = {color_sum[0] / count_casted,                            \
                                           color_sum[1] / count_casted,                            \
                                           color_sum[2] / count_casted};                           \
        return avg_color;                                                                          \
    }

//=================================================================================================
// Voxel Data
//=================================================================================================

// Simple voxel data structure for just storing and managing points and colors.
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
template <typename Tpos, typename Tcolor = float> struct VoxelDataT {
    VOXEL_DATA_USING_TYPE(Tpos, Tcolor)

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS(Tpos)
    VOXEL_COLOR_MEMBERS(Tcolor)

    VOXEL_POSITION_METHODS(Tpos)
    VOXEL_COLOR_METHODS(Tcolor)

    void reset() {
        count = 0;
        position_sum = {static_cast<Tpos>(0), static_cast<Tpos>(0), static_cast<Tpos>(0)};
        color_sum = {static_cast<Tcolor>(0), static_cast<Tcolor>(0), static_cast<Tcolor>(0)};
    }
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using VoxelData = VoxelDataT<float, float>;
using VoxelDataD = VoxelDataT<double, float>;
using VoxelDataF = VoxelDataT<float, float>;

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

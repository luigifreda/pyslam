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

// This file contains SIMD-optimized implementations for VoxelGridT
// It should be included inline within the VoxelGridT class definition

#if USE_SIMD

namespace volumetric {

// Point information structure for grouping points by voxel key
// CoordT is the coordinate type (float or double)
template <typename CoordT> struct PointInfo {
    CoordT x, y, z;
    float color_x, color_y, color_z;
};

// Unified SIMD-optimized version (processes 4 points at a time)
// Uses pre-partitioning to avoid race conditions when called from parallel code.
// Groups points by voxel key first, then processes each voxel's points in SIMD batches.
// Template parameter HasColors controls whether color processing is enabled at compile time
template <typename VoxelDataT>
template <bool HasColors>
void VoxelGridT<VoxelDataT>::integrate_raw_simd_impl(const float *pts_ptr, const float *cols_ptr,
                                                     size_t num_points) {
#ifdef TBB_FOUND
    // Use enumerable_thread_specific to collect thread-local maps without contention
    // Each thread builds its own map independently, then we merge sequentially after parallel phase
    using LocalGroupsMap =
        std::unordered_map<VoxelKey, std::vector<PointInfo<float>>, VoxelKeyHash>;
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
            const float x = pts_ptr[idx + 0];
            const float y = pts_ptr[idx + 1];
            const float z = pts_ptr[idx + 2];

            const VoxelKey key = get_voxel_key_inv<float, float>(x, y, z, inv_voxel_size_);

            PointInfo<float> info{x, y, z};
            if constexpr (HasColors) {
                info.color_x = cols_ptr[idx + 0];
                info.color_y = cols_ptr[idx + 1];
                info.color_z = cols_ptr[idx + 2];
            }

            local_groups[key].push_back(info);
        }
    });

    // Merge thread-local maps into final result sequentially (no contention)
    // Use std::unordered_map since merge is sequential and subsequent parallel processing is
    // read-only
    std::unordered_map<VoxelKey, std::vector<PointInfo<float>>, VoxelKeyHash> voxel_groups;

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
                std::vector<PointInfo<float>> vec;
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
    // Use SIMD batching within each voxel's points
    tbb::parallel_for_each(voxel_groups.begin(), voxel_groups.end(), [&](const auto &pair) {
        const VoxelKey &key = pair.first;
        const std::vector<PointInfo<float>> &points = pair.second;

        // Get or create voxel (concurrent map is thread-safe)
        auto [it, inserted] = grid_.insert({key, VoxelDataT()});
        auto &v = it->second;

        // Process points for this voxel in SIMD batches
        const size_t simd_width = 4;
        const size_t num_points_voxel = points.size();
        const size_t simd_end_voxel = (num_points_voxel / simd_width) * simd_width;

        const __m128 inv_voxel_size_vec = _mm_set1_ps(inv_voxel_size_);

        // Process SIMD batches
        for (size_t batch_start = 0; batch_start < simd_end_voxel; batch_start += simd_width) {
            alignas(16) float x_vals[4], y_vals[4], z_vals[4];
            alignas(16) float color_x_vals[4], color_y_vals[4], color_z_vals[4];

            // Load batch
            for (size_t j = 0; j < simd_width; ++j) {
                const auto &info = points[batch_start + j];
                x_vals[j] = info.x;
                y_vals[j] = info.y;
                z_vals[j] = info.z;
                if constexpr (HasColors) {
                    color_x_vals[j] = info.color_x;
                    color_y_vals[j] = info.color_y;
                    color_z_vals[j] = info.color_z;
                }
            }

            // Accumulate batch update
            struct VoxelUpdate {
                double pos_sum[3] = {0.0, 0.0, 0.0};
                float col_sum[3] = {0.0f, 0.0f, 0.0f};
                int count = 0;
            };
            VoxelUpdate batch_update;

            for (size_t j = 0; j < simd_width; ++j) {
                batch_update.pos_sum[0] += static_cast<double>(x_vals[j]);
                batch_update.pos_sum[1] += static_cast<double>(y_vals[j]);
                batch_update.pos_sum[2] += static_cast<double>(z_vals[j]);
                if constexpr (HasColors) {
                    batch_update.col_sum[0] += color_x_vals[j];
                    batch_update.col_sum[1] += color_y_vals[j];
                    batch_update.col_sum[2] += color_z_vals[j];
                }
                batch_update.count++;
            }

            // Apply batch update to voxel
            if (inserted || v.count == 0) {
                v.position_sum[0] = batch_update.pos_sum[0];
                v.position_sum[1] = batch_update.pos_sum[1];
                v.position_sum[2] = batch_update.pos_sum[2];
                if constexpr (HasColors) {
                    v.color_sum[0] = batch_update.col_sum[0];
                    v.color_sum[1] = batch_update.col_sum[1];
                    v.color_sum[2] = batch_update.col_sum[2];
                }
                v.count = batch_update.count;
                inserted = false;
            } else {
                // Use SIMD for color addition
                v.position_sum[0] += batch_update.pos_sum[0];
                v.position_sum[1] += batch_update.pos_sum[1];
                v.position_sum[2] += batch_update.pos_sum[2];

                if constexpr (HasColors) {
                    __m128 v_col_vec =
                        _mm_set_ps(0.0f, v.color_sum[2], v.color_sum[1], v.color_sum[0]);
                    __m128 u_col_vec = _mm_set_ps(0.0f, batch_update.col_sum[2],
                                                  batch_update.col_sum[1], batch_update.col_sum[0]);
                    __m128 result = _mm_add_ps(v_col_vec, u_col_vec);
                    alignas(16) float result_arr[4];
                    _mm_store_ps(result_arr, result);
                    v.color_sum[0] = result_arr[0];
                    v.color_sum[1] = result_arr[1];
                    v.color_sum[2] = result_arr[2];
                }
                v.count += batch_update.count;
            }
        }

        // Process remaining points for this voxel
        for (size_t i = simd_end_voxel; i < num_points_voxel; ++i) {
            const auto &info = points[i];
            if (v.count == 0) {
                v.update_point(info.x, info.y, info.z);
                if constexpr (HasColors) {
                    v.update_color(info.color_x, info.color_y, info.color_z);
                }
                v.count = 1;
            } else {
                v.update_point(info.x, info.y, info.z);
                if constexpr (HasColors) {
                    v.update_color(info.color_x, info.color_y, info.color_z);
                }
                ++v.count;
            }
        }
    });
#else
    // Sequential version: original SIMD code without pre-partitioning
    const size_t simd_width = 4;
    const size_t simd_end = (num_points / simd_width) * simd_width;

    // Broadcast inv_voxel_size_ to SIMD register
    const __m128 inv_voxel_size_vec = _mm_set1_ps(inv_voxel_size_);

    // Process 4 points at a time
    alignas(16) float x_vals[4], y_vals[4], z_vals[4];
    alignas(16) float color_x_vals[4], color_y_vals[4],
        color_z_vals[4]; // Only used if HasColors

    for (size_t i = 0; i < simd_end; i += simd_width) {

        // Load coordinates
        x_vals[0] = pts_ptr[(i + 0) * 3 + 0];
        x_vals[1] = pts_ptr[(i + 1) * 3 + 0];
        x_vals[2] = pts_ptr[(i + 2) * 3 + 0];
        x_vals[3] = pts_ptr[(i + 3) * 3 + 0];
        y_vals[0] = pts_ptr[(i + 0) * 3 + 1];
        y_vals[1] = pts_ptr[(i + 1) * 3 + 1];
        y_vals[2] = pts_ptr[(i + 2) * 3 + 1];
        y_vals[3] = pts_ptr[(i + 3) * 3 + 1];
        z_vals[0] = pts_ptr[(i + 0) * 3 + 2];
        z_vals[1] = pts_ptr[(i + 1) * 3 + 2];
        z_vals[2] = pts_ptr[(i + 2) * 3 + 2];
        z_vals[3] = pts_ptr[(i + 3) * 3 + 2];

        // Load colors (only if HasColors is true)
        if constexpr (HasColors) {
            color_x_vals[0] = cols_ptr[(i + 0) * 3 + 0];
            color_x_vals[1] = cols_ptr[(i + 1) * 3 + 0];
            color_x_vals[2] = cols_ptr[(i + 2) * 3 + 0];
            color_x_vals[3] = cols_ptr[(i + 3) * 3 + 0];
            color_y_vals[0] = cols_ptr[(i + 0) * 3 + 1];
            color_y_vals[1] = cols_ptr[(i + 1) * 3 + 1];
            color_y_vals[2] = cols_ptr[(i + 2) * 3 + 1];
            color_y_vals[3] = cols_ptr[(i + 3) * 3 + 1];
            color_z_vals[0] = cols_ptr[(i + 0) * 3 + 2];
            color_z_vals[1] = cols_ptr[(i + 1) * 3 + 2];
            color_z_vals[2] = cols_ptr[(i + 2) * 3 + 2];
            color_z_vals[3] = cols_ptr[(i + 3) * 3 + 2];
        }

        // Compute voxel keys using SIMD
        __m128 x_simd = _mm_load_ps(x_vals);
        __m128 y_simd = _mm_load_ps(y_vals);
        __m128 z_simd = _mm_load_ps(z_vals);

        // Multiply by inv_voxel_size and floor
        __m128 x_scaled = _mm_mul_ps(x_simd, inv_voxel_size_vec);
        __m128 y_scaled = _mm_mul_ps(y_simd, inv_voxel_size_vec);
        __m128 z_scaled = _mm_mul_ps(z_simd, inv_voxel_size_vec);

        // Convert to int32
        alignas(16) int32_t x_int[4], y_int[4], z_int[4];
        x_scaled = _mm_floor_ps(x_scaled);
        y_scaled = _mm_floor_ps(y_scaled);
        z_scaled = _mm_floor_ps(z_scaled);

        _mm_store_si128((__m128i *)x_int, _mm_cvtps_epi32(x_scaled));
        _mm_store_si128((__m128i *)y_int, _mm_cvtps_epi32(y_scaled));
        _mm_store_si128((__m128i *)z_int, _mm_cvtps_epi32(z_scaled));

        // Group points by voxel key to batch updates
        // Use a small local map to accumulate updates for the same voxel
        struct VoxelUpdate {
            double pos_sum[3] = {0.0, 0.0, 0.0};
            float col_sum[3] = {0.0f, 0.0f, 0.0f}; // Only used if HasColors
            int count = 0;
        };
        std::unordered_map<VoxelKey, VoxelUpdate, VoxelKeyHash> batch_updates;
        batch_updates.reserve(simd_width);

        // Accumulate updates for points mapping to the same voxel
        for (size_t j = 0; j < simd_width; ++j) {
            const VoxelKey key{x_int[j], y_int[j], z_int[j]};
            auto &update = batch_updates[key];
            update.pos_sum[0] += static_cast<double>(x_vals[j]);
            update.pos_sum[1] += static_cast<double>(y_vals[j]);
            update.pos_sum[2] += static_cast<double>(z_vals[j]);
            if constexpr (HasColors) {
                update.col_sum[0] += color_x_vals[j];
                update.col_sum[1] += color_y_vals[j];
                update.col_sum[2] += color_z_vals[j];
            }
            update.count++;
        }

        // Apply batched updates using SIMD where possible
        for (const auto &[key, update] : batch_updates) {
#ifdef TBB_FOUND
            auto [it, inserted] = grid_.insert({key, VoxelDataT()});
#else
            auto [it, inserted] = grid_.try_emplace(key);
#endif
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize with accumulated values
                // If count==0, the voxel was reset and should be treated as new
                v.position_sum[0] = update.pos_sum[0];
                v.position_sum[1] = update.pos_sum[1];
                v.position_sum[2] = update.pos_sum[2];
                if constexpr (HasColors) {
                    v.color_sum[0] = update.col_sum[0];
                    v.color_sum[1] = update.col_sum[1];
                    v.color_sum[2] = update.col_sum[2];
                }
                v.count = update.count;
            } else {
                // Existing voxel: add accumulated values
                // Position (double) - use scalar for now (AVX2 double SIMD would require
                // alignment)
                v.position_sum[0] += update.pos_sum[0];
                v.position_sum[1] += update.pos_sum[1];
                v.position_sum[2] += update.pos_sum[2];

                if constexpr (HasColors) {
                    // Color (float) - use SIMD for addition
                    // _mm_set_ps takes arguments in reverse order (w, z, y, x)
                    __m128 v_col_vec =
                        _mm_set_ps(0.0f, v.color_sum[2], v.color_sum[1], v.color_sum[0]);
                    __m128 u_col_vec =
                        _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1], update.col_sum[0]);
                    __m128 result = _mm_add_ps(v_col_vec, u_col_vec);

                    alignas(16) float result_arr[4];
                    _mm_store_ps(result_arr, result);
                    v.color_sum[0] = result_arr[0];
                    v.color_sum[1] = result_arr[1];
                    v.color_sum[2] = result_arr[2];
                }

                v.count += update.count;
            }
        }
    }

    // Process remaining points with scalar code
    for (size_t i = simd_end; i < num_points; ++i) {
        const size_t idx = i * 3;
        const float x = pts_ptr[idx + 0];
        const float y = pts_ptr[idx + 1];
        const float z = pts_ptr[idx + 2];

        const VoxelKey key = get_voxel_key_inv<float, float>(x, y, z, inv_voxel_size_);

        auto [it, inserted] = grid_.try_emplace(key);
        auto &v = it->second;

        if (inserted || v.count == 0) {
            // New voxel or reset voxel: initialize and update
            // If count==0, the voxel was reset and should be treated as new
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            ++v.count;
        }
    }
#endif // TBB_FOUND
}

#endif

#if USE_DOUBLE_SIMD

// Unified SIMD-optimized version for double precision points
// Uses pre-partitioning to avoid race conditions when called from parallel code.
// Groups points by voxel key first, then processes each voxel's points in SIMD batches.
// Template parameter HasColors controls whether color processing is enabled at compile time
template <typename VoxelDataT>
template <bool HasColors>
void VoxelGridT<VoxelDataT>::integrate_raw_simd_impl_double(const double *pts_ptr,
                                                            const float *cols_ptr,
                                                            size_t num_points) {
#ifdef TBB_FOUND
    // Use enumerable_thread_specific to collect thread-local maps without contention
    // Each thread builds its own map independently, then we merge sequentially after parallel phase
    using LocalGroupsMap =
        std::unordered_map<VoxelKey, std::vector<PointInfo<double>>, VoxelKeyHash>;
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
            const double x = pts_ptr[idx + 0];
            const double y = pts_ptr[idx + 1];
            const double z = pts_ptr[idx + 2];

            const VoxelKey key = get_voxel_key_inv<double, double>(x, y, z, inv_voxel_size_);

            PointInfo<double> info{x, y, z};
            if constexpr (HasColors) {
                info.color_x = cols_ptr[idx + 0];
                info.color_y = cols_ptr[idx + 1];
                info.color_z = cols_ptr[idx + 2];
            }

            local_groups[key].push_back(info);
        }
    });

    // Merge thread-local maps into final result sequentially (no contention)
    // Use std::unordered_map since merge is sequential and subsequent parallel processing is
    // read-only
    std::unordered_map<VoxelKey, std::vector<PointInfo<double>>, VoxelKeyHash> voxel_groups;

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
                std::vector<PointInfo<double>> vec;
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
    // Use SIMD batching within each voxel's points
    tbb::parallel_for_each(voxel_groups.begin(), voxel_groups.end(), [&](const auto &pair) {
        const VoxelKey &key = pair.first;
        const std::vector<PointInfo<double>> &points = pair.second;

        auto [it, inserted] = grid_.insert({key, VoxelDataT()});
        auto &v = it->second;

        // Process points for this voxel in SIMD batches
#if USE_AVX2
        const size_t simd_width = 2; // AVX2: 2 points at a time
        const double inv_voxel_size_d = static_cast<double>(inv_voxel_size_);
        const __m256d inv_voxel_size_vec = _mm256_set1_pd(inv_voxel_size_d);
#else
                              const size_t simd_width = 2; // SSE: 2 points at a time
                              const double inv_voxel_size_d = static_cast<double>(inv_voxel_size_);
                              const __m128d inv_voxel_size_vec = _mm_set1_pd(inv_voxel_size_d);
#endif
        const size_t num_points_voxel = points.size();
        const size_t simd_end_voxel = (num_points_voxel / simd_width) * simd_width;

        // Process SIMD batches
        for (size_t batch_start = 0; batch_start < simd_end_voxel; batch_start += simd_width) {
            alignas(32) double x_vals[2], y_vals[2], z_vals[2];
            alignas(16) float color_x_vals[2], color_y_vals[2], color_z_vals[2];

            // Load batch
            for (size_t j = 0; j < simd_width; ++j) {
                const auto &info = points[batch_start + j];
                x_vals[j] = info.x;
                y_vals[j] = info.y;
                z_vals[j] = info.z;
                if constexpr (HasColors) {
                    color_x_vals[j] = info.color_x;
                    color_y_vals[j] = info.color_y;
                    color_z_vals[j] = info.color_z;
                }
            }

            // Accumulate batch update
            struct VoxelUpdate {
                double pos_sum[3] = {0.0, 0.0, 0.0};
                float col_sum[3] = {0.0f, 0.0f, 0.0f};
                int count = 0;
            };
            VoxelUpdate batch_update;

            for (size_t j = 0; j < simd_width; ++j) {
                batch_update.pos_sum[0] += x_vals[j];
                batch_update.pos_sum[1] += y_vals[j];
                batch_update.pos_sum[2] += z_vals[j];
                if constexpr (HasColors) {
                    batch_update.col_sum[0] += color_x_vals[j];
                    batch_update.col_sum[1] += color_y_vals[j];
                    batch_update.col_sum[2] += color_z_vals[j];
                }
                batch_update.count++;
            }

            // Apply batch update to voxel
            if (inserted || v.count == 0) {
                v.position_sum[0] = batch_update.pos_sum[0];
                v.position_sum[1] = batch_update.pos_sum[1];
                v.position_sum[2] = batch_update.pos_sum[2];
                if constexpr (HasColors) {
                    v.color_sum[0] = batch_update.col_sum[0];
                    v.color_sum[1] = batch_update.col_sum[1];
                    v.color_sum[2] = batch_update.col_sum[2];
                }
                v.count = batch_update.count;
                inserted = false;
            } else {
#if USE_AVX2
                // Use AVX2 for double addition
                __m256d v_pos_vec =
                    _mm256_set_pd(0.0, v.position_sum[2], v.position_sum[1], v.position_sum[0]);
                __m256d u_pos_vec = _mm256_set_pd(0.0, batch_update.pos_sum[2],
                                                  batch_update.pos_sum[1], batch_update.pos_sum[0]);
                __m256d result_pos = _mm256_add_pd(v_pos_vec, u_pos_vec);
                alignas(32) double result_pos_arr[4];
                _mm256_store_pd(result_pos_arr, result_pos);
                v.position_sum[0] = result_pos_arr[0];
                v.position_sum[1] = result_pos_arr[1];
                v.position_sum[2] = result_pos_arr[2];
#else
                                      // Use SSE2 for double addition
                                      __m128d v_pos_01 = _mm_loadu_pd(&v.position_sum[0]);
                                      __m128d u_pos_01 = _mm_loadu_pd(&batch_update.pos_sum[0]);
                                      __m128d result_01 = _mm_add_pd(v_pos_01, u_pos_01);
                                      _mm_storeu_pd(&v.position_sum[0], result_01);
                                      v.position_sum[2] += batch_update.pos_sum[2];
#endif
                if constexpr (HasColors) {
                    __m128 v_col_vec =
                        _mm_set_ps(0.0f, v.color_sum[2], v.color_sum[1], v.color_sum[0]);
                    __m128 u_col_vec = _mm_set_ps(0.0f, batch_update.col_sum[2],
                                                  batch_update.col_sum[1], batch_update.col_sum[0]);
                    __m128 result_col = _mm_add_ps(v_col_vec, u_col_vec);
                    alignas(16) float result_col_arr[4];
                    _mm_store_ps(result_col_arr, result_col);
                    v.color_sum[0] = result_col_arr[0];
                    v.color_sum[1] = result_col_arr[1];
                    v.color_sum[2] = result_col_arr[2];
                }
                v.count += batch_update.count;
            }
        }

        // Process remaining points for this voxel
        for (size_t i = simd_end_voxel; i < num_points_voxel; ++i) {
            const auto &info = points[i];
            if (v.count == 0) {
                v.update_point(info.x, info.y, info.z);
                if constexpr (HasColors) {
                    v.update_color(info.color_x, info.color_y, info.color_z);
                }
                v.count = 1;
            } else {
                v.update_point(info.x, info.y, info.z);
                if constexpr (HasColors) {
                    v.update_color(info.color_x, info.color_y, info.color_z);
                }
                ++v.count;
            }
        }
    });
#else
    // Sequential version: original SIMD code without pre-partitioning
#if USE_AVX2
    // AVX2: process 4 doubles at a time (2 points, since each point has 3 coordinates)
    const size_t simd_width = 2; // 2 points = 6 doubles = 2 * __m256d registers
    const size_t simd_end = (num_points / simd_width) * simd_width;

    // Broadcast inv_voxel_size_ to SIMD register (convert float to double)
    const double inv_voxel_size_d = static_cast<double>(inv_voxel_size_);
    const __m256d inv_voxel_size_vec = _mm256_set1_pd(inv_voxel_size_d);

    // Process 2 points at a time (6 doubles total)
    alignas(32) double x_vals[2], y_vals[2], z_vals[2];
    alignas(16) float color_x_vals[2], color_y_vals[2],
        color_z_vals[2]; // Only used if HasColors

    for (size_t i = 0; i < simd_end; i += simd_width) {
        // Load coordinates
        x_vals[0] = pts_ptr[(i + 0) * 3 + 0];
        x_vals[1] = pts_ptr[(i + 1) * 3 + 0];
        y_vals[0] = pts_ptr[(i + 0) * 3 + 1];
        y_vals[1] = pts_ptr[(i + 1) * 3 + 1];
        z_vals[0] = pts_ptr[(i + 0) * 3 + 2];
        z_vals[1] = pts_ptr[(i + 1) * 3 + 2];

        // Load colors (only if HasColors is true)
        if constexpr (HasColors) {
            color_x_vals[0] = cols_ptr[(i + 0) * 3 + 0];
            color_x_vals[1] = cols_ptr[(i + 1) * 3 + 0];
            color_y_vals[0] = cols_ptr[(i + 0) * 3 + 1];
            color_y_vals[1] = cols_ptr[(i + 1) * 3 + 1];
            color_z_vals[0] = cols_ptr[(i + 0) * 3 + 2];
            color_z_vals[1] = cols_ptr[(i + 1) * 3 + 2];
        }

        // Compute voxel keys using SIMD
        // Note: We're loading 2 doubles into a 256-bit register (which can hold 4)
        // This is still efficient as we process 2 points at once
        __m256d x_simd = _mm256_loadu_pd(x_vals);
        __m256d y_simd = _mm256_loadu_pd(y_vals);
        __m256d z_simd = _mm256_loadu_pd(z_vals);

        // Multiply by inv_voxel_size and floor
        __m256d x_scaled = _mm256_mul_pd(x_simd, inv_voxel_size_vec);
        __m256d y_scaled = _mm256_mul_pd(y_simd, inv_voxel_size_vec);
        __m256d z_scaled = _mm256_mul_pd(z_simd, inv_voxel_size_vec);

        x_scaled = _mm256_floor_pd(x_scaled);
        y_scaled = _mm256_floor_pd(y_scaled);
        z_scaled = _mm256_floor_pd(z_scaled);

        // Convert to int32 (need to extract and convert)
        // Store to aligned arrays (only first 2 doubles are used)
        alignas(32) double x_dbl[4], y_dbl[4], z_dbl[4];
        _mm256_storeu_pd(x_dbl, x_scaled);
        _mm256_storeu_pd(y_dbl, y_scaled);
        _mm256_storeu_pd(z_dbl, z_scaled);

        alignas(16) int32_t x_int[2], y_int[2], z_int[2];
        x_int[0] = static_cast<int32_t>(x_dbl[0]);
        x_int[1] = static_cast<int32_t>(x_dbl[1]);
        y_int[0] = static_cast<int32_t>(y_dbl[0]);
        y_int[1] = static_cast<int32_t>(y_dbl[1]);
        z_int[0] = static_cast<int32_t>(z_dbl[0]);
        z_int[1] = static_cast<int32_t>(z_dbl[1]);

        // Group points by voxel key to batch updates
        struct VoxelUpdate {
            double pos_sum[3] = {0.0, 0.0, 0.0};
            float col_sum[3] = {0.0f, 0.0f, 0.0f}; // Only used if HasColors
            int count = 0;
        };
        std::unordered_map<VoxelKey, VoxelUpdate, VoxelKeyHash> batch_updates;
        batch_updates.reserve(simd_width);

        // Accumulate updates for points mapping to the same voxel
        for (size_t j = 0; j < simd_width; ++j) {
            const VoxelKey key{x_int[j], y_int[j], z_int[j]};
            auto &update = batch_updates[key];
            update.pos_sum[0] += x_vals[j];
            update.pos_sum[1] += y_vals[j];
            update.pos_sum[2] += z_vals[j];
            if constexpr (HasColors) {
                update.col_sum[0] += color_x_vals[j];
                update.col_sum[1] += color_y_vals[j];
                update.col_sum[2] += color_z_vals[j];
            }
            update.count++;
        }

        // Apply batched updates using SIMD where possible
        for (const auto &[key, update] : batch_updates) {
#ifdef TBB_FOUND
            auto [it, inserted] = grid_.insert({key, VoxelDataT()});
#else
            auto [it, inserted] = grid_.try_emplace(key);
#endif
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize with accumulated values
                // If count==0, the voxel was reset and should be treated as new
                v.position_sum[0] = update.pos_sum[0];
                v.position_sum[1] = update.pos_sum[1];
                v.position_sum[2] = update.pos_sum[2];
                if constexpr (HasColors) {
                    v.color_sum[0] = update.col_sum[0];
                    v.color_sum[1] = update.col_sum[1];
                    v.color_sum[2] = update.col_sum[2];
                }
                v.count = update.count;
            } else {
                // Existing voxel: add accumulated values using SIMD for doubles
                // Use AVX2 for double addition
                __m256d v_pos_vec =
                    _mm256_set_pd(0.0, v.position_sum[2], v.position_sum[1], v.position_sum[0]);
                __m256d u_pos_vec =
                    _mm256_set_pd(0.0, update.pos_sum[2], update.pos_sum[1], update.pos_sum[0]);
                __m256d result_pos = _mm256_add_pd(v_pos_vec, u_pos_vec);

                alignas(32) double result_pos_arr[4];
                _mm256_store_pd(result_pos_arr, result_pos);
                v.position_sum[0] = result_pos_arr[0];
                v.position_sum[1] = result_pos_arr[1];
                v.position_sum[2] = result_pos_arr[2];

                if constexpr (HasColors) {
                    // Color (float) - use SSE for addition
                    __m128 v_col_vec =
                        _mm_set_ps(0.0f, v.color_sum[2], v.color_sum[1], v.color_sum[0]);
                    __m128 u_col_vec =
                        _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1], update.col_sum[0]);
                    __m128 result_col = _mm_add_ps(v_col_vec, u_col_vec);

                    alignas(16) float result_col_arr[4];
                    _mm_store_ps(result_col_arr, result_col);
                    v.color_sum[0] = result_col_arr[0];
                    v.color_sum[1] = result_col_arr[1];
                    v.color_sum[2] = result_col_arr[2];
                }

                v.count += update.count;
            }
        }
    }

    // Process remaining points with scalar code
    for (size_t i = simd_end; i < num_points; ++i) {
        const size_t idx = i * 3;
        const double x = pts_ptr[idx + 0];
        const double y = pts_ptr[idx + 1];
        const double z = pts_ptr[idx + 2];

        const VoxelKey key = get_voxel_key_inv<double, double>(x, y, z, inv_voxel_size_);

#ifdef TBB_FOUND
        auto [it, inserted] = grid_.insert({key, VoxelDataT()});
#else
        auto [it, inserted] = grid_.try_emplace(key);
#endif
        auto &v = it->second;

        if (inserted || v.count == 0) {
            // New voxel or reset voxel: initialize and update
            // If count==0, the voxel was reset and should be treated as new
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            ++v.count;
        }
    }
#else
    // SSE4.1: process 2 doubles at a time (1 point, since each point has 3 coordinates)
    // We'll process 2 points but handle coordinates separately
    const size_t simd_width = 2; // 2 points
    const size_t simd_end = (num_points / simd_width) * simd_width;

    // Broadcast inv_voxel_size_ to SIMD register (convert float to double)
    const double inv_voxel_size_d = static_cast<double>(inv_voxel_size_);
    const __m128d inv_voxel_size_vec = _mm_set1_pd(inv_voxel_size_d);

    // Process 2 points at a time
    alignas(16) double x_vals[2], y_vals[2], z_vals[2];
    alignas(16) float color_x_vals[2], color_y_vals[2],
        color_z_vals[2]; // Only used if HasColors

    for (size_t i = 0; i < simd_end; i += simd_width) {
        // Load coordinates
        x_vals[0] = pts_ptr[(i + 0) * 3 + 0];
        x_vals[1] = pts_ptr[(i + 1) * 3 + 0];
        y_vals[0] = pts_ptr[(i + 0) * 3 + 1];
        y_vals[1] = pts_ptr[(i + 1) * 3 + 1];
        z_vals[0] = pts_ptr[(i + 0) * 3 + 2];
        z_vals[1] = pts_ptr[(i + 1) * 3 + 2];

        // Load colors (only if HasColors is true)
        if constexpr (HasColors) {
            color_x_vals[0] = cols_ptr[(i + 0) * 3 + 0];
            color_x_vals[1] = cols_ptr[(i + 1) * 3 + 0];
            color_y_vals[0] = cols_ptr[(i + 0) * 3 + 1];
            color_y_vals[1] = cols_ptr[(i + 1) * 3 + 1];
            color_z_vals[0] = cols_ptr[(i + 0) * 3 + 2];
            color_z_vals[1] = cols_ptr[(i + 1) * 3 + 2];
        }

        // Compute voxel keys using SIMD (process x, y, z separately)
        __m128d x_simd = _mm_load_pd(x_vals);
        __m128d y_simd = _mm_load_pd(y_vals);
        __m128d z_simd = _mm_load_pd(z_vals);

        // Multiply by inv_voxel_size and floor
        __m128d x_scaled = _mm_mul_pd(x_simd, inv_voxel_size_vec);
        __m128d y_scaled = _mm_mul_pd(y_simd, inv_voxel_size_vec);
        __m128d z_scaled = _mm_mul_pd(z_simd, inv_voxel_size_vec);

        x_scaled = _mm_floor_pd(x_scaled);
        y_scaled = _mm_floor_pd(y_scaled);
        z_scaled = _mm_floor_pd(z_scaled);

        // Convert to int32
        alignas(16) double x_dbl[2], y_dbl[2], z_dbl[2];
        _mm_store_pd(x_dbl, x_scaled);
        _mm_store_pd(y_dbl, y_scaled);
        _mm_store_pd(z_dbl, z_scaled);

        alignas(16) int32_t x_int[2], y_int[2], z_int[2];
        x_int[0] = static_cast<int32_t>(x_dbl[0]);
        x_int[1] = static_cast<int32_t>(x_dbl[1]);
        y_int[0] = static_cast<int32_t>(y_dbl[0]);
        y_int[1] = static_cast<int32_t>(y_dbl[1]);
        z_int[0] = static_cast<int32_t>(z_dbl[0]);
        z_int[1] = static_cast<int32_t>(z_dbl[1]);

        // Group points by voxel key to batch updates
        struct VoxelUpdate {
            double pos_sum[3] = {0.0, 0.0, 0.0};
            float col_sum[3] = {0.0f, 0.0f, 0.0f}; // Only used if HasColors
            int count = 0;
        };
        std::unordered_map<VoxelKey, VoxelUpdate, VoxelKeyHash> batch_updates;
        batch_updates.reserve(simd_width);

        // Accumulate updates for points mapping to the same voxel
        for (size_t j = 0; j < simd_width; ++j) {
            const VoxelKey key{x_int[j], y_int[j], z_int[j]};
            auto &update = batch_updates[key];
            update.pos_sum[0] += x_vals[j];
            update.pos_sum[1] += y_vals[j];
            update.pos_sum[2] += z_vals[j];
            if constexpr (HasColors) {
                update.col_sum[0] += color_x_vals[j];
                update.col_sum[1] += color_y_vals[j];
                update.col_sum[2] += color_z_vals[j];
            }
            update.count++;
        }

        // Apply batched updates using SIMD where possible
        for (const auto &[key, update] : batch_updates) {
#ifdef TBB_FOUND
            auto [it, inserted] = grid_.insert({key, VoxelDataT()});
#else
            auto [it, inserted] = grid_.try_emplace(key);
#endif
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize with accumulated values
                // If count==0, the voxel was reset and should be treated as new
                v.position_sum[0] = update.pos_sum[0];
                v.position_sum[1] = update.pos_sum[1];
                v.position_sum[2] = update.pos_sum[2];
                if constexpr (HasColors) {
                    v.color_sum[0] = update.col_sum[0];
                    v.color_sum[1] = update.col_sum[1];
                    v.color_sum[2] = update.col_sum[2];
                }
                v.count = update.count;
            } else {
                // Existing voxel: add accumulated values using SIMD for doubles
                // Use SSE2 for double addition (process 2 doubles at a time)
                // For position_sum, we have 3 doubles, so process [0,1] then handle [2]
                // separately
                __m128d v_pos_01 = _mm_loadu_pd(&v.position_sum[0]);
                __m128d u_pos_01 = _mm_loadu_pd(&update.pos_sum[0]);
                __m128d result_01 = _mm_add_pd(v_pos_01, u_pos_01);
                _mm_storeu_pd(&v.position_sum[0], result_01);
                v.position_sum[2] += update.pos_sum[2];

                if constexpr (HasColors) {
                    // Color (float) - use SSE for addition
                    __m128 v_col_vec =
                        _mm_set_ps(0.0f, v.color_sum[2], v.color_sum[1], v.color_sum[0]);
                    __m128 u_col_vec =
                        _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1], update.col_sum[0]);
                    __m128 result_col = _mm_add_ps(v_col_vec, u_col_vec);

                    alignas(16) float result_col_arr[4];
                    _mm_store_ps(result_col_arr, result_col);
                    v.color_sum[0] = result_col_arr[0];
                    v.color_sum[1] = result_col_arr[1];
                    v.color_sum[2] = result_col_arr[2];
                }

                v.count += update.count;
            }
        }
    }

    // Process remaining points with scalar code
    for (size_t i = simd_end; i < num_points; ++i) {
        const size_t idx = i * 3;
        const double x = pts_ptr[idx + 0];
        const double y = pts_ptr[idx + 1];
        const double z = pts_ptr[idx + 2];

        const VoxelKey key =
            get_voxel_key_inv<double, double>(x, y, z, static_cast<double>(inv_voxel_size_));

#ifdef TBB_FOUND
        auto [it, inserted] = grid_.insert({key, VoxelDataT()});
#else
        auto [it, inserted] = grid_.try_emplace(key);
#endif
        auto &v = it->second;

        if (inserted || v.count == 0) {
            // New voxel or reset voxel: initialize and update
            // If count==0, the voxel was reset and should be treated as new
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            v.count = 1;
        } else {
            // Existing voxel: just update
            v.update_point(x, y, z);
            if constexpr (HasColors) {
                const float color_x = cols_ptr[idx + 0];
                const float color_y = cols_ptr[idx + 1];
                const float color_z = cols_ptr[idx + 2];
                v.update_color(color_x, color_y, color_z);
            }
            ++v.count;
        }
    }
#endif // USE_AVX2
#endif // TBB_FOUND
}

} // namespace volumetric

#endif

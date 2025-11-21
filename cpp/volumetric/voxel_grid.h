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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

// SIMD support
#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD 1
#define USE_AVX2 1
#define USE_DOUBLE_SIMD 1
#elif defined(__SSE4_1__) || defined(__SSE4_2__)
#include <emmintrin.h> // SSE2 for double SIMD (__m128d)
#include <smmintrin.h> // SSE4.1 includes floor_ps and floor_pd
#define USE_SIMD 1
#define USE_AVX2 0
#define USE_DOUBLE_SIMD 1
#else
#define USE_SIMD 0
#define USE_AVX2 0
#define USE_DOUBLE_SIMD 0
#endif

#include "voxel_data.h"
#include "voxel_hashing.h"

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

namespace py = pybind11;

namespace volumetric {

// VoxelGrid class with direct voxel hashing
template <typename VoxelDataT> class VoxelGridT {
  public:
    explicit VoxelGridT(float voxel_size = 0.05)
        : voxel_size_(voxel_size), inv_voxel_size_(1.0f / voxel_size) {
        static_assert(Voxel<VoxelDataT>, "VoxelDataT must satisfy the Voxel concept");
    }

    // Insert a point cloud into the voxel grid
    template <typename Tp, typename Tc>
    void integrate(py::array_t<Tp> points, py::array_t<Tc> colors) {
        auto pts_info = points.request();
        auto cols_info = colors.request();

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

        integrate_raw<Tp, Tc>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0],
                              static_cast<const Tc *>(cols_info.ptr));
    }

    // Internal method that does the actual integration work (can be called without GIL)
    template <typename Tp, typename Tc>
    void integrate_raw(const Tp *pts_ptr, size_t num_points, const Tc *cols_ptr) {
        // Use SIMD for float types when available
        if constexpr (std::is_same_v<Tp, float> && std::is_same_v<Tc, float> && USE_SIMD) {
            integrate_raw_simd(pts_ptr, cols_ptr, num_points);
        } else if constexpr (std::is_same_v<Tp, double> && std::is_same_v<Tc, float> &&
                             USE_DOUBLE_SIMD) {
            integrate_raw_simd(pts_ptr, cols_ptr, num_points);
        } else {
            integrate_raw_scalar(pts_ptr, cols_ptr, num_points);
        }
    }

    // Internal method that does the actual integration work (can be called without GIL)
    template <typename Tp> void integrate_raw(const Tp *pts_ptr, size_t num_points) {
        // Use SIMD for float types when available
        if constexpr (std::is_same_v<Tp, float> && USE_SIMD) {
            integrate_raw_simd(pts_ptr, num_points);
        } else if constexpr (std::is_same_v<Tp, double> && USE_DOUBLE_SIMD) {
            integrate_raw_simd(pts_ptr, num_points);
        } else {
            integrate_raw_scalar(pts_ptr, num_points);
        }
    }

    template <typename Tp, typename Tc>
    void integrate_raw_scalar(const Tp *pts_ptr, const Tc *cols_ptr, size_t num_points) {
        integrate_raw_scalar_impl<Tp, Tc, std::is_same_v<Tc, float>>(pts_ptr, cols_ptr, num_points);
    }

    template <typename Tp> void integrate_raw_scalar(const Tp *pts_ptr, size_t num_points) {
        integrate_raw_scalar_impl<Tp, float, false>(pts_ptr, nullptr,
                                                    num_points); // fake float type for colors
    }

    // Scalar version (fallback for non-float types or when SIMD unavailable)
    template <typename Tp, typename Tc, bool HasColors>
    void integrate_raw_scalar_impl(const Tp *pts_ptr, const Tc *cols_ptr, size_t num_points) {
#ifdef TBB_FOUND
        // Parallel version using TBB with concurrent_unordered_map (thread-safe, no mutex needed)
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points),
                          [&](const tbb::blocked_range<size_t> &range) {
                              for (size_t i = range.begin(); i < range.end(); ++i) {
                                  const size_t idx = i * 3;
                                  const Tp x = pts_ptr[idx + 0];
                                  const Tp y = pts_ptr[idx + 1];
                                  const Tp z = pts_ptr[idx + 2];

                                  Tc color_x, color_y, color_z;
                                  if constexpr (HasColors) {
                                      color_x = cols_ptr[idx + 0];
                                      color_y = cols_ptr[idx + 1];
                                      color_z = cols_ptr[idx + 2];
                                  }

                                  const VoxelKey key =
                                      get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

                                  // concurrent_unordered_map is thread-safe, no mutex needed
                                  auto [it, inserted] = grid_.insert({key, VoxelDataT()});
                                  auto &v = it->second;

                                  if (inserted || v.count == 0) {
                                      // New voxel or reset voxel: initialize and update
                                      // If count==0, the voxel was reset and should be treated as
                                      // new
                                      v.update_point(x, y, z);
                                      if constexpr (HasColors) {
                                          v.update_color(color_x, color_y, color_z);
                                      }
                                      v.count = 1;
                                  } else {
                                      // Existing voxel: just update
                                      v.update_point(x, y, z);
                                      if constexpr (HasColors) {
                                          v.update_color(color_x, color_y, color_z);
                                      }
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

            Tc color_x, color_y, color_z;
            if constexpr (HasColors) {
                color_x = cols_ptr[idx + 0];
                color_y = cols_ptr[idx + 1];
                color_z = cols_ptr[idx + 2];
            }

            const VoxelKey key = get_voxel_key_inv<Tp, float>(x, y, z, inv_voxel_size_);

            // Use try_emplace to avoid double lookup - returns pair<iterator, bool>
            auto [it, inserted] = grid_.try_emplace(key);
            auto &v = it->second;

            if (inserted || v.count == 0) {
                // New voxel or reset voxel: initialize and update
                // If count==0, the voxel was reset and should be treated as new
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                v.count = 1;
            } else {
                // Existing voxel: just update
                v.update_point(x, y, z);
                if constexpr (HasColors) {
                    v.update_color(color_x, color_y, color_z);
                }
                ++v.count;
            }
        }
#endif
    }

#if USE_SIMD

    // Wrapper functions that call the unified template implementation
    void integrate_raw_simd(const float *pts_ptr, const float *cols_ptr, size_t num_points) {
        integrate_raw_simd_impl<true>(pts_ptr, cols_ptr, num_points);
    }

    void integrate_raw_simd(const float *pts_ptr, size_t num_points) {
        integrate_raw_simd_impl<false>(pts_ptr, nullptr, num_points);
    }

#endif

#if USE_DOUBLE_SIMD

    // Wrapper functions for double precision points
    void integrate_raw_simd(const double *pts_ptr, const float *cols_ptr, size_t num_points) {
        integrate_raw_simd_impl_double<true>(pts_ptr, cols_ptr, num_points);
    }

    void integrate_raw_simd(const double *pts_ptr, size_t num_points) {
        integrate_raw_simd_impl_double<false>(pts_ptr, nullptr, num_points);
    }

#endif

#if USE_SIMD

    // Unified SIMD-optimized version (processes 4 points at a time)
    // Template parameter HasColors controls whether color processing is enabled at compile time
    template <bool HasColors>
    void integrate_raw_simd_impl(const float *pts_ptr, const float *cols_ptr, size_t num_points) {
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
                        __m128 u_col_vec = _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1],
                                                      update.col_sum[0]);
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
    }

#endif

#if USE_DOUBLE_SIMD

    // Unified SIMD-optimized version for double precision points
    // Template parameter HasColors controls whether color processing is enabled at compile time
    template <bool HasColors>
    void integrate_raw_simd_impl_double(const double *pts_ptr, const float *cols_ptr,
                                        size_t num_points) {
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
                        __m128 u_col_vec = _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1],
                                                      update.col_sum[0]);
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
                        __m128 u_col_vec = _mm_set_ps(0.0f, update.col_sum[2], update.col_sum[1],
                                                      update.col_sum[0]);
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
#endif
    }

#endif

    // Insert a point cloud into the voxel grid
    template <typename Tp> void integrate_points(py::array_t<Tp> points) {
        auto pts_info = points.request();

        // Validate array shape: points should be (N, 3)
        if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
            throw std::runtime_error("points must be a 2D array with shape (N, 3)");
        }
        if (!pts_info.ptr) {
            throw std::runtime_error("points array must be contiguous");
        }

        integrate_raw_scalar<Tp, float>(static_cast<const Tp *>(pts_info.ptr), pts_info.shape[0]);
    }

    std::vector<std::array<double, 3>> get_points() const {
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

    std::vector<std::array<float, 3>> get_colors() const {
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

    std::pair<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>>
    get_voxel_data(int min_count = 1) const {
#ifdef TBB_FOUND
        // Parallel version: collect keys first, filter, then process in parallel
        // Use isolate() to prevent deadlock from nested parallelism
        std::vector<VoxelKey> keys;
        keys.reserve(grid_.size());
        for (const auto &[key, v] : grid_) {
            if (v.count >= min_count) {
                keys.push_back(key);
            }
        }
        std::vector<std::array<double, 3>> points(keys.size());
        std::vector<std::array<float, 3>> colors(keys.size());
        tbb::this_task_arena::isolate([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                              [&](const tbb::blocked_range<size_t> &range) {
                                  for (size_t i = range.begin(); i < range.end(); ++i) {
                                      const auto &v = grid_.at(keys[i]);
                                      points[i] = v.get_position();
                                      colors[i] = v.get_color();
                                  }
                              });
        });
        return {points, colors};
#else
        // Sequential version
        std::vector<std::array<double, 3>> points;
        points.reserve(grid_.size());
        std::vector<std::array<float, 3>> colors;
        colors.reserve(grid_.size());

        // Always filter by min_count to exclude reset voxels (count=0) unless explicitly requested
        for (const auto &[key, v] : grid_) {
            if (v.count >= min_count) {
                points.push_back(v.get_position());
                colors.push_back(v.get_color());
            }
        }
        return {points, colors};
#endif
    }

    // Clear the voxel grid
    void clear() { grid_.clear(); }

    // Get the size of the voxel grid
    size_t size() const { return grid_.size(); }

    // Check if the voxel grid is empty
    bool empty() const { return grid_.empty(); }

  private:
    float voxel_size_;
    float inv_voxel_size_; // Precomputed 1.0f / voxel_size_ for faster division
#ifdef TBB_FOUND
    tbb::concurrent_unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#else
    std::unordered_map<VoxelKey, VoxelDataT, VoxelKeyHash> grid_;
#endif
};

using VoxelGrid = VoxelGridT<VoxelData>;

} // namespace volumetric
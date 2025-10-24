/*
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

#include <climits>
#include <cstdint>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <type_traits>
#include <vector>

namespace pyslam {

enum class NormType { None = -1, Hamming = 0, L2 = 1 };

// Return squared L2 if you don't need the sqrt for comparisons.
enum class L2Variant { TrueL2, Squared };

// -------------------------------------------------------------- //
// ------------------------ Conversions ------------------------ //
// -------------------------------------------------------------- //

inline NormType convert_cv2_norm_type_to_norm_type(const int &cv2_norm_type) {
    // Convert cv2 norm type to NormType
    switch (cv2_norm_type) {
    case cv::NORM_HAMMING:
        return pyslam::NormType::Hamming;
    case cv::NORM_L2:
        return pyslam::NormType::L2;
    default:
        // Default to L2 for unknown norm types
        return pyslam::NormType::None;
    }
}

inline int convert_norm_type_to_cv2_norm_type(const NormType &norm_type) {
    // Convert NormType to cv2 norm type
    switch (norm_type) {
    case pyslam::NormType::Hamming:
        return cv::NORM_HAMMING;
    case pyslam::NormType::L2:
        return cv::NORM_L2;
    default:
        // Default to L2 for unknown norm types
        return cv::NORM_L2;
    }
}

// -------------------------------------------------------------- //
// ------------------------ Hamming (binary) -------------------- //
// -------------------------------------------------------------- //

inline bool is_orb_256(const cv::Mat &m) noexcept {
    return m.type() == CV_8UC1 && m.total() == 32 && m.isContinuous();
}

// Cross-compiler popcount for 64-bit
inline int popcount64(uint64_t x) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    return static_cast<int>(__popcnt64(x));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(static_cast<unsigned long long>(x));
#else
    // Very portable fallback (slower, rarely used)
    using T = uint64_t;
    T v = x;
    v = v - ((v >> 1) & (T) ~(T)0 / 3);
    v = (v & (T) ~(T)0 / 15 * 3) + ((v >> 2) & (T) ~(T)0 / 15 * 3);
    v = (v + (v >> 4)) & (T) ~(T)0 / 255 * 15;
    return static_cast<int>((T)(v * ((T) ~(T)0 / 255)) >> (sizeof(T) - 1) * CHAR_BIT);
#endif
}

// Fast path for 256-bit descriptors (e.g., ORB). Requires CV_8UC1, 32 bytes, continuous.
inline int binary_descriptor_distance_orb256(const cv::Mat &a, const cv::Mat &b) noexcept {
    const auto *pa = reinterpret_cast<const uint64_t *>(a.ptr());
    const auto *pb = reinterpret_cast<const uint64_t *>(b.ptr());
    // 4 * 64 = 256 bits
    int dist = 0;
    dist += popcount64(pa[0] ^ pb[0]);
    dist += popcount64(pa[1] ^ pb[1]);
    dist += popcount64(pa[2] ^ pb[2]);
    dist += popcount64(pa[3] ^ pb[3]);
    return dist;
}

// Generic binary Hamming for any multiple of 64 bits (continuous CV_8UC1).
inline int binary_descriptor_distance(const cv::Mat &a, const cv::Mat &b) noexcept {

#ifndef NDEBUG
    // Expect same type/size and contiguous memory.
    if (a.type() != CV_8UC1 || b.type() != CV_8UC1 || a.total() != b.total() || !a.isContinuous() ||
        !b.isContinuous()) {
        std::cerr << "binary_descriptor_distance: invalid descriptors\n";
        return std::numeric_limits<int>::max();
    }

    // Fast path for common ORB 256-bit
    if (is_orb_256(a) && is_orb_256(b))
        return binary_descriptor_distance_orb256(a, b);
#endif

    const auto nbytes = a.total();
    const auto *pa = reinterpret_cast<const uint64_t *>(a.ptr());
    const auto *pb = reinterpret_cast<const uint64_t *>(b.ptr());

    const size_t n64 = nbytes / sizeof(uint64_t);
    int dist = 0;
    for (size_t i = 0; i < n64; ++i)
        dist += popcount64(pa[i] ^ pb[i]);

    // Handle leftover bytes (if any)
    const uint8_t *pa8 = reinterpret_cast<const uint8_t *>(pa + n64);
    const uint8_t *pb8 = reinterpret_cast<const uint8_t *>(pb + n64);
    const size_t rem = nbytes - n64 * sizeof(uint64_t);
    for (size_t i = 0; i < rem; ++i) {
        dist += static_cast<int>(
            __builtin_popcount(static_cast<unsigned>(pa8[i] ^ pb8[i]))); // small tail
    }
    return dist;
}

inline std::vector<int> binary_descriptor_distances(const cv::Mat &d,
                                                    const std::vector<cv::Mat> &vec_descriptors) {
    std::vector<int> distances;
    distances.reserve(vec_descriptors.size());

    // Pre-validate once
    const bool use_fast = is_orb_256(d);
    for (const auto &m : vec_descriptors) {
        if (use_fast && is_orb_256(m)) {
            distances.emplace_back(binary_descriptor_distance_orb256(d, m));
        } else {
            distances.emplace_back(binary_descriptor_distance(d, m));
        }
    }
    return distances;
}
// -------------------------------------------------------------- //
// ------------------------ L2 (float) -------------------------- //
// -------------------------------------------------------------- //

inline bool is_float_vec(const cv::Mat &m) noexcept {
    return m.type() == CV_32FC1 && m.isContinuous();
}

inline float float_descriptor_distance(const cv::Mat &a, const cv::Mat &b) noexcept {
#ifndef NDEBUG
    if (!is_float_vec(a) || !is_float_vec(b) || a.total() != b.total()) {
        std::cerr << "float_descriptor_distance: invalid descriptors\n";
        return std::numeric_limits<float>::infinity();
    }
#endif

    const float *pa = a.ptr<float>();
    const float *pb = b.ptr<float>();
    const size_t n = a.total();

    // Manual L2 (no sqrt) is often enough; if you need true L2, take sqrtf at the end.
    // Choose one; here we return true L2 to match cv::norm semantics.
    float acc = 0.f;
    size_t i = 0;

    // Unroll by 8 for throughput (safe with -ffast-math off)
    for (; i + 8 <= n; i += 8) {
        float d0 = pa[i + 0] - pb[i + 0];
        float d1 = pa[i + 1] - pb[i + 1];
        float d2 = pa[i + 2] - pb[i + 2];
        float d3 = pa[i + 3] - pb[i + 3];
        float d4 = pa[i + 4] - pb[i + 4];
        float d5 = pa[i + 5] - pb[i + 5];
        float d6 = pa[i + 6] - pb[i + 6];
        float d7 = pa[i + 7] - pb[i + 7];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }
    for (; i < n; ++i) {
        float d0 = pa[i] - pb[i];
        acc += d0 * d0;
    }
    return std::sqrt(acc);
}

#if defined(__AVX512F__)
#include <immintrin.h>
static inline float hsum512_ps(__m512 v) noexcept {
    // _mm512_reduce_add_ps is available with AVX-512VL/AVX-512DQ on many toolchains,
    // but to be safe use a manual reduction:
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 sum256 = _mm256_add_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(sum256);
    __m128 high128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    __m128 s = _mm_hadd_ps(sum128, sum128);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#elif defined(__AVX2__)
#include <immintrin.h>
static inline float hsum256_ps(__m256 v) noexcept {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);  // (sum3,sum3,sum1,sum1)
    __m128 sums = _mm_add_ps(sum, shuf); // (s3+s2, s3+s2, s1+s0, s1+s0)
    shuf = _mm_movehl_ps(shuf, sums);    // (   ,    , s3+s2,   )
    sums = _mm_add_ss(sums, shuf);       // s3+s2+s1+s0
    return _mm_cvtss_f32(sums);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

template <L2Variant variant>
inline float l2_distance_simd(const cv::Mat &a, const cv::Mat &b) noexcept {
    if (!is_float_vec(a) || !is_float_vec(b) || a.total() != b.total())
        return std::numeric_limits<float>::infinity();

    const float *pa = a.ptr<float>();
    const float *pb = b.ptr<float>();
    size_t n = a.total();

    float acc_scalar = 0.f;

#if defined(__AVX512F__)
    // AVX-512: 16 floats per iteration
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
#if defined(__FMA__)
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(pa + i);
        __m512 vb = _mm512_loadu_ps(pb + i);
        __m512 d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc); // acc += d*d
    }
#else
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(pa + i);
        __m512 vb = _mm512_loadu_ps(pb + i);
        __m512 d = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(d, d);
        acc = _mm512_add_ps(acc, sq);
    }
#endif
    acc_scalar += hsum512_ps(acc);

    // tail: handle remaining <=15 floats with AVX2 if available, else scalar
    size_t tail = n % 16;
    size_t j = n - tail;
#if defined(__AVX2__)
    // Use AVX2 for chunks of 8 in the tail
    if (tail >= 8) {
        __m256 vacc = _mm256_setzero_ps();
#if defined(__FMA__)
        __m256 va = _mm256_loadu_ps(pa + j);
        __m256 vb = _mm256_loadu_ps(pb + j);
        __m256 d = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(d, d, vacc);
#else
        __m256 va = _mm256_loadu_ps(pa + j);
        __m256 vb = _mm256_loadu_ps(pb + j);
        __m256 d = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(d, d);
        vacc = _mm256_add_ps(vacc, sq);
#endif
        acc_scalar += hsum256_ps(vacc);
        j += 8;
        tail -= 8;
    }
#endif
    for (; tail > 0; --tail, ++j) {
        float d0 = pa[j] - pb[j];
        acc_scalar += d0 * d0;
    }

#elif defined(__AVX2__)
    // AVX2: 8 floats per iteration
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
#if defined(__FMA__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(pa + i);
        __m256 vb = _mm256_loadu_ps(pb + i);
        __m256 d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc); // acc += d*d
    }
#else
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(pa + i);
        __m256 vb = _mm256_loadu_ps(pb + i);
        __m256 d = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(d, d);
        acc = _mm256_add_ps(acc, sq);
    }
#endif
    acc_scalar += hsum256_ps(acc);
    // tail
    for (; i < n; ++i) {
        float d0 = pa[i] - pb[i];
        acc_scalar += d0 * d0;
    }

#elif defined(__ARM_NEON)
    // NEON: 4 floats per iteration
    float32x4_t vacc = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(pa + i);
        float32x4_t vb = vld1q_f32(pb + i);
        float32x4_t d = vsubq_f32(va, vb);
        vacc = vmlaq_f32(vacc, d, d); // vacc += d*d
    }
    float32x2_t vlow = vget_low_f32(vacc);
    float32x2_t vhigh = vget_high_f32(vacc);
    float32x2_t vsum = vadd_f32(vlow, vhigh);
    float sum2 = vget_lane_f32(vsum, 0) + vget_lane_f32(vsum, 1);
    acc_scalar += sum2;
    // tail
    for (; i < n; ++i) {
        float d0 = pa[i] - pb[i];
        acc_scalar += d0 * d0;
    }

#else
    // Scalar fallback (portable and decent with -O3)
    for (size_t i = 0; i < n; ++i) {
        float d0 = pa[i] - pb[i];
        acc_scalar += d0 * d0;
    }
#endif

    if constexpr (variant == L2Variant::Squared) {
        return acc_scalar;
    } else {
        return std::sqrt(acc_scalar);
    }
}

// Convenience wrapper to match your previous API
inline float float_descriptor_distance_simd(const cv::Mat &a, const cv::Mat &b) noexcept {
    return l2_distance_simd<L2Variant::TrueL2>(a, b);
}

inline std::vector<float> float_descriptor_distances(const cv::Mat &d,
                                                     const std::vector<cv::Mat> &vec_descriptors) {
    std::vector<float> distances;
    distances.reserve(vec_descriptors.size());
    for (const auto &m : vec_descriptors)
#if 0    
        distances.emplace_back(float_descriptor_distance(d, m));
#else
        distances.emplace_back(float_descriptor_distance_simd(d, m));
#endif
    return distances;
}

// -------------------------------------------------------------- //
// ------------------------ Unified API ------------------------ //
// -------------------------------------------------------------- //

template <NormType dist_type>
inline float descriptor_distance(const cv::Mat &a, const cv::Mat &b) noexcept {
    if constexpr (dist_type == NormType::Hamming) {
        return binary_descriptor_distance(a, b);
    } else if constexpr (dist_type == NormType::L2) {
        return float_descriptor_distance(a, b);
    } else {
        std::cerr << "Invalid distance type\n";
        return std::numeric_limits<float>::infinity();
    }
}

template <>
inline float descriptor_distance<NormType::Hamming>(const cv::Mat &a, const cv::Mat &b) noexcept {
    return binary_descriptor_distance(a, b);
}

template <>
inline float descriptor_distance<NormType::L2>(const cv::Mat &a, const cv::Mat &b) noexcept {
    return float_descriptor_distance(a, b);
}

inline float descriptor_distance(const cv::Mat &a, const cv::Mat &b,
                                 const NormType &dist_type) noexcept {
    switch (dist_type) {
    case NormType::Hamming:
        return binary_descriptor_distance(a, b);
    case NormType::L2:
        return float_descriptor_distance(a, b);
    default:
        std::cerr << "Invalid distance type\n";
        return std::numeric_limits<float>::infinity();
    }
}

inline std::vector<float> descriptor_distances(const cv::Mat &d,
                                               const std::vector<cv::Mat> &vec_descriptors,
                                               NormType dist_type) {
    switch (dist_type) {
    case NormType::Hamming: {
        const auto ints = binary_descriptor_distances(d, vec_descriptors);
        return std::vector<float>(ints.begin(), ints.end());
    }
    case NormType::L2:
        return float_descriptor_distances(d, vec_descriptors);
    default:
        std::cerr << "Invalid distance type\n";
        return {};
    }
}

inline std::vector<std::vector<float>>
compute_distances_matrix(const std::vector<cv::Mat> &descriptors, NormType dist_type) {
    const int num_descriptors = descriptors.size();
    // compute the descriptor distances (all pairs)
    std::vector<std::vector<float>> distances_matrix(num_descriptors,
                                                     std::vector<float>(num_descriptors, 0.0f));
    switch (dist_type) {
    case NormType::Hamming: {
        for (size_t i = 0; i < num_descriptors; ++i) {
            for (size_t j = i + 1; j < num_descriptors; ++j) {
                const float distance = binary_descriptor_distance(descriptors[i], descriptors[j]);
                distances_matrix[i][j] = distance;
                distances_matrix[j][i] = distance;
            }
        }
        break;
    }
    case NormType::L2: {
        for (size_t i = 0; i < num_descriptors; ++i) {
            for (size_t j = i + 1; j < num_descriptors; ++j) {
                const float distance = float_descriptor_distance(descriptors[i], descriptors[j]);
                distances_matrix[i][j] = distance;
                distances_matrix[j][i] = distance;
            }
        }
        break;
    }
    default:
        std::cerr << "Invalid distance type\n";
        return {};
    }
    return distances_matrix;
}

} // namespace pyslam

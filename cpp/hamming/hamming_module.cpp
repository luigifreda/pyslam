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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__AVX2__) || defined(__SSE3__) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace py = pybind11;

// -------------------------
// Helpers: popcount scalar
// -------------------------
static inline uint32_t popcnt64(uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt64(x));
#else
    return static_cast<uint32_t>(__builtin_popcountll(x));
#endif
}

static inline uint32_t hamming_scalar_u8(const uint8_t *a, const uint8_t *b, size_t nbytes) {
    uint32_t sum = 0;

    // 64-bit chunks
    size_t i = 0;
    for (; i + 8 <= nbytes; i += 8) {
        uint64_t wa, wb;
        std::memcpy(&wa, a + i, 8);
        std::memcpy(&wb, b + i, 8);
        sum += popcnt64(wa ^ wb);
    }

    // remainder bytes (tableless)
    for (; i < nbytes; ++i) {
        sum += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(a[i] ^ b[i])));
    }
    return sum;
}

// --------------------------------------------------
// SIMD nibble-popcount (AVX2/SSE): counts bits in bytes
// Technique: PSHUFB on low/high nibbles + SAD to sum
// --------------------------------------------------
#if defined(__AVX2__)
static inline uint32_t hamming_avx2_u8(const uint8_t *a, const uint8_t *b, size_t nbytes) {
    const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1,
                                            2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    uint64_t acc = 0;
    size_t i = 0;

    // 32-byte blocks
    for (; i + 32 <= nbytes; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
        __m256i vx = _mm256_xor_si256(va, vb);

        __m256i lo = _mm256_and_si256(vx, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vx, 4), low_mask);

        __m256i pc_lo = _mm256_shuffle_epi8(lookup, lo);
        __m256i pc_hi = _mm256_shuffle_epi8(lookup, hi);
        __m256i pc = _mm256_add_epi8(pc_lo, pc_hi);

        // Sum bytes into 4x uint64 using SAD (sum of abs diffs to zero)
        __m256i sad = _mm256_sad_epu8(pc, zero);
        // Extract and accumulate
        acc += static_cast<uint64_t>(_mm256_extract_epi64(sad, 0));
        acc += static_cast<uint64_t>(_mm256_extract_epi64(sad, 1));
        acc += static_cast<uint64_t>(_mm256_extract_epi64(sad, 2));
        acc += static_cast<uint64_t>(_mm256_extract_epi64(sad, 3));
    }

    // tail
    if (i < nbytes) {
        acc += hamming_scalar_u8(a + i, b + i, nbytes - i);
    }
    return static_cast<uint32_t>(acc);
}
#endif

#if defined(__SSE3__) || defined(__SSE2__)
static inline uint32_t hamming_sse_u8(const uint8_t *a, const uint8_t *b, size_t nbytes) {
    const __m128i lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    const __m128i zero = _mm_setzero_si128();

    uint64_t acc = 0;
    size_t i = 0;

    // 16-byte blocks
    for (; i + 16 <= nbytes; i += 16) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        __m128i vx = _mm_xor_si128(va, vb);

        __m128i lo = _mm_and_si128(vx, low_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(vx, 4), low_mask);

        __m128i pc_lo = _mm_shuffle_epi8(lookup, lo);
        __m128i pc_hi = _mm_shuffle_epi8(lookup, hi);
        __m128i pc = _mm_add_epi8(pc_lo, pc_hi);

        __m128i sad = _mm_sad_epu8(pc, zero);
        acc += static_cast<uint64_t>(_mm_extract_epi64(sad, 0));
        acc += static_cast<uint64_t>(_mm_extract_epi64(sad, 1));
    }

    // tail
    if (i < nbytes) {
        acc += hamming_scalar_u8(a + i, b + i, nbytes - i);
    }
    return static_cast<uint32_t>(acc);
}
#endif

static inline uint32_t hamming_u8_best(const uint8_t *a, const uint8_t *b, size_t nbytes) {
#if defined(__AVX2__)
    return hamming_avx2_u8(a, b, nbytes);
#elif defined(__SSE3__) || defined(__SSE2__)
    return hamming_sse_u8(a, b, nbytes);
#else
    return hamming_scalar_u8(a, b, nbytes);
#endif
}

// -------------------------
// Zero-copy array checks
// -------------------------
static inline void require_uint8_c_contig(const py::array &arr, const char *name) {
    if (!py::isinstance<py::array>(arr)) {
        throw std::invalid_argument(std::string(name) + " must be a numpy array");
    }
    if (arr.dtype().kind() != 'u' || arr.itemsize() != 1) {
        throw std::invalid_argument(std::string(name) + " must have dtype=uint8");
    }
    // Refuse non C-contiguous to guarantee zero-copy
    if (!(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(name) +
                                    " must be C-contiguous (no-copy requirement)");
    }
}

// -------------------------
// Python bindings
// -------------------------
static int hamming_1d(py::array a, py::array b) {
    require_uint8_c_contig(a, "a");
    require_uint8_c_contig(b, "b");
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("hamming(a,b): both must be 1D uint8 arrays");
    }
    if (a.shape(0) != b.shape(0)) {
        throw std::invalid_argument("hamming(a,b): shapes must match");
    }
    const auto nbytes = static_cast<size_t>(a.shape(0));
    const auto *ap = static_cast<const uint8_t *>(a.data());
    const auto *bp = static_cast<const uint8_t *>(b.data());
    return static_cast<int>(hamming_u8_best(ap, bp, nbytes));
}

static py::array_t<uint16_t> hamming_many(py::array query, py::array descs) {
    require_uint8_c_contig(query, "query");
    require_uint8_c_contig(descs, "descs");

    if (query.ndim() != 1 || descs.ndim() != 2) {
        throw std::invalid_argument(
            "hamming_many(query,descs): query must be (B,), descs must be (N,B)");
    }
    const ssize_t B = query.shape(0);
    if (descs.shape(1) != B) {
        throw std::invalid_argument(
            "hamming_many(query,descs): descs.shape[1] must equal query.shape[0]");
    }

    const ssize_t N = descs.shape(0);
    auto out = py::array_t<uint16_t>(N);
    auto *outp = static_cast<uint16_t *>(out.mutable_data());

    const auto *q = static_cast<const uint8_t *>(query.data());
    const auto *D = static_cast<const uint8_t *>(descs.data());

    // Distance fits in uint16 for typical descriptor sizes (e.g., <= 2048 bits).
    for (ssize_t i = 0; i < N; ++i) {
        const uint8_t *di = D + static_cast<size_t>(i) * static_cast<size_t>(B);
        outp[i] = static_cast<uint16_t>(hamming_u8_best(q, di, static_cast<size_t>(B)));
    }
    return out;
}

static py::array_t<uint16_t> hamming_pairwise(py::array a, py::array b) {
    require_uint8_c_contig(a, "a");
    require_uint8_c_contig(b, "b");

    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("hamming_pairwise(a,b): both must be 2D uint8 arrays (N,B)");
    }
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
        throw std::invalid_argument("hamming_pairwise(a,b): shapes must match");
    }

    const ssize_t N = a.shape(0);
    const ssize_t B = a.shape(1);

    auto out = py::array_t<uint16_t>(N);
    auto *outp = static_cast<uint16_t *>(out.mutable_data());

    const auto *ap = static_cast<const uint8_t *>(a.data());
    const auto *bp = static_cast<const uint8_t *>(b.data());

    for (ssize_t i = 0; i < N; ++i) {
        const uint8_t *ai = ap + static_cast<size_t>(i) * static_cast<size_t>(B);
        const uint8_t *bi = bp + static_cast<size_t>(i) * static_cast<size_t>(B);
        outp[i] = static_cast<uint16_t>(hamming_u8_best(ai, bi, static_cast<size_t>(B)));
    }
    return out;
}

//
static inline py::array borrow_array_no_copy(py::handle h, const char *name) {
    if (!py::isinstance<py::array>(h)) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a numpy array (no-copy requirement)");
    }
    // Borrow handle (no allocation, no copy)
    return py::reinterpret_borrow<py::array>(h);
}

// ------------------------------------------------------------------
// Drop-in replacements for the WRONG byte-counting implementations
// ------------------------------------------------------------------

// Replacement for:
// @njit
// def hamming_distance(a, b):
static int hamming_bits(py::array a, py::array b) {
    require_uint8_c_contig(a, "a");
    require_uint8_c_contig(b, "b");

    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("hamming_bits(a,b): both must be 1D uint8 arrays");
    }
    if (a.shape(0) != b.shape(0)) {
        throw std::invalid_argument("hamming_bits(a,b): shapes must match");
    }

    const auto nbytes = static_cast<size_t>(a.shape(0));
    const auto *ap = static_cast<const uint8_t *>(a.data());
    const auto *bp = static_cast<const uint8_t *>(b.data());

    return static_cast<int>(hamming_u8_best(ap, bp, nbytes));
}

// Replacement for:
// def hamming_distances(a, b):
// Supports multiple modes:
// 1. a is 1D (B,), b is 2D (N,B) -> returns (N,) distances from a to each row of b
// 2. a is 1D (B,), b is Python list/sequence of 1D arrays -> converts b to 2D and computes
// distances
// 3. a is 2D (N,B), b is 2D (N,B) -> returns (N,) pairwise distances
static py::array_t<uint16_t> hamming_bits_many(py::object a_obj, py::object b_obj) {
    // ---- a must already be a numpy array: true zero-copy ----
    py::array a = borrow_array_no_copy(a_obj, "a");
    require_uint8_c_contig(a, "a");

    // ---------------------------------------------------------
    // Case A: b is a numpy array (zero-copy view)
    // ---------------------------------------------------------
    if (py::isinstance<py::array>(b_obj)) {
        py::array b = py::reinterpret_borrow<py::array>(b_obj);
        require_uint8_c_contig(b, "b");

        // Handle case 1: a is 1D, b is 2D -> (N,) distances
        if (a.ndim() == 1 && b.ndim() == 2) {
            const ssize_t B = a.shape(0);
            if (b.shape(1) != B) {
                throw std::invalid_argument(
                    "hamming_distances(a,b): if a is 1D, b.shape[1] must equal a.shape[0]");
            }
            const ssize_t N = b.shape(0);

            auto out = py::array_t<uint16_t>(N);
            auto *outp = static_cast<uint16_t *>(out.mutable_data());

            const auto *ap = static_cast<const uint8_t *>(a.data());
            const auto *bp = static_cast<const uint8_t *>(b.data());

            for (ssize_t i = 0; i < N; ++i) {
                const uint8_t *bi = bp + static_cast<size_t>(i) * static_cast<size_t>(B);
                outp[i] = static_cast<uint16_t>(hamming_u8_best(ap, bi, static_cast<size_t>(B)));
            }
            return out;
        }

        // Handle case 2: a is 2D, b is 2D (pairwise) -> (N,) distances
        if (a.ndim() == 2 && b.ndim() == 2) {
            if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
                throw std::invalid_argument(
                    "hamming_distances(a,b): if both are 2D, shapes must match");
            }
            const ssize_t N = a.shape(0);
            const ssize_t B = a.shape(1);

            auto out = py::array_t<uint16_t>(N);
            auto *outp = static_cast<uint16_t *>(out.mutable_data());

            const auto *ap = static_cast<const uint8_t *>(a.data());
            const auto *bp = static_cast<const uint8_t *>(b.data());

            for (ssize_t i = 0; i < N; ++i) {
                const uint8_t *ai = ap + static_cast<size_t>(i) * static_cast<size_t>(B);
                const uint8_t *bi = bp + static_cast<size_t>(i) * static_cast<size_t>(B);
                outp[i] = static_cast<uint16_t>(hamming_u8_best(ai, bi, static_cast<size_t>(B)));
            }
            return out;
        }

        throw std::invalid_argument("hamming_distances(a,b): with numpy b, supported shapes are "
                                    "a:(B,) with b:(N,B) OR a:(N,B) with b:(N,B)");
    }

    // ---------------------------------------------------------
    // Case B: b is a Python sequence of numpy arrays (zero-copy per element)
    // ---------------------------------------------------------
    if (py::isinstance<py::sequence>(b_obj)) {
        py::sequence b_seq = py::reinterpret_borrow<py::sequence>(b_obj);
        const ssize_t N = py::len(b_seq);
        if (N == 0) {
            throw std::invalid_argument("hamming_distances(a,b): b sequence is empty");
        }

        // For the sequence case, we support:
        //   a: (B,) and b_seq: [ (B,), (B,), ... ]   -> (N,)
        if (a.ndim() != 1) {
            throw std::invalid_argument(
                "hamming_distances(a,b): if b is a sequence, a must be 1D (B,) for true zero-copy");
        }
        const ssize_t B = a.shape(0);
        const auto *ap = static_cast<const uint8_t *>(a.data());

        auto out = py::array_t<uint16_t>(N);
        auto *outp = static_cast<uint16_t *>(out.mutable_data());

        for (ssize_t i = 0; i < N; ++i) {
            py::handle elem_h = b_seq[i];
            py::array bi_arr = borrow_array_no_copy(elem_h, "b[i]");
            require_uint8_c_contig(bi_arr, "b[i]");

            if (bi_arr.ndim() != 1 || bi_arr.shape(0) != B) {
                throw std::invalid_argument(
                    "hamming_distances(a,b): each b[i] must be 1D with shape (B,) matching a");
            }

            const auto *bp = static_cast<const uint8_t *>(bi_arr.data());
            outp[i] = static_cast<uint16_t>(hamming_u8_best(ap, bp, static_cast<size_t>(B)));
        }
        return out;
    }

    throw std::invalid_argument(
        "hamming_distances(a,b): b must be a numpy array or a Python sequence of numpy arrays "
        "(no-copy requirement)");
}

PYBIND11_MODULE(hamming, m) {
    m.doc() = "Zero-copy SIMD Hamming distance for uint8 descriptors (AVX2/SSE/scalar)";

    m.def("hamming", &hamming_1d, "Hamming distance between two uint8 1D descriptors");
    m.def("hamming_many", &hamming_many, "Hamming distances: query (B,) vs descs (N,B)");
    m.def("hamming_pairwise", &hamming_pairwise,
          "Hamming distances: a (N,B) vs b (N,B) elementwise");

    m.def("hamming_distance", &hamming_bits,
          "Correct bit-level Hamming distance (drop-in for np.count_nonzero(a!=b))");

    m.def("hamming_distances", &hamming_bits_many,
          "Correct bit-level Hamming distances: a (1D or 2D) vs b (2D array or sequence). "
          "If a is 1D (B,), computes distances from a to each row of b (N,B). "
          "If a is 2D (N,B), computes pairwise distances with b (N,B). "
          "If b is a Python list/sequence of 1D arrays, efficiently converts to 2D array.");
}

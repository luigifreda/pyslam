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
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "utils/serialization_numpy.h"

namespace py = pybind11;
using pyslam::cvmat_to_numpy;
using pyslam::numpy_to_cvmat;
using pyslam::numpy_to_cvmat_auto;

// Template-based fill function for different data types
template <typename T> static void fill_mat(cv::Mat &m, T base_value) {
    const int ch = m.channels();
    for (int r = 0; r < m.rows; ++r) {
        auto *row = m.ptr<T>(r);
        for (int c = 0; c < m.cols; ++c) {
            for (int k = 0; k < ch; ++k) {
                // Use different patterns for different types to avoid overflow
                if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                    row[c * ch + k] = static_cast<T>((r * 31 + c * 17 + k * 7) & 0xFF);
                } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) {
                    row[c * ch + k] = static_cast<T>((r * 31 + c * 17 + k * 7) % 1000);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    row[c * ch + k] = static_cast<T>((r * 31 + c * 17 + k * 7) % 100000);
                } else if constexpr (std::is_same_v<T, float>) {
                    // Use integer-valued floats to avoid FP comparison issues
                    row[c * ch + k] = static_cast<float>((r * 13 + c * 5 + k) % 1000);
                } else if constexpr (std::is_same_v<T, double>) {
                    // Use integer-valued doubles to avoid FP comparison issues
                    row[c * ch + k] = static_cast<double>((r * 13 + c * 5 + k) % 1000);
                } else {
                    row[c * ch + k] = static_cast<T>(base_value + r * 31 + c * 17 + k * 7);
                }
            }
        }
    }
}

// Specialized fill functions for each depth type
static void fill_u8(cv::Mat &m) { fill_mat<uint8_t>(m, 0); }
static void fill_s8(cv::Mat &m) { fill_mat<int8_t>(m, 0); }
static void fill_u16(cv::Mat &m) { fill_mat<uint16_t>(m, 0); }
static void fill_s16(cv::Mat &m) { fill_mat<int16_t>(m, 0); }
static void fill_s32(cv::Mat &m) { fill_mat<int32_t>(m, 0); }
static void fill_f32(cv::Mat &m) { fill_mat<float>(m, 0.0f); }
static void fill_f64(cv::Mat &m) { fill_mat<double>(m, 0.0); }

#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
static void fill_f16(cv::Mat &m) {
    const int ch = m.channels();
    for (int r = 0; r < m.rows; ++r) {
        auto *row = m.ptr<uint16_t>(r); // CV_16F is stored as uint16_t
        for (int c = 0; c < m.cols; ++c) {
            for (int k = 0; k < ch; ++k) {
                // Simple pattern for half-precision floats
                row[c * ch + k] = static_cast<uint16_t>((r * 31 + c * 17 + k * 7) % 1000);
            }
        }
    }
}
#endif

static bool mats_equal_exact(const cv::Mat &a, const cv::Mat &b) {
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    if (a.empty())
        return b.empty();
    const cv::Mat ac = a.isContinuous() ? a : a.clone();
    const cv::Mat bc = b.isContinuous() ? b : b.clone();
    const size_t bytes = ac.total() * ac.elemSize();
    return bytes == bc.total() * bc.elemSize() && std::memcmp(ac.data, bc.data, bytes) == 0;
}

static void debug_numpy_array(const py::array &a, const std::string &label) {
    if (!a) {
        std::cout << label << ": null array" << std::endl;
        return;
    }

    auto info = a.request();
    std::cout << label << ": dtype_num=" << a.dtype().num() << ", format=" << py::str(a.dtype())
              << ", itemsize=" << a.dtype().itemsize() << ", ndim=" << a.ndim() << std::endl;
}

static void test_roundtrip_dtype(int depth_hint, int channels) {
    // Choose sizes that are not multiples to exercise strides
    const int rows = 23, cols = 31;
    const int type = CV_MAKETYPE(depth_hint, channels);
    cv::Mat m(rows, cols, type);

    // Fill with appropriate data based on depth type
    switch (depth_hint) {
    case CV_8U:
        fill_u8(m);
        break;
    case CV_8S:
        fill_s8(m);
        break;
    case CV_16U:
        fill_u16(m);
        break;
    case CV_16S:
        fill_s16(m);
        break;
    case CV_32S:
        fill_s32(m);
        break;
    case CV_32F:
        fill_f32(m);
        break;
    case CV_64F:
        fill_f64(m);
        break;
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F:
        fill_f16(m);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported depth_hint in test: " + std::to_string(depth_hint));
    }

    // Must be continuous due to CV_Assert in cvmat_to_numpy
    CV_Assert(m.isContinuous());

    // cv::Mat -> numpy
    py::array a = cvmat_to_numpy(m);
    if (!a)
        throw std::runtime_error("cvmat_to_numpy returned empty array");

    // Debug output for CV_64F
    if (depth_hint == CV_64F) {
        debug_numpy_array(a, "CV_64F numpy array");
    }

    // Verify dimensions
    if (channels == 1) {
        if (a.ndim() != 2)
            throw std::runtime_error("ndim mismatch (expect 2)");
        if (a.shape(0) != rows || a.shape(1) != cols)
            throw std::runtime_error("shape mismatch for 2D");
    } else {
        if (a.ndim() != 3)
            throw std::runtime_error("ndim mismatch (expect 3)");
        if (a.shape(0) != rows || a.shape(1) != cols || a.shape(2) != channels)
            throw std::runtime_error("shape mismatch for 3D");
    }

    // numpy -> cv::Mat (clone-owning)
    cv::Mat back = numpy_to_cvmat(a, depth_hint);
    if (!mats_equal_exact(m, back))
        throw std::runtime_error("roundtrip mismatch for depth_hint=" + std::to_string(depth_hint) +
                                 " ch=" + std::to_string(channels));

    // Test auto-detection as well (except for CV_16F which can't be auto-detected)
    if (depth_hint != CV_16F) {
        cv::Mat back_auto = numpy_to_cvmat_auto(a);
        if (!mats_equal_exact(m, back_auto))
            throw std::runtime_error(
                "auto-detection mismatch for depth_hint=" + std::to_string(depth_hint) +
                " ch=" + std::to_string(channels));
    }
}

static void test_empty_handling() {
    cv::Mat empty;
    py::array a = cvmat_to_numpy(empty);

    // Accept either: a null handle OR a valid array with zero elements
    if (a) {
        // If it's a real array, it must represent "empty"
        // size() is robust here; ndim can be 0 or 1 depending on construction
        if (a.size() != 0)
            throw std::runtime_error("empty cv::Mat should yield a zero-sized array");
    }

    cv::Mat back = numpy_to_cvmat(a, CV_8U);
    if (!back.empty())
        throw std::runtime_error("empty array should yield empty cv::Mat");
}

static void test_all_depth_types() {
    // Test all supported depth types with 1 and 3 channels
    const std::vector<int> depth_types = {CV_8U,
                                          CV_8S,
                                          CV_16U,
                                          CV_16S,
                                          CV_32S,
                                          CV_32F,
                                          CV_64F
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
                                          ,
                                          CV_16F
#endif
    };

    const std::vector<int> channel_counts = {1, 3};

    for (int depth : depth_types) {
        for (int channels : channel_counts) {
            std::cout << "Testing depth=" << depth << " channels=" << channels << std::endl;
            test_roundtrip_dtype(depth, channels);
        }
    }
}

static void test_edge_cases() {
    // Test single element matrices
    test_roundtrip_dtype(CV_8U, 1);
    test_roundtrip_dtype(CV_32F, 1);

    // Test very small matrices
    cv::Mat tiny(1, 1, CV_8UC1);
    tiny.at<uint8_t>(0, 0) = 42;
    py::array a = cvmat_to_numpy(tiny);
    cv::Mat back = numpy_to_cvmat(a, CV_8U);
    if (!mats_equal_exact(tiny, back))
        throw std::runtime_error("tiny matrix roundtrip failed");
}

int main() {
    try {
        // Start embedded Python and ensure NumPy is initialized
        py::scoped_interpreter guard{};
        py::module_::import("numpy");

        std::cout << "Testing empty matrix handling..." << std::endl;
        test_empty_handling();

        std::cout << "Testing all depth types..." << std::endl;
        test_all_depth_types();

        std::cout << "Testing edge cases..." << std::endl;
        test_edge_cases();

        std::cout << "All numpy <-> cv::Mat tests passed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
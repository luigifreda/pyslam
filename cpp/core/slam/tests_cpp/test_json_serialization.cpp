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
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

// Include project header (path is from this file under repo root)
#include "utils/serialization_json.h"

using pyslam::cv_depth_to_numpy_dtype;
using pyslam::cv_mat_to_json;
using pyslam::cv_mat_to_json_array;
using pyslam::cv_mat_to_json_raw;
using pyslam::eigen_matrix_to_json_array;
using pyslam::flexible_json_to_eigen_matrix;
using pyslam::json_array_to_cv_mat;
using pyslam::json_array_to_eigen_matrix;
using pyslam::json_to_cv_mat;
using pyslam::json_to_cv_mat_raw;
using pyslam::safe_json_get;
using pyslam::safe_json_get_array;
using pyslam::safe_json_get_array_nullable_int;
using pyslam::safe_parse_keypoints;
using pyslam::safe_parse_vector3d;

static void fill_int_pattern(cv::Mat &m) {
    // deterministic integer pattern that is exactly representable across formats
    const int channels = m.channels();
    for (int r = 0; r < m.rows; ++r) {
        switch (m.depth()) {
        case CV_8U: {
            uint8_t *row = m.ptr<uint8_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 1000) - 500;
                    row[c * channels + ch] =
                        static_cast<uint8_t>(std::max(0, std::min(255, val + 128)));
                }
            }
            break;
        }
        case CV_8S: {
            int8_t *row = m.ptr<int8_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 1000) - 500;
                    row[c * channels + ch] =
                        static_cast<int8_t>(std::max(-128, std::min(127, val % 128)));
                }
            }
            break;
        }
        case CV_16U: {
            uint16_t *row = m.ptr<uint16_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 1000) - 500;
                    row[c * channels + ch] = static_cast<uint16_t>(val + 500);
                }
            }
            break;
        }
        case CV_16S: {
            int16_t *row = m.ptr<int16_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 1000) - 500;
                    row[c * channels + ch] = static_cast<int16_t>(val);
                }
            }
            break;
        }
        case CV_32S: {
            int32_t *row = m.ptr<int32_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 1000) - 500;
                    row[c * channels + ch] = static_cast<int32_t>(val * 1000);
                }
            }
            break;
        }
        case CV_32F: {
            float *row = m.ptr<float>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 1000;
                    row[c * channels + ch] = static_cast<float>(iv);
                }
            }
            break;
        }
        case CV_64F: {
            double *row = m.ptr<double>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 1000;
                    row[c * channels + ch] = static_cast<double>(iv);
                }
            }
            break;
        }
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
        case CV_16F: {
            // Fill with deterministic 16-bit pattern; compare is byte-wise so this is fine
            uint16_t *row = m.ptr<uint16_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 65536;
                    row[c * channels + ch] = static_cast<uint16_t>(iv);
                }
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported depth in fill_int_pattern");
        }
    }
}

static void fill_float_pattern(cv::Mat &m) {
    // For float depths, use integer-valued samples to avoid FP rounding ambiguity
    const int channels = m.channels();
    for (int r = 0; r < m.rows; ++r) {
        if (m.depth() == CV_32F) {
            float *row = m.ptr<float>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 1000;
                    row[c * channels + ch] = static_cast<float>(iv);
                }
            }
        } else if (m.depth() == CV_64F) {
            double *row = m.ptr<double>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 1000;
                    row[c * channels + ch] = static_cast<double>(iv);
                }
            }
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
        } else if (m.depth() == CV_16F) {
            // Fill raw 16-bit values; tests compare byte-wise
            uint16_t *row = m.ptr<uint16_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int iv = (r * 13 + c * 7 + ch * 3) % 65536;
                    row[c * channels + ch] = static_cast<uint16_t>(iv);
                }
            }
#endif
        } else {
            // For non-float depths delegate to integer filler
            // (shouldn't be called with other depths by the test)
            throw std::runtime_error("fill_float_pattern called with non-float depth");
        }
    }
}

static bool mats_equal_exact(const cv::Mat &a, const cv::Mat &b) {
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    if (a.empty())
        return true;

    // Compare byte-wise for robustness across depths
    if (!a.isContinuous() || !b.isContinuous()) {
        cv::Mat ac = a.isContinuous() ? a : a.clone();
        cv::Mat bc = b.isContinuous() ? b : b.clone();
        return ac.total() * ac.elemSize() == bc.total() * bc.elemSize() &&
               std::memcmp(ac.data, bc.data, ac.total() * ac.elemSize()) == 0;
    }
    return std::memcmp(a.data, b.data, a.total() * a.elemSize()) == 0;
}

static void test_roundtrip_b64(int depth, int channels) {
    cv::Mat m(23, 31, CV_MAKETYPE(depth, channels));
    if (depth == CV_32F || depth == CV_64F
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
        || depth == CV_16F
#endif
    )
        fill_float_pattern(m);
    else
        fill_int_pattern(m);

    // also exercise non-contiguous by taking an ROI
    cv::Mat big(30, 40, m.type(), cv::Scalar::all(0));
    m.copyTo(big(cv::Rect(3, 4, m.cols, m.rows)));
    cv::Mat roi = big(cv::Rect(3, 4, m.cols, m.rows)); // may be non-contiguous

    // encode/decode
    nlohmann::json j = cv_mat_to_json(roi);
    cv::Mat dec = json_to_cv_mat(j);

    if (!mats_equal_exact(m, dec)) {
        throw std::runtime_error("B64 roundtrip failed for depth=" + std::to_string(depth) +
                                 " channels=" + std::to_string(channels));
    }
}

static void test_roundtrip_raw(int depth, int channels) {
    cv::Mat m(15, 27, CV_MAKETYPE(depth, channels));
    if (depth == CV_32F || depth == CV_64F
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
        || depth == CV_16F
#endif
    )
        fill_float_pattern(m);
    else
        fill_int_pattern(m);

    // encode/decode
    nlohmann::json j = cv_mat_to_json_raw(m);
    cv::Mat dec = json_to_cv_mat_raw(j);

    if (!mats_equal_exact(m, dec)) {
        throw std::runtime_error("RAW roundtrip failed for depth=" + std::to_string(depth) +
                                 " channels=" + std::to_string(channels));
    }
}

static void test_empty() {
    cv::Mat empty;
    // B64
    auto j1 = cv_mat_to_json(empty);
    if (!j1.is_null())
        throw std::runtime_error("Empty mat should serialize to null (B64)");
    auto d1 = json_to_cv_mat(j1);
    if (!d1.empty())
        throw std::runtime_error("Empty mat decode (B64) should be empty");

    // RAW
    auto j2 = cv_mat_to_json_raw(empty);
    if (!j2.is_null())
        throw std::runtime_error("Empty mat should serialize to null (RAW)");
    auto d2 = json_to_cv_mat_raw(j2);
    if (!d2.empty())
        throw std::runtime_error("Empty mat decode (RAW) should be empty");
}

static void test_channel4_and_small_sizes() {
    // 4-channel support
    test_roundtrip_b64(CV_8U, 4);
    test_roundtrip_raw(CV_8U, 4);
    test_roundtrip_b64(CV_32F, 4);
    test_roundtrip_raw(CV_32F, 4);

    // Small/odd sizes to stress Base64 padding
    auto run_sizes = [&](int depth, int ch) {
        std::vector<std::pair<int, int>> sizes = {{1, 1}, {1, 2}, {2, 1}, {3, 1}, {1, 3}};
        for (auto &rc : sizes) {
            int r = rc.first, c = rc.second;
            cv::Mat m(r, c, CV_MAKETYPE(depth, ch));
            if (depth == CV_32F || depth == CV_64F
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
                || depth == CV_16F
#endif
            )
                fill_float_pattern(m);
            else
                fill_int_pattern(m);
            auto jb = cv_mat_to_json(m);
            auto mb = json_to_cv_mat(jb);
            if (!mats_equal_exact(m, mb))
                throw std::runtime_error("B64 small-size fail: depth=" + std::to_string(depth) +
                                         " ch=" + std::to_string(ch) +
                                         " size=" + std::to_string(r) + "x" + std::to_string(c));
            auto jr = cv_mat_to_json_raw(m);
            auto mr = json_to_cv_mat_raw(jr);
            if (!mats_equal_exact(m, mr))
                throw std::runtime_error("RAW small-size fail: depth=" + std::to_string(depth) +
                                         " ch=" + std::to_string(ch) +
                                         " size=" + std::to_string(r) + "x" + std::to_string(c));
        }
    };
    run_sizes(CV_8U, 1);
    run_sizes(CV_8U, 3);
    run_sizes(CV_16U, 1);
}

static void test_stringify_parse_roundtrip() {
    {
        cv::Mat m(7, 5, CV_16UC1);
        fill_int_pattern(m);
        auto j = cv_mat_to_json(m);
        std::string s = j.dump();
        nlohmann::json j2 = nlohmann::json::parse(s);
        auto m2 = json_to_cv_mat(j2);
        if (!mats_equal_exact(m, m2))
            throw std::runtime_error("B64 stringify/parse fail");
    }
    {
        cv::Mat m(6, 4, CV_32FC3);
        fill_float_pattern(m);
        auto j = cv_mat_to_json_raw(m);
        std::string s = j.dump();
        nlohmann::json j2 = nlohmann::json::parse(s);
        auto m2 = json_to_cv_mat_raw(j2);
        if (!mats_equal_exact(m, m2))
            throw std::runtime_error("RAW stringify/parse fail");
    }
}

static void test_negative_cases() {
    // Wrong dtype
    {
        cv::Mat m(2, 2, CV_8UC1);
        auto j = cv_mat_to_json(m);
        j["dtype"] = "u8_bad";
        bool threw = false;
        try {
            (void)json_to_cv_mat(j);
        } catch (...) {
            threw = true;
        }
        if (!threw)
            throw std::runtime_error("Expected throw on bad dtype");
    }
    // Shape/data mismatch (RAW)
    {
        cv::Mat m(2, 2, CV_16UC1);
        auto j = cv_mat_to_json_raw(m);
        // remove one element
        if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
            j["data"].erase(j["data"].begin());
        }
        bool threw = false;
        try {
            (void)json_to_cv_mat_raw(j);
        } catch (...) {
            threw = true;
        }
        if (!threw)
            throw std::runtime_error("Expected throw on raw size mismatch");
    }
    // Corrupted Base64
    {
        cv::Mat m(3, 3, CV_8UC1);
        auto j = cv_mat_to_json(m);
        std::string s = j["data"].get<std::string>();
        if (!s.empty())
            s[0] = '#'; // invalid char
        j["data"] = s;
        bool failed = false;
        try {
            (void)json_to_cv_mat(j);
        } catch (...) {
            failed = true;
        }
        if (!failed)
            throw std::runtime_error("Expected failure on bad base64");
    }
}

static void test_explicit_noncontiguous_roi() {
    // ROI with different step than width*elemSize to force non-contiguous copy
    cv::Mat m(10, 10, CV_8UC3);
    fill_int_pattern(m);
    cv::Mat stepped(m, cv::Rect(1, 0, 8, 10));
    auto j = cv_mat_to_json(stepped);
    auto d = json_to_cv_mat(j);
    if (!mats_equal_exact(m(cv::Rect(1, 0, 8, 10)), d))
        throw std::runtime_error("B64 stepped ROI fail");
}

static void test_medium_large() {
    // Keep moderate to avoid long runtimes but large enough to test chunking
    cv::Mat m(480, 640, CV_8UC1);
    fill_int_pattern(m);
    auto j = cv_mat_to_json(m);
    auto d = json_to_cv_mat(j);
    if (!mats_equal_exact(m, d))
        throw std::runtime_error("B64 medium-large fail");
}

// ===============================
// Test functions for JSON helpers
// ===============================

static void test_safe_json_get() {
    nlohmann::json j = {{"int_val", 42},
                        {"float_val", 3.14},
                        {"string_val", "hello"},
                        {"null_val", nullptr},
                        {"missing_key", "exists"}};

    // Test existing keys
    if (safe_json_get<int>(j, "int_val", -1) != 42)
        throw std::runtime_error("safe_json_get int failed");
    if (std::abs(safe_json_get<double>(j, "float_val", -1.0) - 3.14) > 1e-6)
        throw std::runtime_error("safe_json_get float failed");
    if (safe_json_get<std::string>(j, "string_val", "") != "hello")
        throw std::runtime_error("safe_json_get string failed");

    // Test null values
    if (safe_json_get<int>(j, "null_val", -1) != -1)
        throw std::runtime_error("safe_json_get null failed");

    // Test missing keys
    if (safe_json_get<int>(j, "nonexistent", 99) != 99)
        throw std::runtime_error("safe_json_get missing key failed");

    // Test type conversion error handling
    if (safe_json_get<int>(j, "string_val", 99) != 99)
        throw std::runtime_error("safe_json_get type error handling failed");
}

static void test_safe_json_get_array() {
    nlohmann::json j = {{"int_array", {1, 2, 3, 4, 5}},      {"float_array", {1.1, 2.2, 3.3}},
                        {"string_array", {"a", "b", "c"}},   {"null_array", nullptr},
                        {"string_encoded", "\"[1,2,3,4]\""}, // double-encoded JSON
                        {"invalid_string", "not_json"},      {"missing_key", "exists"}};

    // Test regular arrays
    auto int_arr = safe_json_get_array<int>(j, "int_array");
    if (int_arr.size() != 5 || int_arr[0] != 1 || int_arr[4] != 5)
        throw std::runtime_error("safe_json_get_array int failed");

    auto float_arr = safe_json_get_array<double>(j, "float_array");
    if (float_arr.size() != 3 || std::abs(float_arr[0] - 1.1) > 1e-6)
        throw std::runtime_error("safe_json_get_array float failed");

    auto string_arr = safe_json_get_array<std::string>(j, "string_array");
    if (string_arr.size() != 3 || string_arr[0] != "a")
        throw std::runtime_error("safe_json_get_array string failed");

    // Test null arrays
    auto null_arr = safe_json_get_array<int>(j, "null_array");
    if (!null_arr.empty())
        throw std::runtime_error("safe_json_get_array null failed");

    // Test double-encoded JSON strings
    auto encoded_arr = safe_json_get_array<int>(j, "string_encoded");
    if (encoded_arr.size() != 4 || encoded_arr[0] != 1 || encoded_arr[3] != 4)
        throw std::runtime_error("safe_json_get_array encoded string failed");

    // Test invalid JSON strings
    auto invalid_arr = safe_json_get_array<int>(j, "invalid_string");
    if (!invalid_arr.empty())
        throw std::runtime_error("safe_json_get_array invalid string failed");

    // Test missing keys
    auto missing_arr = safe_json_get_array<int>(j, "nonexistent");
    if (!missing_arr.empty())
        throw std::runtime_error("safe_json_get_array missing key failed");
}

static void test_safe_json_get_array_nullable_int() {
    nlohmann::json j = {{"normal_array", {1, 2, 3, 4}},
                        {"nullable_array", {1, nullptr, 3, nullptr}},
                        {"mixed_types", {1, 2.5, nullptr, "invalid"}},
                        {"string_encoded", "\"[1,null,3,null]\""},
                        {"null_array", nullptr},
                        {"missing_key", "exists"}};

    // Test normal array
    auto normal = safe_json_get_array_nullable_int(j, "normal_array");
    if (normal.size() != 4 || normal[0] != 1 || normal[3] != 4)
        throw std::runtime_error("safe_json_get_array_nullable_int normal failed");

    // Test nullable array
    auto nullable = safe_json_get_array_nullable_int(j, "nullable_array", -1);
    if (nullable.size() != 4 || nullable[0] != 1 || nullable[1] != -1 || nullable[2] != 3 ||
        nullable[3] != -1)
        throw std::runtime_error("safe_json_get_array_nullable_int nullable failed");

    // Test mixed types
    auto mixed = safe_json_get_array_nullable_int(j, "mixed_types", -1);
    if (mixed.size() != 4 || mixed[0] != 1 || mixed[1] != 2 || mixed[2] != -1 || mixed[3] != -1)
        throw std::runtime_error("safe_json_get_array_nullable_int mixed failed");

    // Test encoded string
    auto encoded = safe_json_get_array_nullable_int(j, "string_encoded", -1);
    if (encoded.size() != 4 || encoded[0] != 1 || encoded[1] != -1 || encoded[2] != 3 ||
        encoded[3] != -1)
        throw std::runtime_error("safe_json_get_array_nullable_int encoded failed");

    // Test null and missing
    auto null_arr = safe_json_get_array_nullable_int(j, "null_array");
    if (!null_arr.empty())
        throw std::runtime_error("safe_json_get_array_nullable_int null failed");

    auto missing = safe_json_get_array_nullable_int(j, "nonexistent");
    if (!missing.empty())
        throw std::runtime_error("safe_json_get_array_nullable_int missing failed");
}

static void test_safe_parse_vector3d() {
    nlohmann::json j = {{"vector3f", {1.0f, 2.0f, 3.0f}},
                        {"vector3d", {1.0, 2.0, 3.0}},
                        {"string_encoded", "\"[1.0,2.0,3.0]\""},
                        {"wrong_size", {1.0, 2.0}}, // only 2 elements
                        {"null_vector", nullptr},
                        {"missing_key", "exists"}};

    // Test float vector
    auto vec3f = safe_parse_vector3d<float>(j, "vector3f");
    if (vec3f.size() != 3 || std::abs(vec3f[0] - 1.0f) > 1e-6 || std::abs(vec3f[1] - 2.0f) > 1e-6 ||
        std::abs(vec3f[2] - 3.0f) > 1e-6)
        throw std::runtime_error("safe_parse_vector3d float failed");

    // Test double vector
    auto vec3d = safe_parse_vector3d<double>(j, "vector3d");
    if (vec3d.size() != 3 || std::abs(vec3d[0] - 1.0) > 1e-6 || std::abs(vec3d[1] - 2.0) > 1e-6 ||
        std::abs(vec3d[2] - 3.0) > 1e-6)
        throw std::runtime_error("safe_parse_vector3d double failed");

    // Test encoded string
    auto encoded = safe_parse_vector3d<float>(j, "string_encoded");
    if (encoded.size() != 3 || std::abs(encoded[0] - 1.0f) > 1e-6)
        throw std::runtime_error("safe_parse_vector3d encoded failed");

    // Test wrong size (should return zero vector)
    auto wrong_size = safe_parse_vector3d<float>(j, "wrong_size");
    if (wrong_size != pyslam::Vec3f::Zero())
        throw std::runtime_error("safe_parse_vector3d wrong size failed");

    // Test null and missing (should return zero vector)
    auto null_vec = safe_parse_vector3d<float>(j, "null_vector");
    if (null_vec != pyslam::Vec3f::Zero())
        throw std::runtime_error("safe_parse_vector3d null failed");

    auto missing_vec = safe_parse_vector3d<float>(j, "nonexistent");
    if (missing_vec != pyslam::Vec3f::Zero())
        throw std::runtime_error("safe_parse_vector3d missing failed");
}

static void test_safe_parse_keypoints() {
    nlohmann::json j = {{"keypoints", {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}},
                        {"string_encoded", "\"[[1.0,2.0],[3.0,4.0],[5.0,6.0]]\""},
                        {"incomplete_kp", {{1.0f, 2.0f}, {3.0f}}}, // second kp has only 1 element
                        {"null_keypoints", nullptr},
                        {"missing_key", "exists"}};

    // Test normal keypoints
    auto kps = safe_parse_keypoints<float>(j, "keypoints");
    if (kps.rows() != 3 || kps.cols() != 2 || std::abs(kps(0, 0) - 1.0f) > 1e-6 ||
        std::abs(kps(0, 1) - 2.0f) > 1e-6 || std::abs(kps(2, 0) - 5.0f) > 1e-6 ||
        std::abs(kps(2, 1) - 6.0f) > 1e-6)
        throw std::runtime_error("safe_parse_keypoints normal failed");

    // Test encoded string
    auto encoded_kps = safe_parse_keypoints<float>(j, "string_encoded");
    if (encoded_kps.rows() != 3 || encoded_kps.cols() != 2)
        throw std::runtime_error("safe_parse_keypoints encoded failed");

    // Test incomplete keypoints (should handle gracefully)
    auto incomplete_kps = safe_parse_keypoints<float>(j, "incomplete_kp");
    if (incomplete_kps.rows() != 2 || incomplete_kps.cols() != 2)
        throw std::runtime_error("safe_parse_keypoints incomplete failed");

    // Test null and missing (should return empty matrix)
    auto null_kps = safe_parse_keypoints<float>(j, "null_keypoints");
    if (null_kps.rows() != 0 || null_kps.cols() != 2)
        throw std::runtime_error("safe_parse_keypoints null failed");

    auto missing_kps = safe_parse_keypoints<float>(j, "nonexistent");
    if (missing_kps.rows() != 0 || missing_kps.cols() != 2)
        throw std::runtime_error("safe_parse_keypoints missing failed");
}

// ===============================
// Test functions for Eigen helpers
// ===============================

static void test_eigen_matrix_to_json_array() {
    // Test 2x2 matrix
    Eigen::Matrix2f mat2f;
    mat2f << 1.0f, 2.0f, 3.0f, 4.0f;
    auto json2f = eigen_matrix_to_json_array(mat2f);
    if (!json2f.is_array() || json2f.size() != 2 || json2f[0].size() != 2 ||
        json2f[1].size() != 2 || json2f[0][0].get<float>() != 1.0f ||
        json2f[1][1].get<float>() != 4.0f)
        throw std::runtime_error("eigen_matrix_to_json_array 2x2 failed");

    // Test 3x3 matrix
    Eigen::Matrix3d mat3d;
    mat3d << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    auto json3d = eigen_matrix_to_json_array(mat3d);
    if (!json3d.is_array() || json3d.size() != 3 || json3d[0].size() != 3 ||
        json3d[2].size() != 3 || json3d[0][0].get<double>() != 1.0 ||
        json3d[2][2].get<double>() != 9.0)
        throw std::runtime_error("eigen_matrix_to_json_array 3x3 failed");

    // Test vector (treated as column matrix)
    Eigen::Vector3f vec3f(1.0f, 2.0f, 3.0f);
    auto json_vec = eigen_matrix_to_json_array(vec3f);
    if (!json_vec.is_array() || json_vec.size() != 3 || json_vec[0].size() != 1 ||
        json_vec[2].size() != 1 || json_vec[0][0].get<float>() != 1.0f ||
        json_vec[2][0].get<float>() != 3.0f)
        throw std::runtime_error("eigen_matrix_to_json_array vector failed");
}

static void test_json_array_to_eigen_matrix() {
    // Test 2x2 matrix
    nlohmann::json json2f = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Eigen::Matrix2f mat2f;
    if (!json_array_to_eigen_matrix<float, 2, 2>(json2f, mat2f))
        throw std::runtime_error("json_array_to_eigen_matrix 2x2 conversion failed");
    if (std::abs(mat2f(0, 0) - 1.0f) > 1e-6 || std::abs(mat2f(1, 1) - 4.0f) > 1e-6)
        throw std::runtime_error("json_array_to_eigen_matrix 2x2 values failed");

    // Test 3x3 matrix
    nlohmann::json json3d = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    Eigen::Matrix3d mat3d;
    if (!json_array_to_eigen_matrix<double, 3, 3>(json3d, mat3d))
        throw std::runtime_error("json_array_to_eigen_matrix 3x3 conversion failed");
    if (std::abs(mat3d(0, 0) - 1.0) > 1e-6 || std::abs(mat3d(2, 2) - 9.0) > 1e-6)
        throw std::runtime_error("json_array_to_eigen_matrix 3x3 values failed");

    // Test vector
    nlohmann::json json_vec = {{1.0f}, {2.0f}, {3.0f}};
    Eigen::Vector3f vec3f;
    if (!json_array_to_eigen_matrix<float, 3, 1>(json_vec, vec3f))
        throw std::runtime_error("json_array_to_eigen_matrix vector conversion failed");
    if (std::abs(vec3f[0] - 1.0f) > 1e-6 || std::abs(vec3f[2] - 3.0f) > 1e-6)
        throw std::runtime_error("json_array_to_eigen_matrix vector values failed");

    // Test failure cases
    nlohmann::json wrong_size = {{1.0f, 2.0f}, {3.0f}}; // wrong number of columns
    Eigen::Matrix2f mat_fail;
    if (json_array_to_eigen_matrix<float, 2, 2>(wrong_size, mat_fail))
        throw std::runtime_error("json_array_to_eigen_matrix should fail on wrong size");

    nlohmann::json not_array = "not_an_array";
    if (json_array_to_eigen_matrix<float, 2, 2>(not_array, mat_fail))
        throw std::runtime_error("json_array_to_eigen_matrix should fail on non-array");
}

static void test_flexible_json_to_eigen_matrix() {
    // Test regular nested array
    nlohmann::json regular = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Eigen::Matrix2f mat_regular;
    if (!flexible_json_to_eigen_matrix<float, 2, 2>(regular, mat_regular))
        throw std::runtime_error("flexible_json_to_eigen_matrix regular failed");
    if (std::abs(mat_regular(0, 0) - 1.0f) > 1e-6)
        throw std::runtime_error("flexible_json_to_eigen_matrix regular values failed");

    // Test double-encoded string
    nlohmann::json encoded = "\"[[1.0,2.0],[3.0,4.0]]\"";
    Eigen::Matrix2f mat_encoded;
    if (!flexible_json_to_eigen_matrix<float, 2, 2>(encoded, mat_encoded))
        throw std::runtime_error("flexible_json_to_eigen_matrix encoded failed");
    if (std::abs(mat_encoded(0, 0) - 1.0f) > 1e-6)
        throw std::runtime_error("flexible_json_to_eigen_matrix encoded values failed");

    // Test null (should fail)
    nlohmann::json null_val = nullptr;
    Eigen::Matrix2f mat_null;
    if (flexible_json_to_eigen_matrix<float, 2, 2>(null_val, mat_null))
        throw std::runtime_error("flexible_json_to_eigen_matrix should fail on null");

    // Test invalid string (should fail)
    nlohmann::json invalid = "not_valid_json";
    Eigen::Matrix2f mat_invalid;
    if (flexible_json_to_eigen_matrix<float, 2, 2>(invalid, mat_invalid))
        throw std::runtime_error("flexible_json_to_eigen_matrix should fail on invalid string");
}

// ===============================
// Test functions for CV::Mat array functions
// ===============================

static void test_cv_mat_to_json_array() {
    // Test single channel matrix
    cv::Mat mat1c(2, 3, CV_32F);
    mat1c.at<float>(0, 0) = 1.0f;
    mat1c.at<float>(0, 1) = 2.0f;
    mat1c.at<float>(0, 2) = 3.0f;
    mat1c.at<float>(1, 0) = 4.0f;
    mat1c.at<float>(1, 1) = 5.0f;
    mat1c.at<float>(1, 2) = 6.0f;

    auto json1c = cv_mat_to_json_array(mat1c);
    if (!json1c.is_array() || json1c.size() != 6) // flattened to 1D
        throw std::runtime_error("cv_mat_to_json_array single channel size failed");
    if (json1c[0].get<float>() != 1.0f || json1c[5].get<float>() != 6.0f)
        throw std::runtime_error("cv_mat_to_json_array single channel values failed");

    // Test 3-channel matrix
    cv::Mat mat3c(2, 2, CV_32FC3);
    cv::Vec3f pixel1(1.0f, 2.0f, 3.0f);
    cv::Vec3f pixel2(4.0f, 5.0f, 6.0f);
    mat3c.at<cv::Vec3f>(0, 0) = pixel1;
    mat3c.at<cv::Vec3f>(0, 1) = pixel2;
    mat3c.at<cv::Vec3f>(1, 0) = pixel2;
    mat3c.at<cv::Vec3f>(1, 1) = pixel1;

    auto json3c = cv_mat_to_json_array(mat3c);
    if (!json3c.is_array() || json3c.size() != 2) // 2 rows
        throw std::runtime_error("cv_mat_to_json_array 3-channel size failed");
    if (!json3c[0].is_array() || json3c[0].size() != 2) // 2 columns
        throw std::runtime_error("cv_mat_to_json_array 3-channel row size failed");
    if (!json3c[0][0].is_array() || json3c[0][0].size() != 3) // 3 channels
        throw std::runtime_error("cv_mat_to_json_array 3-channel pixel size failed");
    if (json3c[0][0][0].get<float>() != 1.0f || json3c[0][0][2].get<float>() != 3.0f)
        throw std::runtime_error("cv_mat_to_json_array 3-channel values failed");

    // Test empty matrix
    cv::Mat empty;
    auto json_empty = cv_mat_to_json_array(empty);
    if (!json_empty.is_null())
        throw std::runtime_error("cv_mat_to_json_array empty should be null");
}

static void test_json_array_to_cv_mat() {
    // Test single channel array
    nlohmann::json json1c = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto mat1c = json_array_to_cv_mat(json1c);
    if (mat1c.rows != 6 || mat1c.cols != 1 || mat1c.channels() != 1)
        throw std::runtime_error("json_array_to_cv_mat single channel size failed");
    if (std::abs(mat1c.at<float>(0) - 1.0f) > 1e-6 || std::abs(mat1c.at<float>(5) - 6.0f) > 1e-6)
        throw std::runtime_error("json_array_to_cv_mat single channel values failed");

    // Test 2D single channel array
    nlohmann::json json2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto mat2d = json_array_to_cv_mat(json2d);
    if (mat2d.rows != 2 || mat2d.cols != 2 || mat2d.channels() != 1)
        throw std::runtime_error("json_array_to_cv_mat 2D size failed");
    if (std::abs(mat2d.at<float>(0, 0) - 1.0f) > 1e-6 ||
        std::abs(mat2d.at<float>(1, 1) - 4.0f) > 1e-6)
        throw std::runtime_error("json_array_to_cv_mat 2D values failed");

    // Test 3-channel array
    nlohmann::json json3c = {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}}};
    auto mat3c = json_array_to_cv_mat(json3c);
    if (mat3c.rows != 2 || mat3c.cols != 2 || mat3c.channels() != 3)
        throw std::runtime_error("json_array_to_cv_mat 3-channel size failed");
    cv::Vec3f pixel = mat3c.at<cv::Vec3f>(0, 0);
    if (std::abs(pixel[0] - 1.0f) > 1e-6 || std::abs(pixel[2] - 3.0f) > 1e-6)
        throw std::runtime_error("json_array_to_cv_mat 3-channel values failed");

    // Test null and empty arrays
    nlohmann::json null_json = nullptr;
    auto mat_null = json_array_to_cv_mat(null_json);
    if (!mat_null.empty())
        throw std::runtime_error("json_array_to_cv_mat null should be empty");

    nlohmann::json empty_json = nlohmann::json::array();
    auto mat_empty = json_array_to_cv_mat(empty_json);
    if (!mat_empty.empty())
        throw std::runtime_error("json_array_to_cv_mat empty should be empty");

    // Test error cases
    nlohmann::json not_array = "not_an_array";
    bool threw = false;
    try {
        (void)json_array_to_cv_mat(not_array);
    } catch (const std::exception &) {
        threw = true;
    }
    if (!threw)
        throw std::runtime_error("json_array_to_cv_mat should throw on non-array");
}

// ===============================
// Test functions for utility functions
// ===============================

static void test_cv_depth_to_numpy_dtype() {
    if (std::string(cv_depth_to_numpy_dtype(CV_8U)) != "uint8")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_8U failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_8S)) != "int8")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_8S failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_16U)) != "uint16")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_16U failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_16S)) != "int16")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_16S failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_32S)) != "int32")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_32S failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_32F)) != "float32")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_32F failed");
    if (std::string(cv_depth_to_numpy_dtype(CV_64F)) != "float64")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_64F failed");
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    if (std::string(cv_depth_to_numpy_dtype(CV_16F)) != "float16")
        throw std::runtime_error("cv_depth_to_numpy_dtype CV_16F failed");
#endif
    if (std::string(cv_depth_to_numpy_dtype(999)) != "uint8") // fallback
        throw std::runtime_error("cv_depth_to_numpy_dtype fallback failed");
}

// ===============================
// Integration tests
// ===============================

static void test_eigen_roundtrip() {
    // Test Eigen matrix roundtrip
    Eigen::Matrix3f original;
    original << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;

    auto json = eigen_matrix_to_json_array(original);
    Eigen::Matrix3f reconstructed;
    if (!json_array_to_eigen_matrix<float, 3, 3>(json, reconstructed))
        throw std::runtime_error("Eigen roundtrip conversion failed");

    if (!original.isApprox(reconstructed, 1e-6f))
        throw std::runtime_error("Eigen roundtrip values failed");
}

static void test_cv_mat_array_roundtrip() {
    // Test CV::Mat array roundtrip
    cv::Mat original(3, 3, CV_32FC3);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cv::Vec3f pixel(r * 3.0f + c, (r * 3.0f + c) + 1.0f, (r * 3.0f + c) + 2.0f);
            original.at<cv::Vec3f>(r, c) = pixel;
        }
    }

    auto json = cv_mat_to_json_array(original);
    auto reconstructed = json_array_to_cv_mat(json);

    if (original.size() != reconstructed.size() || original.type() != reconstructed.type())
        throw std::runtime_error("CV::Mat array roundtrip size/type failed");

    // Compare values
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cv::Vec3f orig_pixel = original.at<cv::Vec3f>(r, c);
            cv::Vec3f recon_pixel = reconstructed.at<cv::Vec3f>(r, c);
            for (int ch = 0; ch < 3; ++ch) {
                if (std::abs(orig_pixel[ch] - recon_pixel[ch]) > 1e-6f)
                    throw std::runtime_error("CV::Mat array roundtrip values failed");
            }
        }
    }
}

// =================================================================================================
// Main
// =================================================================================================

int main() {
    try {
        test_empty();

        const std::vector<int> depths = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};
        const std::vector<int> chans = {1, 3};

        // Base64 path
        for (int d : depths)
            for (int c : chans)
                test_roundtrip_b64(d, c);

        // Raw numeric path
        for (int d : depths)
            for (int c : chans)
                test_roundtrip_raw(d, c);

#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
        for (int c : chans) {
            test_roundtrip_b64(CV_16F, c);
            test_roundtrip_raw(CV_16F, c);
        }
#endif

        // Additional coverage
        test_channel4_and_small_sizes();
        test_stringify_parse_roundtrip();
        test_negative_cases();
        test_explicit_noncontiguous_roi();
        test_medium_large();

        std::cout << "All cv::Mat serialization tests passed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
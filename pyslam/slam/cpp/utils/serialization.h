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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "eigen_aliases.h"

namespace pyslam {

// ===============================
// JSON helpers
// ===============================

// Helper function to safely parse JSON with error handling
template <typename T>
inline T safe_json_get(const nlohmann::json &j, const std::string &key, const T &default_value) {
    try {
        if (j.contains(key) && !j[key].is_null()) {
            return j[key].get<T>();
        }
    } catch (const std::exception &e) {
        // Log warning if needed, but continue with default value
    }
    return default_value;
}

// Helper function to safely parse optional JSON arrays
template <typename T>
inline std::vector<T> safe_json_get_array(const nlohmann::json &j, const std::string &key) {
    try {
        if (j.contains(key) && !j[key].is_null()) {
            return j[key].get<std::vector<T>>();
        }
    } catch (const std::exception &e) {
        // Return empty vector on error
    }
    return std::vector<T>();
}

// Helper function to safely parse Eigen::Vector3d from JSON
template <typename Scalar>
inline Vec3<Scalar> safe_parse_vector3d(const nlohmann::json &j, const std::string &key) {
    try {
        if (j.contains(key) && !j[key].is_null()) {
            auto vec = j[key].get<std::vector<Scalar>>();
            if (vec.size() == 3) {
                return Vec3<Scalar>(vec[0], vec[1], vec[2]);
            }
        }
    } catch (const std::exception &e) {
        // Return zero vector on error
    }
    return Vec3<Scalar>::Zero();
}

// Helper function to safely parse 2D keypoints from JSON
template <typename Scalar>
inline MatNx2<Scalar> safe_parse_keypoints(const nlohmann::json &j, const std::string &key) {
    try {
        if (j.contains(key) && !j[key].is_null()) {
            auto kps_array = j[key].get<std::vector<std::vector<Scalar>>>();
            MatNx2<Scalar> kps(kps_array.size(), 2);
            for (size_t i = 0; i < kps_array.size(); ++i) {
                if (kps_array[i].size() >= 2) {
                    kps(i, 0) = kps_array[i][0];
                    kps(i, 1) = kps_array[i][1];
                }
            }
            return kps;
        }
    } catch (const std::exception &e) {
        // Return empty matrix on error
    }
    return MatNx2<Scalar>(0, 2);
}

// ===============================
// Base64
// ===============================

// --- Small Base64 encoder (no line breaks) ---
inline std::string base64_encode(const unsigned char *data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(4 * ((len + 2) / 3));
    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t v =
            (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        i += 3;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(tbl[(v >> 6) & 0x3F]);
        out.push_back(tbl[v & 0x3F]);
    }
    if (i < len) {
        uint32_t v = uint32_t(data[i]) << 16;
        if (i + 1 < len)
            v |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        if (i + 1 < len) {
            out.push_back(tbl[(v >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

// Base64 decoder
inline std::string base64_decode(const std::string &encoded) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static int rev_tbl[256];
    static bool initialized = false;

    if (!initialized) {
        std::fill(rev_tbl, rev_tbl + 256, -1);
        for (int i = 0; i < 64; ++i) {
            rev_tbl[static_cast<unsigned char>(tbl[i])] = i;
        }
        initialized = true;
    }

    std::string decoded;
    decoded.reserve(3 * encoded.length() / 4);

    size_t i = 0;
    while (i < encoded.length()) {
        if (i + 4 <= encoded.length()) {
            int a = rev_tbl[static_cast<unsigned char>(encoded[i])];
            int b = rev_tbl[static_cast<unsigned char>(encoded[i + 1])];
            int c = rev_tbl[static_cast<unsigned char>(encoded[i + 2])];
            int d = rev_tbl[static_cast<unsigned char>(encoded[i + 3])];

            if (a >= 0 && b >= 0 && c >= 0 && d >= 0) {
                uint32_t v = (a << 18) | (b << 12) | (c << 6) | d;
                decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
                decoded.push_back(static_cast<char>((v >> 8) & 0xFF));
                decoded.push_back(static_cast<char>(v & 0xFF));
            }
            i += 4;
        } else {
            // Handle padding
            int a = rev_tbl[static_cast<unsigned char>(encoded[i])];
            int b = rev_tbl[static_cast<unsigned char>(encoded[i + 1])];

            if (i + 2 < encoded.length()) {
                int c = rev_tbl[static_cast<unsigned char>(encoded[i + 2])];
                if (a >= 0 && b >= 0 && c >= 0) {
                    uint32_t v = (a << 18) | (b << 12) | (c << 6);
                    decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
                    decoded.push_back(static_cast<char>((v >> 8) & 0xFF));
                }
            } else if (a >= 0 && b >= 0) {
                uint32_t v = (a << 18) | (b << 12);
                decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
            }
            break;
        }
    }

    return decoded;
}

// ===============================
// CV::Mat
// ===============================

inline const char *cv_depth_to_numpy_dtype(int depth) {
    switch (depth) {
    case CV_8U:
        return "uint8";
    case CV_8S:
        return "int8";
    case CV_16U:
        return "uint16";
    case CV_16S:
        return "int16";
    case CV_32S:
        return "int32";
    case CV_32F:
        return "float32";
    case CV_64F:
        return "float64";
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F:
        return "float16";
#endif
    default:
        return "uint8"; // fallback
    }
}

// Helper function to convert cv::Mat to JSON with Base64-encoded raw bytes
inline nlohmann::json cv_mat_to_json(const cv::Mat &mat) {
    using nlohmann::json;

    if (mat.empty()) {
        return nullptr; // json null
    }

    // Ensure contiguous memory
    cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();

    const size_t num_bytes = continuous.total() * continuous.elemSize();
    const unsigned char *ptr = continuous.ptr<unsigned char>(0);

    // Base64 encode raw bytes
    std::string b64 = base64_encode(ptr, num_bytes);

    // Describe array layout for the consumer (e.g., NumPy)
    const int depth = continuous.depth();
    const int channels = continuous.channels();
    const char *dtype = cv_depth_to_numpy_dtype(depth);

    // Shape convention: [rows, cols, channels] (omit channels if 1)
    json shape;
    if (channels == 1) {
        shape = {continuous.rows, continuous.cols};
    } else {
        shape = {continuous.rows, continuous.cols, channels};
    }

    return json{{"type", "npB64"}, {"dtype", dtype}, {"shape", shape}, {"data", b64}};
}

// Helper function to convert JSON back to cv::Mat from Base64-encoded data
inline cv::Mat json_to_cv_mat(const nlohmann::json &json_data) {
    if (json_data.is_null()) {
        return cv::Mat();
    }

    // Check if it's our custom format
    if (!json_data.contains("type") || !json_data.contains("dtype") ||
        !json_data.contains("shape") || !json_data.contains("data")) {
        throw std::runtime_error("Invalid JSON format for cv::Mat");
    }

    std::string type = json_data["type"].get<std::string>();
    if (type != "npB64") {
        throw std::runtime_error("Unsupported JSON type for cv::Mat conversion");
    }

    // Get shape and data type
    auto shape = json_data["shape"].get<std::vector<int>>();
    std::string dtype = json_data["dtype"].get<std::string>();
    std::string b64_data = json_data["data"].get<std::string>();

    // Convert base64 back to binary
    std::string binary_data = base64_decode(b64_data);

    // Determine OpenCV data type
    int cv_depth;
    if (dtype == "uint8") {
        cv_depth = CV_8U;
    } else if (dtype == "int8") {
        cv_depth = CV_8S;
    } else if (dtype == "uint16") {
        cv_depth = CV_16U;
    } else if (dtype == "int16") {
        cv_depth = CV_16S;
    } else if (dtype == "int32") {
        cv_depth = CV_32S;
    } else if (dtype == "float32") {
        cv_depth = CV_32F;
    } else if (dtype == "float64") {
        cv_depth = CV_64F;
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    } else if (dtype == "float16") {
        cv_depth = CV_16F;
#endif
    } else {
        throw std::runtime_error("Unsupported data type: " + dtype);
    }

    // Determine channels
    int channels = 1;
    if (shape.size() == 3) {
        channels = shape[2];
    }

    // Create cv::Mat
    int rows = shape[0];
    int cols = shape[1];

    cv::Mat mat(rows, cols, CV_MAKETYPE(cv_depth, channels));

    // Copy data
    std::memcpy(mat.data, binary_data.data(), binary_data.size());

    return mat;
}

} // namespace pyslam
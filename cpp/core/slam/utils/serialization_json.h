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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "base64_codec.h"
#include "eigen_aliases.h"

namespace pyslam {

// Forward declarations for reusable parsing functions
cv::Mat json_array_to_cv_mat(const nlohmann::json &json_data);
cv::Mat json_to_cv_mat(const nlohmann::json &json_data);
cv::Mat json_to_cv_mat_raw(const nlohmann::json &json_data);

// ===============================
// Eigen helpers
// ===============================

// Convert a fixed-size Eigen matrix to nested JSON array [[...], ...]
template <typename Derived>
inline nlohmann::json eigen_matrix_to_json_array(const Eigen::MatrixBase<Derived> &mat) {
    using nlohmann::json;
    json rows = json::array();
    for (int i = 0; i < mat.rows(); ++i) {
        json row = json::array();
        for (int j = 0; j < mat.cols(); ++j) {
            row.push_back(mat(i, j));
        }
        rows.push_back(row);
    }
    return rows;
}

// Parse nested JSON array into a fixed-size Eigen matrix. Returns false on failure
template <typename Scalar, int Rows, int Cols>
inline bool json_array_to_eigen_matrix(const nlohmann::json &jarr,
                                       Eigen::Matrix<Scalar, Rows, Cols> &out) {
    try {
        if (!jarr.is_array() || static_cast<int>(jarr.size()) != Rows)
            return false;
        for (int i = 0; i < Rows; ++i) {
            const auto &row = jarr[i];
            if (!row.is_array() || static_cast<int>(row.size()) != Cols)
                return false;
            for (int j = 0; j < Cols; ++j) {
                out(i, j) = static_cast<Scalar>(row[j].get<double>());
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

// Parse either a nested array or a JSON-encoded string of nested array into Eigen matrix
// This gracefully handles Python's double-encoded pose string case
template <typename Scalar, int Rows, int Cols>
inline bool flexible_json_to_eigen_matrix(const nlohmann::json &jval,
                                          Eigen::Matrix<Scalar, Rows, Cols> &out) {
    if (jval.is_null())
        return false;
    if (jval.is_string()) {
        // Double-encoded: first parse the string into JSON, then parse array
        try {
            auto inner = nlohmann::json::parse(jval.get<std::string>());
            return json_array_to_eigen_matrix<Scalar, Rows, Cols>(inner, out);
        } catch (...) {
            return false;
        }
    }
    // Regular nested array
    return json_array_to_eigen_matrix<Scalar, Rows, Cols>(jval, out);
}

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
            nlohmann::json array_data = j[key];

            // Handle case where Python sends JSON-encoded string (double-encoded)
            if (array_data.is_string()) {
                try {
                    array_data = nlohmann::json::parse(array_data.get<std::string>());
                } catch (const std::exception &e) {
                    // If parsing fails, return empty vector
                    return std::vector<T>();
                }
            }

            return array_data.get<std::vector<T>>();
        }
    } catch (const std::exception &e) {
        // Return empty vector on error
    }
    return std::vector<T>();
}

// Helper function to safely parse optional JSON arrays of integers with null sentinel
inline std::vector<int> safe_json_get_array_nullable_int(const nlohmann::json &j,
                                                         const std::string &key,
                                                         int null_sentinel = -1) {
    try {
        if (!j.contains(key) || j[key].is_null()) {
            return {};
        }
        nlohmann::json arr = j[key];
        if (arr.is_string()) {
            try {
                arr = nlohmann::json::parse(arr.get<std::string>());
            } catch (...) {
                return {};
            }
        }
        if (!arr.is_array()) {
            return {};
        }
        std::vector<int> out;
        out.reserve(arr.size());
        for (const auto &v : arr) {
            if (v.is_null()) {
                out.push_back(null_sentinel);
            } else if (v.is_number_integer()) {
                out.push_back(v.get<int>());
            } else if (v.is_number()) {
                out.push_back(static_cast<int>(v.get<double>()));
            } else {
                // unexpected type, treat as missing
                out.push_back(null_sentinel);
            }
        }
        return out;
    } catch (const std::exception &e) {
        // Return empty vector on error
    }
    return {};
}

// Helper function to safely parse Eigen::Vector3d from JSON - handles both raw arrays and
// JSON-encoded strings
template <typename Scalar>
inline Vec3<Scalar> safe_parse_vector3d(const nlohmann::json &j, const std::string &key) {
    try {
        if (!j.contains(key) || j[key].is_null()) {
            return Vec3<Scalar>::Zero();
        }
        if (j[key].is_string()) {
            // legacy: Python stored a JSON-encoded string
            auto inner = nlohmann::json::parse(j[key].get<std::string>());
            auto vec = inner.get<std::vector<Scalar>>();
            if (vec.size() == 3) {
                return Vec3<Scalar>(vec[0], vec[1], vec[2]);
            }
        } else {
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
            nlohmann::json kps_data = j[key];

            // Handle case where Python sends JSON-encoded string (double-encoded)
            if (kps_data.is_string()) {
                try {
                    kps_data = nlohmann::json::parse(kps_data.get<std::string>());
                } catch (const std::exception &e) {
                    // If parsing fails, return empty matrix
                    return MatNx2<Scalar>(0, 2);
                }
            }

            // Now parse as array of arrays
            auto kps_array = kps_data.get<std::vector<std::vector<Scalar>>>();
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

// Helper function to safely parse pose matrix from JSON - handles both raw arrays and JSON-encoded
// strings
template <typename Scalar, int Rows, int Cols>
inline bool safe_parse_pose_matrix(const nlohmann::json &j, const std::string &key,
                                   Eigen::Matrix<Scalar, Rows, Cols> &matrix) {
    try {
        if (!j.contains(key) || j[key].is_null()) {
            return false;
        }

        nlohmann::json pose_data = j[key];

        // Handle case where Python sends JSON-encoded string (double-encoded)
        if (pose_data.is_string()) {
            try {
                pose_data = nlohmann::json::parse(pose_data.get<std::string>());
            } catch (const std::exception &e) {
                return false;
            }
        }

        return flexible_json_to_eigen_matrix<Scalar, Rows, Cols>(pose_data, matrix);
    } catch (const std::exception &e) {
        return false;
    }
}

// Helper function to safely parse cv::Mat from JSON - handles both base64 and raw arrays
inline cv::Mat safe_parse_cv_mat_flexible(const nlohmann::json &j, const std::string &key) {
    try {
        if (!j.contains(key) || j[key].is_null()) {
            return cv::Mat();
        }

        nlohmann::json mat_data = j[key];

        // Handle case where Python sends JSON-encoded string (double-encoded)
        if (mat_data.is_string()) {
            try {
                // Parse the string to get the inner JSON object
                auto parsed = nlohmann::json::parse(mat_data.get<std::string>());

                // Check format type
                if (parsed.is_object() && parsed.contains("type")) {
                    std::string type = parsed["type"].get<std::string>();
                    if (type == "npB64") {
                        // Base64-encoded format
                        return json_to_cv_mat(parsed);
                    } else if (type == "npRaw") {
                        // Raw numeric array format
                        return json_to_cv_mat_raw(parsed);
                    }
                }
                // Fallback: try as raw array
                return json_array_to_cv_mat(parsed);
            } catch (const std::exception &e) {
                // If parsing fails, return empty matrix
                return cv::Mat();
            }
        } else if (mat_data.is_object()) {
            // Direct object - check format type
            if (mat_data.contains("type")) {
                std::string type = mat_data["type"].get<std::string>();
                if (type == "npB64") {
                    return json_to_cv_mat(mat_data);
                } else if (type == "npRaw") {
                    return json_to_cv_mat_raw(mat_data);
                }
            }
            // Try as array format (fallback for unstructured objects)
            return json_array_to_cv_mat(mat_data);
        } else {
            // Direct array
            return json_array_to_cv_mat(mat_data);
        }
    } catch (const std::exception &e) {
        // Return empty matrix on error
        return cv::Mat();
    }
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

// Helper function to normalize numpy dtype string to simple format
// Handles formats like "<u1", "|u1", ">u1", "u1", "uint8" -> "uint8"
inline std::string normalize_numpy_dtype(const std::string &dtype) {
    // If already in simple format, return as-is
    if (dtype == "uint8" || dtype == "int8" || dtype == "uint16" || dtype == "int16" ||
        dtype == "int32" || dtype == "float32" || dtype == "float64" || dtype == "float16") {
        return dtype;
    }

    // Remove endianness prefix (<, >, |, =)
    std::string normalized = dtype;
    if (!normalized.empty() && (normalized[0] == '<' || normalized[0] == '>' ||
                                normalized[0] == '|' || normalized[0] == '=')) {
        normalized = normalized.substr(1);
    }

    // Map numpy dtype codes to standard names
    if (normalized == "u1" || normalized == "uint8") {
        return "uint8";
    } else if (normalized == "i1" || normalized == "int8") {
        return "int8";
    } else if (normalized == "u2" || normalized == "uint16") {
        return "uint16";
    } else if (normalized == "i2" || normalized == "int16") {
        return "int16";
    } else if (normalized == "i4" || normalized == "int32") {
        return "int32";
    } else if (normalized == "f4" || normalized == "float32") {
        return "float32";
    } else if (normalized == "f8" || normalized == "float64") {
        return "float64";
    } else if (normalized == "f2" || normalized == "float16") {
        return "float16";
    }

    // If we can't normalize it, return original (will throw error later)
    return dtype;
}

// ===============================
// Raw encoding-decoding
// ===============================

// Convert cv::Mat to simple JSON array (like Python's tolist())
inline nlohmann::json cv_mat_to_json_array(const cv::Mat &mat) {
    if (mat.empty()) {
        return nullptr;
    }

    nlohmann::json result = nlohmann::json::array();

    // Ensure contiguous memory
    cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();

    if (mat.channels() == 1) {
        // Single channel - flatten to 1D array
        for (int i = 0; i < mat.total(); ++i) {
            if (mat.type() == CV_32F) {
                result.push_back(continuous.at<float>(i));
            } else if (mat.type() == CV_64F) {
                result.push_back(continuous.at<double>(i));
            } else if (mat.type() == CV_32S) {
                result.push_back(continuous.at<int>(i));
            } else if (mat.type() == CV_8U) {
                result.push_back(static_cast<float>(continuous.at<unsigned char>(i)));
            } else {
                throw std::runtime_error("Unsupported cv::Mat type for JSON array conversion");
            }
        }
    } else {
        // Multi-channel - create nested arrays
        for (int r = 0; r < mat.rows; ++r) {
            nlohmann::json row = nlohmann::json::array();
            for (int c = 0; c < mat.cols; ++c) {
                if (mat.channels() == 2) {
                    nlohmann::json pixel = nlohmann::json::array();
                    if (mat.type() == CV_32FC2) {
                        cv::Vec2f pixel_val = continuous.at<cv::Vec2f>(r, c);
                        pixel.push_back(pixel_val[0]);
                        pixel.push_back(pixel_val[1]);
                    } else if (mat.type() == CV_8UC2) {
                        cv::Vec2b pixel_val = continuous.at<cv::Vec2b>(r, c);
                        pixel.push_back(static_cast<float>(pixel_val[0]));
                        pixel.push_back(static_cast<float>(pixel_val[1]));
                    }
                    row.push_back(pixel);
                } else if (mat.channels() == 3) {
                    nlohmann::json pixel = nlohmann::json::array();
                    if (mat.type() == CV_32FC3) {
                        cv::Vec3f pixel_val = continuous.at<cv::Vec3f>(r, c);
                        pixel.push_back(pixel_val[0]);
                        pixel.push_back(pixel_val[1]);
                        pixel.push_back(pixel_val[2]);
                    } else if (mat.type() == CV_8UC3) {
                        cv::Vec3b pixel_val = continuous.at<cv::Vec3b>(r, c);
                        pixel.push_back(static_cast<float>(pixel_val[0]));
                        pixel.push_back(static_cast<float>(pixel_val[1]));
                        pixel.push_back(static_cast<float>(pixel_val[2]));
                    }
                    row.push_back(pixel);
                }
            }
            result.push_back(row);
        }
    }

    return result;
}

// Convert simple JSON array back to cv::Mat
// Handles both 1D and 2D arrays, and tries to infer appropriate data type (uint8 for descriptors,
// float32 otherwise)
inline cv::Mat json_array_to_cv_mat(const nlohmann::json &json_data) {
    if (json_data.is_null()) {
        return cv::Mat();
    }

    if (!json_data.is_array()) {
        throw std::runtime_error("Expected JSON array for cv::Mat conversion");
    }

    if (json_data.empty()) {
        return cv::Mat();
    }

    // Determine dimensions and channels
    int rows = json_data.size();
    int cols = 1;
    int channels = 1;
    bool is_2d = false;

    if (rows > 0 && json_data[0].is_array()) {
        is_2d = true;
        cols = json_data[0].size();
        if (cols > 0 && json_data[0][0].is_array()) {
            channels = json_data[0][0].size();
        }
    }

    // Try to infer data type: if all values are integers in [0, 255], use uint8 (for descriptors)
    // Otherwise use float32
    bool use_uint8 = true;
    if (is_2d && rows > 0 && cols > 0) {
        try {
            for (int r = 0; r < std::min(rows, 10) && use_uint8; ++r) { // Sample first 10 rows
                if (channels == 1) {
                    for (int c = 0; c < std::min(cols, 10) && use_uint8; ++c) {
                        double val = json_data[r][c].get<double>();
                        if (val < 0 || val > 255 || val != std::floor(val)) {
                            use_uint8 = false;
                        }
                    }
                } else {
                    for (int c = 0; c < std::min(cols, 10) && use_uint8; ++c) {
                        for (int ch = 0; ch < channels && use_uint8; ++ch) {
                            double val = json_data[r][c][ch].get<double>();
                            if (val < 0 || val > 255 || val != std::floor(val)) {
                                use_uint8 = false;
                            }
                        }
                    }
                }
            }
        } catch (...) {
            use_uint8 = false;
        }
    } else if (rows > 0) {
        // 1D array
        try {
            for (int i = 0; i < std::min(rows, 10) && use_uint8; ++i) {
                double val = json_data[i].get<double>();
                if (val < 0 || val > 255 || val != std::floor(val)) {
                    use_uint8 = false;
                }
            }
        } catch (...) {
            use_uint8 = false;
        }
    } else {
        use_uint8 = false;
    }

    // Create appropriate cv::Mat
    cv::Mat result;
    if (channels == 1) {
        if (use_uint8) {
            result = cv::Mat::zeros(rows, cols, CV_8U);
            if (is_2d) {
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        result.at<uint8_t>(r, c) = static_cast<uint8_t>(json_data[r][c].get<int>());
                    }
                }
            } else {
                // 1D array
                for (int r = 0; r < rows; ++r) {
                    result.at<uint8_t>(r) = static_cast<uint8_t>(json_data[r].get<int>());
                }
            }
        } else {
            result = cv::Mat::zeros(rows, cols, CV_32F);
            if (is_2d) {
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        result.at<float>(r, c) = json_data[r][c].get<float>();
                    }
                }
            } else {
                // 1D array
                for (int r = 0; r < rows; ++r) {
                    result.at<float>(r) = json_data[r].get<float>();
                }
            }
        }
    } else if (channels == 2) {
        result = cv::Mat::zeros(rows, cols, CV_32FC2);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cv::Vec2f pixel;
                pixel[0] = json_data[r][c][0].get<float>();
                pixel[1] = json_data[r][c][1].get<float>();
                result.at<cv::Vec2f>(r, c) = pixel;
            }
        }
    } else if (channels == 3) {
        result = cv::Mat::zeros(rows, cols, CV_32FC3);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cv::Vec3f pixel;
                pixel[0] = json_data[r][c][0].get<float>();
                pixel[1] = json_data[r][c][1].get<float>();
                pixel[2] = json_data[r][c][2].get<float>();
                result.at<cv::Vec3f>(r, c) = pixel;
            }
        }
    } else {
        throw std::runtime_error("Unsupported number of channels for cv::Mat conversion");
    }

    return result;
}

// Convert cv::Mat to JSON with raw binary data
inline nlohmann::json cv_mat_to_json_raw(const cv::Mat &mat) {
    using nlohmann::json;

    if (mat.empty()) {
        return nullptr; // json null
    }

    // Ensure contiguous memory
    cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();

    const int depth = continuous.depth();
    const int channels = continuous.channels();
    const char *dtype = cv_depth_to_numpy_dtype(depth);

    // Shape: [rows, cols] or [rows, cols, channels]
    json shape;
    if (channels == 1) {
        shape = {continuous.rows, continuous.cols};
    } else {
        shape = {continuous.rows, continuous.cols, channels};
    }

    // Total number of scalars
    const size_t count = static_cast<size_t>(continuous.total()) * static_cast<size_t>(channels);

    // Emit raw numeric data (no base64)
    json data = json::array();

    switch (depth) {
    case CV_8U: {
        const uint8_t *ptr = continuous.ptr<uint8_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_8S: {
        const int8_t *ptr = continuous.ptr<int8_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_16U: {
        const uint16_t *ptr = continuous.ptr<uint16_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_16S: {
        const int16_t *ptr = continuous.ptr<int16_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_32S: {
        const int32_t *ptr = continuous.ptr<int32_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_32F: {
        const float *ptr = continuous.ptr<float>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
    case CV_64F: {
        const double *ptr = continuous.ptr<double>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F: {
        // Store uint16 raw bit-pattern for float16
        const uint16_t *ptr = continuous.ptr<uint16_t>(0);
        for (size_t i = 0; i < count; ++i)
            data.push_back(ptr[i]);
        break;
    }
#endif
    default:
        throw std::runtime_error("Unsupported cv::Mat depth for raw JSON serialization");
    }

    return json{{"type", "npRaw"}, {"dtype", dtype}, {"shape", shape}, {"data", data}};
}

// Convert JSON with raw binary data back to cv::Mat
inline cv::Mat json_to_cv_mat_raw(const nlohmann::json &json_data) {
    if (json_data.is_null()) {
        return cv::Mat();
    }
    if (!json_data.contains("type") || !json_data.contains("dtype") ||
        !json_data.contains("shape") || !json_data.contains("data")) {
        throw std::runtime_error("Invalid JSON format for raw cv::Mat");
    }

    std::string type = json_data["type"].get<std::string>();
    if (type != "npRaw") {
        throw std::runtime_error("Unsupported JSON type for raw cv::Mat conversion");
    }

    auto shape = json_data["shape"].get<std::vector<int>>();
    std::string dtype = json_data["dtype"].get<std::string>();
    const auto &data = json_data["data"];

    // Determine OpenCV depth
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
        throw std::runtime_error("Unsupported dtype for raw cv::Mat: " + dtype);
    }

    // Infer channels
    int channels = 1;
    if (shape.size() == 3) {
        channels = shape[2];
    }
    const int rows = shape[0];
    const int cols = shape[1];

    // Validate element count
    const size_t expected =
        static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(channels);
    if (!data.is_array() || data.size() != expected) {
        throw std::runtime_error("Raw cv::Mat data size mismatch");
    }

    cv::Mat mat(rows, cols, CV_MAKETYPE(cv_depth, channels));

    // Copy from JSON array into Mat according to depth
    switch (cv_depth) {
    case CV_8U: {
        auto *dst = mat.ptr<uint8_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = static_cast<uint8_t>(data[i].get<int>());
        break;
    }
    case CV_8S: {
        auto *dst = mat.ptr<int8_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = static_cast<int8_t>(data[i].get<int>());
        break;
    }
    case CV_16U: {
        auto *dst = mat.ptr<uint16_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = static_cast<uint16_t>(data[i].get<int>());
        break;
    }
    case CV_16S: {
        auto *dst = mat.ptr<int16_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = static_cast<int16_t>(data[i].get<int>());
        break;
    }
    case CV_32S: {
        auto *dst = mat.ptr<int32_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = data[i].get<int32_t>();
        break;
    }
    case CV_32F: {
        auto *dst = mat.ptr<float>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = data[i].get<float>();
        break;
    }
    case CV_64F: {
        auto *dst = mat.ptr<double>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = data[i].get<double>();
        break;
    }
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F: {
        // Treat JSON numbers as uint16 bit patterns for float16
        auto *dst = mat.ptr<uint16_t>(0);
        for (size_t i = 0; i < expected; ++i)
            dst[i] = static_cast<uint16_t>(data[i].get<int>());
        break;
    }
#endif
    default:
        throw std::runtime_error("Unsupported cv::Mat depth for raw JSON deserialization");
    }

    return mat;
}

// ===============================
// Base64 encoding-decoding
// ===============================

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

    // Normalize numpy dtype string (handles formats like "<u1", "|u1", etc.)
    std::string normalized_dtype = normalize_numpy_dtype(dtype);

    // Convert base64 back to binary
    std::string binary_data = base64_decode(b64_data);

    // Determine OpenCV data type
    int cv_depth;
    if (normalized_dtype == "uint8") {
        cv_depth = CV_8U;
    } else if (normalized_dtype == "int8") {
        cv_depth = CV_8S;
    } else if (normalized_dtype == "uint16") {
        cv_depth = CV_16U;
    } else if (normalized_dtype == "int16") {
        cv_depth = CV_16S;
    } else if (normalized_dtype == "int32") {
        cv_depth = CV_32S;
    } else if (normalized_dtype == "float32") {
        cv_depth = CV_32F;
    } else if (normalized_dtype == "float64") {
        cv_depth = CV_64F;
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    } else if (normalized_dtype == "float16") {
        cv_depth = CV_16F;
#endif
    } else {
        throw std::runtime_error("Unsupported data type: " + dtype +
                                 " (normalized: " + normalized_dtype + ")");
    }

    // Determine channels
    int channels = 1;
    if (shape.size() == 3) {
        channels = shape[2];
    }

    // Create cv::Mat
    int rows = shape[0];
    int cols = shape[1];

    // Validate decoded byte count exactly matches expected
    const size_t elem_size1 =
#if CV_VERSION_MAJOR >= 4
        CV_ELEM_SIZE1(cv_depth);
#else
        CV_ELEM_SIZE1(cv_depth);
#endif
    const size_t expected_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                                  static_cast<size_t>(channels) * elem_size1;
    if (binary_data.size() != expected_bytes) {
        throw std::runtime_error("Base64 data size mismatch");
    }

    cv::Mat mat(rows, cols, CV_MAKETYPE(cv_depth, channels));

    // Copy data
    std::memcpy(mat.data, binary_data.data(), binary_data.size());

    return mat;
}

} // namespace pyslam
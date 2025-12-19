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

#include "camera.h"

#include "utils/messages.h"
#include "utils/serialization_json.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace pyslam {

// Helper function to convert SensorType to Python serialization format
static std::string sensor_type_to_json_string(SensorType sensor_type) {
    switch (sensor_type) {
    case SensorType::MONOCULAR:
        return "SensorType.MONOCULAR";
    case SensorType::STEREO:
        return "SensorType.STEREO";
    case SensorType::RGBD:
        return "SensorType.RGBD";
    default:
        return "SensorType.MONOCULAR";
    }
}

// Helper function to parse SensorType from Python serialization format
static SensorType sensor_type_from_json_string(const std::string &sensor_str) {
    if (sensor_str == "SensorType.MONOCULAR" || sensor_str == "MONOCULAR") {
        return SensorType::MONOCULAR;
    } else if (sensor_str == "SensorType.STEREO" || sensor_str == "STEREO") {
        return SensorType::STEREO;
    } else if (sensor_str == "SensorType.RGBD" || sensor_str == "RGBD") {
        return SensorType::RGBD;
    } else {
        // Fallback: try to parse as lowercase string
        return get_sensor_type(sensor_str);
    }
}

std::string Camera::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"type\": " << static_cast<int>(type) << ", ";
    oss << "\"width\": " << width << ", ";
    oss << "\"height\": " << height << ", ";
    oss << "\"fx\": " << fx << ", ";
    oss << "\"fy\": " << fy << ", ";
    oss << "\"cx\": " << cx << ", ";
    oss << "\"cy\": " << cy << ", ";
    oss << "\"D\": [";
    for (size_t i = 0; i < D.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << D[i];
    }
    oss << "], ";
    oss << "\"fps\": " << (fps >= 0 ? std::to_string(fps) : "null") << ", ";
    oss << "\"bf\": " << (bf >= 0 ? std::to_string(bf) : "null") << ", ";
    oss << "\"b\": " << (b >= 0 ? std::to_string(b) : "null") << ", ";
    oss << "\"depth_factor\": " << (depth_factor > 0 ? std::to_string(depth_factor) : "null")
        << ", ";
    if (std::isinf(depth_threshold)) {
        oss << "\"depth_threshold\": null, ";
    } else {
        oss << "\"depth_threshold\": " << depth_threshold << ", ";
    }
    oss << "\"is_distorted\": " << (is_distorted ? "true" : "false") << ", ";
    oss << "\"u_min\": " << u_min << ", ";
    oss << "\"u_max\": " << u_max << ", ";
    oss << "\"v_min\": " << v_min << ", ";
    oss << "\"v_max\": " << v_max << ", ";
    oss << "\"initialized\": " << (initialized ? "true" : "false") << ", ";
    oss << "\"K\": " << eigen_matrix_to_json_array(K).dump() << ", ";
    oss << "\"Kinv\": " << eigen_matrix_to_json_array(Kinv).dump() << ", ";
    oss << "\"sensor_type\": \"" << sensor_type_to_json_string(sensor_type) << "\"";

    oss << "}";
    return oss.str();
}

void Camera::init_from_json(const std::string &json_str) {
    nlohmann::json json_data = nlohmann::json::parse(json_str);

    // Parse basic camera parameters using safe helpers
    type = static_cast<CameraType>(safe_json_get(json_data, "type", 0));
    width = safe_json_get(json_data, "width", 0);
    height = safe_json_get(json_data, "height", 0);
    fx = safe_json_get(json_data, "fx", 0.0);
    fy = safe_json_get(json_data, "fy", 0.0);
    cx = safe_json_get(json_data, "cx", 0.0);
    cy = safe_json_get(json_data, "cy", 0.0);

    // Parse distortion coefficients using safe array helper
    D = safe_json_get_array<double>(json_data, "D");

    // Handle fps - can be null
    if (json_data.contains("fps") && !json_data["fps"].is_null()) {
        fps = safe_json_get(json_data, "fps", 30);
    } else {
        fps = -1; // Use -1 as sentinel for "not set"
    }
    // Handle bf, b, depth_factor - can be null
    if (json_data.contains("bf") && !json_data["bf"].is_null()) {
        bf = safe_json_get(json_data, "bf", -1.0);
    } else {
        bf = -1.0; // Use -1.0 as sentinel for "not set"
    }
    if (json_data.contains("b") && !json_data["b"].is_null()) {
        b = safe_json_get(json_data, "b", -1.0);
    } else {
        b = -1.0; // Use -1.0 as sentinel for "not set"
    }
    if (json_data.contains("depth_factor") && !json_data["depth_factor"].is_null()) {
        depth_factor = safe_json_get(json_data, "depth_factor", 1.0);
    } else {
        depth_factor = 1.0;
    }

    if (json_data["depth_threshold"].is_null()) {
        depth_threshold = std::numeric_limits<double>::infinity();
    } else {
        depth_threshold =
            safe_json_get(json_data, "depth_threshold", std::numeric_limits<double>::infinity());
    }

    is_distorted = safe_json_get(json_data, "is_distorted", false);
    u_min = safe_json_get(json_data, "u_min", 0.0);
    u_max = safe_json_get(json_data, "u_max", static_cast<double>(width));
    v_min = safe_json_get(json_data, "v_min", 0.0);
    v_max = safe_json_get(json_data, "v_max", static_cast<double>(height));
    initialized = safe_json_get(json_data, "initialized", false);

    // Compute field of view if not already set
    if (fovx == 0.0) {
        fovx = focal2fov(fx, width);
    }
    if (fovy == 0.0) {
        fovy = focal2fov(fy, height);
    }

    // Parse intrinsic matrices using flexible helper (handles both arrays and stringified arrays)
    if (!flexible_json_to_eigen_matrix<double, 3, 3>(json_data["K"], K)) {
        throw std::runtime_error("Failed to parse K matrix from JSON");
    }
    if (!flexible_json_to_eigen_matrix<double, 3, 3>(json_data["Kinv"], Kinv)) {
        throw std::runtime_error("Failed to parse Kinv matrix from JSON");
    }

    // Handle sensor_type - if present, deserialize it; otherwise default to MONOCULAR with warning
    sensor_type = SensorType::MONOCULAR;
    if (json_data.contains("sensor_type") && !json_data["sensor_type"].is_null()) {
        try {
            std::string sensor_str = json_data["sensor_type"].get<std::string>();
            sensor_type = sensor_type_from_json_string(sensor_str);
        } catch (const std::exception &) {
            // If deserialization fails, default to MONOCULAR with warning
            MSG_RED_WARN(
                "Camera: sensor_type not found or invalid in JSON, defaulting to MONOCULAR. "
                "Please ensure sensor_type is explicitly saved in future maps.");
        }
    } else {
        // Backward compatibility: default to MONOCULAR with warning
        MSG_RED_WARN("Camera: sensor_type not found in JSON, defaulting to MONOCULAR. "
                     "Please ensure sensor_type is explicitly saved in future maps.");
    }
}

std::string PinholeCamera::to_json() const {
    std::string camera_json = Camera::to_json();
    return camera_json;
}

PinholeCamera PinholeCamera::from_json(const std::string &json_str) {
    nlohmann::json json_data = nlohmann::json::parse(json_str);

    // Create ConfigDict from the flat JSON structure
    ConfigDict config;
    ConfigDict cam_settings;

    // Map flat JSON fields to ConfigDict structure using safe getters
    cam_settings["Camera.width"] = safe_json_get(json_data, "width", 0);
    cam_settings["Camera.height"] = safe_json_get(json_data, "height", 0);
    cam_settings["Camera.fx"] = safe_json_get(json_data, "fx", 0.0);
    cam_settings["Camera.fy"] = safe_json_get(json_data, "fy", 0.0);
    cam_settings["Camera.cx"] = safe_json_get(json_data, "cx", 0.0);
    cam_settings["Camera.cy"] = safe_json_get(json_data, "cy", 0.0);
    cam_settings["Camera.bf"] = safe_json_get(json_data, "bf", 0.0);
    cam_settings["Camera.fps"] = safe_json_get(json_data, "fps", 30);

    // Handle distortion coefficients array by unpacking into individual parameters
    // This matches how the Camera constructor reads them (k1, k2, p1, p2, k3)
    // Handle both direct arrays and stringified arrays (from Python)
    if (json_data.contains("D") && !json_data["D"].is_null()) {
        std::vector<double> D = safe_json_get_array<double>(json_data, "D");
        if (D.size() >= 1)
            cam_settings["Camera.k1"] = D[0];
        if (D.size() >= 2)
            cam_settings["Camera.k2"] = D[1];
        if (D.size() >= 3)
            cam_settings["Camera.p1"] = D[2];
        if (D.size() >= 4)
            cam_settings["Camera.p2"] = D[3];
        if (D.size() >= 5)
            cam_settings["Camera.k3"] = D[4];
    }

    config["cam_settings"] = cam_settings;

    return PinholeCamera(config);
}

// ===============================

// Helper function to create camera from JSON with proper type detection
CameraPtr create_camera_from_json(const nlohmann::json &camera_json) {
    if (camera_json.is_null()) {
        return nullptr;
    }

    try {
        // Check camera type from JSON
        if (camera_json.contains("camera_type")) {
            std::string camera_type = camera_json["camera_type"].get<std::string>();
            if (camera_type == "PinholeCamera") {
                return std::make_shared<PinholeCamera>(
                    PinholeCamera::from_json(camera_json.dump()));
            }
            // Add support for other camera types here as they are implemented
            else {
                throw std::runtime_error("Unsupported camera type: " + camera_type);
            }
        } else {
            // Fallback: assume PinholeCamera if no type specified
            return std::make_shared<PinholeCamera>(PinholeCamera::from_json(camera_json.dump()));
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to create camera from JSON: " + std::string(e.what()));
    }
}

} // namespace pyslam

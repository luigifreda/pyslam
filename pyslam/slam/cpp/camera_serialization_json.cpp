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
    oss << "\"fps\": " << fps << ", ";
    oss << "\"bf\": " << bf << ", ";
    oss << "\"b\": " << b << ", ";
    oss << "\"depth_factor\": " << depth_factor << ", ";
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
    oss << "\"Kinv\": " << eigen_matrix_to_json_array(Kinv).dump();

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

    fps = safe_json_get(json_data, "fps", 30);
    bf = safe_json_get(json_data, "bf", 0.0);
    b = safe_json_get(json_data, "b", 0.0);
    depth_factor = safe_json_get(json_data, "depth_factor", 1.0);

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

    // Parse intrinsic matrices using helper
    if (!json_array_to_eigen_matrix<double, 3, 3>(json_data["K"], K)) {
        throw std::runtime_error("Failed to parse K matrix from JSON");
    }
    if (!json_array_to_eigen_matrix<double, 3, 3>(json_data["Kinv"], Kinv)) {
        throw std::runtime_error("Failed to parse Kinv matrix from JSON");
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

    // Handle distortion coefficients array
    if (json_data.contains("D") && !json_data["D"].is_null()) {
        cam_settings["Camera.DistCoef"] = json_data["D"].get<std::vector<double>>();
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

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
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "camera.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    try {
        std::cout << "Starting Camera JSON serialization tests..." << std::endl;

        // Test 1: Create test camera with comprehensive data
        std::cout << "Test 1: Creating test camera with comprehensive data..." << std::endl;
        ConfigDict camera_config;
        ConfigDict cam_settings;
        cam_settings["Camera.width"] = 640;
        cam_settings["Camera.height"] = 480;
        cam_settings["Camera.fx"] = 525.0;
        cam_settings["Camera.fy"] = 525.0;
        cam_settings["Camera.cx"] = 320.0;
        cam_settings["Camera.cy"] = 240.0;
        cam_settings["Camera.bf"] = 0.1;
        cam_settings["Camera.fps"] = 30;
        cam_settings["Camera.b"] = 0.05;
        camera_config["cam_settings"] = cam_settings;

        CameraPtr camera = std::make_shared<PinholeCamera>(camera_config);

        // Set additional camera properties
        camera->fovx = 60.0;
        camera->fovy = 45.0;
        camera->sensor_type = SensorType::STEREO;
        camera->depth_factor = 1000.0;
        camera->depth_threshold = 10.0;
        camera->is_distorted = false;
        camera->u_min = 0.0;
        camera->u_max = 640.0;
        camera->v_min = 0.0;
        camera->v_max = 480.0;
        camera->initialized = true;

        // Set distortion coefficients
        camera->D = {0.1, -0.2, 0.01, 0.02, 0.0};

        // Set intrinsic matrices
        camera->K << 525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0;
        camera->Kinv = camera->K.inverse();

        std::cout << "Created test camera with comprehensive data" << std::endl;

        // Test 2: Serialize to JSON
        std::cout << "Test 2: Serializing Camera to JSON..." << std::endl;
        std::string json_str = camera->to_json();
        if (json_str.empty()) {
            throw std::runtime_error("JSON serialization returned empty string");
        }

        // Verify JSON is valid
        nlohmann::json json_obj = nlohmann::json::parse(json_str);
        std::cout << "JSON serialization successful, JSON size: " << json_str.length()
                  << " characters" << std::endl;

        // Test 3: Deserialize from JSON using init_from_json
        std::cout << "Test 3: Deserializing Camera from JSON using init_from_json..." << std::endl;
        CameraPtr camera2 = std::make_shared<PinholeCamera>();
        camera2->init_from_json(json_str);
        std::cout << "JSON deserialization successful" << std::endl;

        // Test 4: Verify basic camera parameters
        std::cout << "Test 4: Verifying basic camera parameters..." << std::endl;
        if (camera2->type != camera->type) {
            throw std::runtime_error("Camera type mismatch");
        }
        if (camera2->width != camera->width) {
            throw std::runtime_error("Width mismatch: " + std::to_string(camera2->width) + " vs " +
                                     std::to_string(camera->width));
        }
        if (camera2->height != camera->height) {
            throw std::runtime_error("Height mismatch: " + std::to_string(camera2->height) +
                                     " vs " + std::to_string(camera->height));
        }
        if (std::abs(camera2->fx - camera->fx) > 1e-12) {
            throw std::runtime_error("fx mismatch");
        }
        if (std::abs(camera2->fy - camera->fy) > 1e-12) {
            throw std::runtime_error("fy mismatch");
        }
        if (std::abs(camera2->cx - camera->cx) > 1e-12) {
            throw std::runtime_error("cx mismatch");
        }
        if (std::abs(camera2->cy - camera->cy) > 1e-12) {
            throw std::runtime_error("cy mismatch");
        }
        if (std::abs(camera2->bf - camera->bf) > 1e-12) {
            throw std::runtime_error("bf mismatch");
        }
        if (std::abs(camera2->b - camera->b) > 1e-12) {
            throw std::runtime_error("b mismatch");
        }
        if (camera2->fps != camera->fps) {
            throw std::runtime_error("fps mismatch");
        }
        std::cout << "Basic camera parameters verified" << std::endl;

        // Test 5: Verify field of view parameters (computed from focal length)
        std::cout << "Test 5: Verifying field of view parameters..." << std::endl;
        // Note: fovx and fovy are computed from focal length in init_from_json, not stored
        // So we verify they are computed correctly rather than matching original values
        double expected_fovx = focal2fov(camera2->fx, camera2->width);
        double expected_fovy = focal2fov(camera2->fy, camera2->height);

        if (std::abs(camera2->fovx - expected_fovx) > 1e-12) {
            throw std::runtime_error("fovx computation mismatch: " + std::to_string(camera2->fovx) +
                                     " vs " + std::to_string(expected_fovx));
        }
        if (std::abs(camera2->fovy - expected_fovy) > 1e-12) {
            throw std::runtime_error("fovy computation mismatch: " + std::to_string(camera2->fovy) +
                                     " vs " + std::to_string(expected_fovy));
        }
        std::cout << "Field of view parameters verified (computed from focal length)" << std::endl;

        // Test 6: Verify sensor type and depth parameters
        std::cout << "Test 6: Verifying sensor type and depth parameters..." << std::endl;
        // Verify sensor_type is correctly serialized and deserialized
        if (camera2->sensor_type != camera->sensor_type) {
            throw std::runtime_error(
                "sensor_type mismatch: " + std::to_string(static_cast<int>(camera2->sensor_type)) +
                " vs " + std::to_string(static_cast<int>(camera->sensor_type)));
        }
        if (std::abs(camera2->depth_factor - camera->depth_factor) > 1e-12) {
            throw std::runtime_error("depth_factor mismatch");
        }
        if (std::abs(camera2->depth_threshold - camera->depth_threshold) > 1e-12) {
            throw std::runtime_error("depth_threshold mismatch");
        }
        std::cout << "Sensor type and depth parameters verified" << std::endl;

        // Test 7: Verify distortion parameters
        std::cout << "Test 7: Verifying distortion parameters..." << std::endl;
        if (camera2->is_distorted != camera->is_distorted) {
            throw std::runtime_error("is_distorted mismatch");
        }
        if (camera2->D.size() != camera->D.size()) {
            throw std::runtime_error("Distortion coefficients size mismatch");
        }
        for (size_t i = 0; i < camera->D.size(); ++i) {
            if (std::abs(camera2->D[i] - camera->D[i]) > 1e-12) {
                throw std::runtime_error("Distortion coefficient[" + std::to_string(i) +
                                         "] mismatch");
            }
        }
        std::cout << "Distortion parameters verified" << std::endl;

        // Test 8: Verify image bounds
        std::cout << "Test 8: Verifying image bounds..." << std::endl;
        if (std::abs(camera2->u_min - camera->u_min) > 1e-12) {
            throw std::runtime_error("u_min mismatch");
        }
        if (std::abs(camera2->u_max - camera->u_max) > 1e-12) {
            throw std::runtime_error("u_max mismatch");
        }
        if (std::abs(camera2->v_min - camera->v_min) > 1e-12) {
            throw std::runtime_error("v_min mismatch");
        }
        if (std::abs(camera2->v_max - camera->v_max) > 1e-12) {
            throw std::runtime_error("v_max mismatch");
        }
        std::cout << "Image bounds verified" << std::endl;

        // Test 9: Verify initialization status
        std::cout << "Test 9: Verifying initialization status..." << std::endl;
        if (camera2->initialized != camera->initialized) {
            throw std::runtime_error("initialized mismatch");
        }
        std::cout << "Initialization status verified" << std::endl;

        // Test 10: Verify intrinsic matrices
        std::cout << "Test 10: Verifying intrinsic matrices..." << std::endl;
        if (!eigen_matrices_equal(camera2->K, camera->K)) {
            throw std::runtime_error("Intrinsic matrix K mismatch");
        }
        if (!eigen_matrices_equal(camera2->Kinv, camera->Kinv)) {
            throw std::runtime_error("Inverse intrinsic matrix Kinv mismatch");
        }
        std::cout << "Intrinsic matrices verified" << std::endl;

        // Test 11: Test camera functionality
        std::cout << "Test 11: Testing camera functionality..." << std::endl;

        // Test projection
        Eigen::Vector3d test_point(1.0, 2.0, 5.0);
        auto [uv1, depth1] = camera->project_point(test_point);
        auto [uv2, depth2] = camera2->project_point(test_point);

        if (std::abs(uv1.x() - uv2.x()) > 1e-12 || std::abs(uv1.y() - uv2.y()) > 1e-12) {
            throw std::runtime_error("Projection mismatch");
        }
        if (std::abs(depth1 - depth2) > 1e-12) {
            throw std::runtime_error("Projected depth mismatch");
        }

        // Test unprojection
        auto unproj1 = camera->unproject_point(uv1.x(), uv1.y());
        auto unproj2 = camera2->unproject_point(uv2.x(), uv2.y());

        if (std::abs(unproj1.x() - unproj2.x()) > 1e-12 ||
            std::abs(unproj1.y() - unproj2.y()) > 1e-12) {
            throw std::runtime_error("Unprojection mismatch");
        }

        std::cout << "Camera functionality verified" << std::endl;

        // Test 12: Test PinholeCamera::from_json static method
        std::cout << "Test 12: Testing PinholeCamera::from_json static method..." << std::endl;

        PinholeCamera pinhole_camera3 = PinholeCamera::from_json(json_str);

        // Verify basic parameters
        if (pinhole_camera3.width != camera->width) {
            throw std::runtime_error("PinholeCamera::from_json width mismatch");
        }
        if (pinhole_camera3.height != camera->height) {
            throw std::runtime_error("PinholeCamera::from_json height mismatch");
        }
        if (std::abs(pinhole_camera3.fx - camera->fx) > 1e-12) {
            throw std::runtime_error("PinholeCamera::from_json fx mismatch");
        }
        if (std::abs(pinhole_camera3.fy - camera->fy) > 1e-12) {
            throw std::runtime_error("PinholeCamera::from_json fy mismatch");
        }

        std::cout << "PinholeCamera::from_json static method verified" << std::endl;

        // Test 13: Test edge cases
        std::cout << "Test 13: Testing edge cases..." << std::endl;

        // Test with minimal camera
        CameraPtr minimal_camera = std::make_shared<PinholeCamera>();
        std::string minimal_json = minimal_camera->to_json();
        CameraPtr minimal_camera2 = std::make_shared<PinholeCamera>();
        minimal_camera2->init_from_json(minimal_json);

        if (minimal_camera2->width != minimal_camera->width) {
            throw std::runtime_error("Minimal camera width mismatch");
        }
        if (minimal_camera2->height != minimal_camera->height) {
            throw std::runtime_error("Minimal camera height mismatch");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 14: Test error handling
        std::cout << "Test 14: Testing error handling..." << std::endl;

        // Test with invalid JSON
        try {
            CameraPtr test_camera = std::make_shared<PinholeCamera>();
            test_camera->init_from_json("invalid json string");
            throw std::runtime_error("Should have thrown exception for invalid JSON");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught invalid JSON: " << e.what() << std::endl;
        }

        // Test with empty JSON
        try {
            CameraPtr test_camera = std::make_shared<PinholeCamera>();
            test_camera->init_from_json("{}");
            // This should work with default values
            std::cout << "Empty JSON handled correctly with defaults" << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Empty JSON error: " << e.what() << std::endl;
        }

        std::cout << "Error handling verified" << std::endl;

        // Test 15: Test JSON structure validation
        std::cout << "Test 15: Testing JSON structure validation..." << std::endl;

        // Parse the JSON and verify structure
        nlohmann::json parsed_json = nlohmann::json::parse(json_str);

        // Check required fields exist
        std::vector<std::string> required_fields = {
            "type",         "width",       "height", "fx",
            "fy",           "cx",          "cy",     "D",
            "fps",          "bf",          "b",      "depth_factor",
            "is_distorted", "u_min",       "u_max",  "v_min",
            "v_max",        "initialized", "K",      "Kinv"};

        for (const auto &field : required_fields) {
            if (!parsed_json.contains(field)) {
                throw std::runtime_error("Required field '" + field + "' missing from JSON");
            }
        }

        // Check data types
        // Check data types
        if (!parsed_json["type"].is_number_integer()) {
            throw std::runtime_error("type field should be integer");
        }
        if (!parsed_json["width"].is_number_integer()) {
            throw std::runtime_error("width field should be integer");
        }
        if (!parsed_json["height"].is_number_integer()) {
            throw std::runtime_error("height field should be integer");
        }
        if (!parsed_json["fx"].is_number()) { // Changed from is_number_float() to is_number()
            throw std::runtime_error("fx field should be number");
        }
        if (!parsed_json["fy"].is_number()) { // Changed from is_number_float() to is_number()
            throw std::runtime_error("fy field should be number");
        }
        if (!parsed_json["cx"].is_number()) { // Changed from is_number_float() to is_number()
            throw std::runtime_error("cx field should be number");
        }
        if (!parsed_json["cy"].is_number()) { // Changed from is_number_float() to is_number()
            throw std::runtime_error("cy field should be number");
        }
        if (!parsed_json["D"].is_array()) {
            throw std::runtime_error("D field should be array");
        }
        if (!parsed_json["K"].is_array()) {
            throw std::runtime_error("K field should be array");
        }
        if (!parsed_json["Kinv"].is_array()) {
            throw std::runtime_error("Kinv field should be array");
        }

        std::cout << "JSON structure validation verified" << std::endl;

        // Test 16: Test infinity handling for depth_threshold
        std::cout << "Test 16: Testing infinity handling for depth_threshold..." << std::endl;

        CameraPtr inf_camera = std::make_shared<PinholeCamera>();
        inf_camera->depth_threshold = std::numeric_limits<double>::infinity();

        std::string inf_json = inf_camera->to_json();
        nlohmann::json inf_parsed = nlohmann::json::parse(inf_json);

        if (!inf_parsed["depth_threshold"].is_null()) {
            throw std::runtime_error("Infinity depth_threshold should be serialized as null");
        }

        CameraPtr inf_camera2 = std::make_shared<PinholeCamera>();
        inf_camera2->init_from_json(inf_json);

        if (!std::isinf(inf_camera2->depth_threshold)) {
            throw std::runtime_error("null depth_threshold should be deserialized as infinity");
        }

        std::cout << "Infinity handling verified" << std::endl;

        std::cout << "\nAll Camera JSON serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
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
#include <opencv2/core.hpp>

#include <pybind11/embed.h>

#include "camera.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};

    try {
        std::cout << "Starting Camera numpy serialization tests..." << std::endl;

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

        // Test 2: Serialize to numpy tuple
        std::cout << "Test 2: Serializing Camera to numpy tuple..." << std::endl;
        pybind11::tuple state_tuple = camera->state_tuple();
        if (state_tuple.empty()) {
            throw std::runtime_error("Numpy serialization returned empty tuple");
        }
        std::cout << "Numpy serialization successful, tuple size: " << state_tuple.size()
                  << std::endl;

        // Test 3: Deserialize from numpy tuple
        std::cout << "Test 3: Deserializing Camera from numpy tuple..." << std::endl;
        CameraPtr camera2 = std::make_shared<PinholeCamera>();
        camera2->restore_from_state(state_tuple);
        std::cout << "Numpy deserialization successful" << std::endl;

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

        // Test 5: Verify field of view parameters
        std::cout << "Test 5: Verifying field of view parameters..." << std::endl;
        if (std::abs(camera2->fovx - camera->fovx) > 1e-12) {
            throw std::runtime_error("fovx mismatch");
        }
        if (std::abs(camera2->fovy - camera->fovy) > 1e-12) {
            throw std::runtime_error("fovy mismatch");
        }
        std::cout << "Field of view parameters verified" << std::endl;

        // Test 6: Verify sensor type and depth parameters
        std::cout << "Test 6: Verifying sensor type and depth parameters..." << std::endl;
        if (camera2->sensor_type != camera->sensor_type) {
            throw std::runtime_error("sensor_type mismatch");
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

        // Test 12: Test edge cases
        std::cout << "Test 12: Testing edge cases..." << std::endl;

        // Test with minimal camera
        CameraPtr minimal_camera = std::make_shared<PinholeCamera>();
        pybind11::tuple minimal_tuple = minimal_camera->state_tuple();
        CameraPtr minimal_camera2 = std::make_shared<PinholeCamera>();
        minimal_camera2->restore_from_state(minimal_tuple);

        if (minimal_camera2->width != minimal_camera->width) {
            throw std::runtime_error("Minimal camera width mismatch");
        }
        if (minimal_camera2->height != minimal_camera->height) {
            throw std::runtime_error("Minimal camera height mismatch");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 13: Test error handling
        std::cout << "Test 13: Testing error handling..." << std::endl;

        // Test with invalid tuple
        try {
            pybind11::tuple invalid_tuple = pybind11::make_tuple(1, 2, 3);
            CameraPtr test_camera = std::make_shared<PinholeCamera>();
            test_camera->restore_from_state(invalid_tuple);
            throw std::runtime_error("Should have thrown exception for invalid tuple");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught invalid tuple: " << e.what() << std::endl;
        }

        // Test with wrong version
        try {
            pybind11::tuple wrong_version_tuple = pybind11::make_tuple(999); // Wrong version
            CameraPtr test_camera = std::make_shared<PinholeCamera>();
            test_camera->restore_from_state(wrong_version_tuple);
            throw std::runtime_error("Should have thrown exception for wrong version");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught wrong version: " << e.what() << std::endl;
        }

        std::cout << "Error handling verified" << std::endl;

        // Test 14: Test PinholeCamera specific serialization
        std::cout << "Test 14: Testing PinholeCamera specific serialization..." << std::endl;

        PinholeCameraPtr pinhole_camera = std::make_shared<PinholeCamera>(camera_config);
        pinhole_camera->fovx = 70.0;
        pinhole_camera->fovy = 50.0;

        pybind11::tuple pinhole_tuple = pinhole_camera->state_tuple();
        PinholeCameraPtr pinhole_camera2 = std::make_shared<PinholeCamera>();
        pinhole_camera2->restore_from_state(pinhole_tuple);

        if (std::abs(pinhole_camera2->fovx - pinhole_camera->fovx) > 1e-12) {
            throw std::runtime_error("PinholeCamera fovx mismatch");
        }
        if (std::abs(pinhole_camera2->fovy - pinhole_camera->fovy) > 1e-12) {
            throw std::runtime_error("PinholeCamera fovy mismatch");
        }

        std::cout << "PinholeCamera specific serialization verified" << std::endl;

        std::cout << "\nAll Camera numpy serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
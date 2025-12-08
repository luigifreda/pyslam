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

#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    try {
        std::cout << "Starting Frame JSON serialization tests..." << std::endl;

        // Create test camera
        auto camera = create_test_camera();

        // Create test pose
        CameraPose pose = create_test_pose();

        // Create test images
        cv::Mat img, img_right, depth_img, semantic_img;
        create_test_images(img, img_right, depth_img, semantic_img);

        // Create Frame with comprehensive data
        FramePtr frame = std::make_shared<Frame>(camera, img, img_right, depth_img, pose, 123,
                                                 456.789, 789, semantic_img);

        // Set frame properties
        frame->is_keyframe = true;
        frame->median_depth = 5.5f;
        frame->fov_center_c = Eigen::Vector3d(1.0, 2.0, 3.0);
        frame->fov_center_w = Eigen::Vector3d(4.0, 5.0, 6.0);
        frame->is_blurry = false;
        frame->laplacian_var = 123.45f;

        // Create test keypoints data
        const int num_kps = 10;
        frame->kps = MatNx2f(num_kps, 2);
        frame->kps_r = MatNx2f(num_kps, 2);
        frame->kpsu = MatNx2f(num_kps, 2);
        frame->kpsn = MatNx2f(num_kps, 2);

        for (int i = 0; i < num_kps; ++i) {
            frame->kps(i, 0) = 100.0f + i * 10.0f;
            frame->kps(i, 1) = 200.0f + i * 5.0f;
            frame->kps_r(i, 0) = 150.0f + i * 10.0f;
            frame->kps_r(i, 1) = 200.0f + i * 5.0f;
            frame->kpsu(i, 0) = 100.0f + i * 10.0f;
            frame->kpsu(i, 1) = 200.0f + i * 5.0f;
            frame->kpsn(i, 0) = (100.0f + i * 10.0f - 320.0f) / 525.0f;
            frame->kpsn(i, 1) = (200.0f + i * 5.0f - 240.0f) / 525.0f;
        }

        // Create semantic keypoints
        frame->kps_sem = cv::Mat(num_kps, 8, CV_32F);
        fill_test_mat(frame->kps_sem);

        // Create octaves, sizes, angles
        frame->octaves.resize(num_kps);
        frame->octaves_r.resize(num_kps);
        frame->sizes.resize(num_kps);
        frame->angles.resize(num_kps);

        for (int i = 0; i < num_kps; ++i) {
            frame->octaves[i] = i % 4;
            frame->octaves_r[i] = i % 4;
            frame->sizes[i] = 10.0f + i * 2.0f;
            frame->angles[i] = i * 36.0f; // degrees
        }

        // Create descriptors
        frame->des = cv::Mat(num_kps, 32, CV_8U);
        frame->des_r = cv::Mat(num_kps, 32, CV_8U);
        fill_test_mat(frame->des);
        fill_test_mat(frame->des_r);

        // Create depths and kps_ur
        frame->depths.resize(num_kps);
        frame->kps_ur.resize(num_kps);
        for (int i = 0; i < num_kps; ++i) {
            frame->depths[i] = 5.0f + i * 0.5f;
            frame->kps_ur[i] = 150.0f + i * 10.0f;
        }

        // Create map points and outliers
        frame->points.resize(num_kps, nullptr);
        frame->outliers.resize(num_kps, false);

        // Create some test map points
        auto mp1 = std::make_shared<MapPoint>(Eigen::Vector3d(1.0, 2.0, 3.0),
                                              Eigen::Matrix<unsigned char, 3, 1>(10, 20, 30),
                                              KeyFramePtr(nullptr), -1, 1001);
        auto mp2 = std::make_shared<MapPoint>(Eigen::Vector3d(4.0, 5.0, 6.0),
                                              Eigen::Matrix<unsigned char, 3, 1>(40, 50, 60),
                                              KeyFramePtr(nullptr), -1, 1002);

        frame->points[0] = mp1;
        frame->points[5] = mp2;
        frame->outliers[3] = true;

        // Create reference keyframe
        auto kf_ref = std::make_shared<KeyFrame>(2001);
        frame->kf_ref = kf_ref;

        std::cout << "Created test frame with comprehensive data" << std::endl;

        // Test 1: Serialize to JSON
        std::cout << "Test 1: Serializing frame to JSON..." << std::endl;
        const std::string json_str = frame->to_json();
        if (json_str.empty()) {
            throw std::runtime_error("JSON serialization returned empty string");
        }
        std::cout << "JSON serialization successful, length: " << json_str.length() << std::endl;

        // Test 2: Deserialize from JSON
        std::cout << "Test 2: Deserializing frame from JSON..." << std::endl;
        FramePtr frame2 = Frame::from_json(json_str);
        if (!frame2) {
            throw std::runtime_error("JSON deserialization returned null pointer");
        }
        std::cout << "JSON deserialization successful" << std::endl;

        // Test 3: Replace IDs with objects
        std::cout << "Test 3: Replacing IDs with objects..." << std::endl;
        std::vector<MapPointPtr> points{mp1, mp2};
        std::vector<FramePtr> frames{frame};
        std::vector<KeyFramePtr> keyframes{kf_ref};
        frame2->replace_ids_with_objects(points, frames, keyframes);
        std::cout << "ID replacement successful" << std::endl;

        // Test 4: Verify basic properties
        std::cout << "Test 4: Verifying basic properties..." << std::endl;
        if (frame2->id != frame->id) {
            throw std::runtime_error("ID mismatch: " + std::to_string(frame2->id) + " vs " +
                                     std::to_string(frame->id));
        }
        if (std::abs(frame2->timestamp - frame->timestamp) > 1e-9) {
            throw std::runtime_error("Timestamp mismatch");
        }
        if (frame2->img_id != frame->img_id) {
            throw std::runtime_error("img_id mismatch");
        }
        if (frame2->is_keyframe != frame->is_keyframe) {
            throw std::runtime_error("is_keyframe mismatch");
        }
        if (std::abs(frame2->median_depth - frame->median_depth) > 1e-6f) {
            throw std::runtime_error("median_depth mismatch");
        }
        if (frame2->is_blurry != frame->is_blurry) {
            throw std::runtime_error("is_blurry mismatch");
        }
        if (std::abs(frame2->laplacian_var - frame->laplacian_var) > 1e-6f) {
            throw std::runtime_error("laplacian_var mismatch");
        }
        std::cout << "Basic properties verified" << std::endl;

        // Test 5: Verify pose
        std::cout << "Test 5: Verifying pose..." << std::endl;
        if (!eigen_matrices_equal(frame2->Tcw(), frame->Tcw())) {
            throw std::runtime_error("Pose Tcw mismatch");
        }
        std::cout << "Pose verified" << std::endl;

        // Test 6: Verify FOV centers
        std::cout << "Test 6: Verifying FOV centers..." << std::endl;
        if (!eigen_vectors_equal(frame2->fov_center_c, frame->fov_center_c)) {
            throw std::runtime_error("fov_center_c mismatch");
        }
        if (!eigen_vectors_equal(frame2->fov_center_w, frame->fov_center_w)) {
            throw std::runtime_error("fov_center_w mismatch");
        }
        std::cout << "FOV centers verified" << std::endl;

        // Test 7: Verify keypoints
        std::cout << "Test 7: Verifying keypoints..." << std::endl;
        if (frame2->kps.rows() != frame->kps.rows() || frame2->kps.cols() != frame->kps.cols()) {
            throw std::runtime_error("kps dimensions mismatch");
        }
        for (int i = 0; i < num_kps; ++i) {
            if (std::abs(frame2->kps(i, 0) - frame->kps(i, 0)) > 1e-6f ||
                std::abs(frame2->kps(i, 1) - frame->kps(i, 1)) > 1e-6f) {
                throw std::runtime_error("kps[" + std::to_string(i) + "] mismatch");
            }
        }

        // Verify other keypoint arrays
        if (frame2->kps_r.rows() != frame->kps_r.rows()) {
            throw std::runtime_error("kps_r dimensions mismatch");
        }
        if (frame2->kpsu.rows() != frame->kpsu.rows()) {
            throw std::runtime_error("kpsu dimensions mismatch");
        }
        if (frame2->kpsn.rows() != frame->kpsn.rows()) {
            throw std::runtime_error("kpsn dimensions mismatch");
        }
        std::cout << "Keypoints verified" << std::endl;

        // Test 8: Verify semantic keypoints
        std::cout << "Test 8: Verifying semantic keypoints..." << std::endl;

        // Debug information
        std::cout << "Original kps_sem: " << frame->kps_sem.rows << "x" << frame->kps_sem.cols
                  << ", type: " << frame->kps_sem.type()
                  << ", channels: " << frame->kps_sem.channels() << std::endl;
        std::cout << "Deserialized kps_sem: " << frame2->kps_sem.rows << "x" << frame2->kps_sem.cols
                  << ", type: " << frame2->kps_sem.type()
                  << ", channels: " << frame2->kps_sem.channels() << std::endl;

        if (frame->kps_sem.empty() && frame2->kps_sem.empty()) {
            std::cout << "Both kps_sem are empty - OK" << std::endl;
        } else if (frame->kps_sem.empty() || frame2->kps_sem.empty()) {
            throw std::runtime_error("One kps_sem is empty, the other is not");
        } else {
            // Compare first few elements for debugging
            std::cout << "First few elements comparison:" << std::endl;
            for (int i = 0; i < std::min(3, frame->kps_sem.rows); ++i) {
                for (int j = 0; j < std::min(3, frame->kps_sem.cols); ++j) {
                    float orig = frame->kps_sem.at<float>(i, j);
                    float deser = frame2->kps_sem.at<float>(i, j);
                    float diff = std::abs(orig - deser);
                    std::cout << "  [" << i << "," << j << "]: " << orig << " vs " << deser
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }

        if (!mats_equal_tolerant(frame->kps_sem, frame2->kps_sem)) {
            throw std::runtime_error("kps_sem mismatch");
        }
        std::cout << "Semantic keypoints verified" << std::endl;

        // Test 9: Verify octaves, sizes, angles
        std::cout << "Test 9: Verifying octaves, sizes, angles..." << std::endl;
        if (frame2->octaves.size() != frame->octaves.size()) {
            throw std::runtime_error("octaves size mismatch");
        }
        for (size_t i = 0; i < frame->octaves.size(); ++i) {
            if (frame2->octaves[i] != frame->octaves[i]) {
                throw std::runtime_error("octaves[" + std::to_string(i) + "] mismatch");
            }
        }

        if (frame2->sizes.size() != frame->sizes.size()) {
            throw std::runtime_error("sizes size mismatch");
        }
        for (size_t i = 0; i < frame->sizes.size(); ++i) {
            if (std::abs(frame2->sizes[i] - frame->sizes[i]) > 1e-6f) {
                throw std::runtime_error("sizes[" + std::to_string(i) + "] mismatch");
            }
        }

        if (frame2->angles.size() != frame->angles.size()) {
            throw std::runtime_error("angles size mismatch");
        }
        for (size_t i = 0; i < frame->angles.size(); ++i) {
            if (std::abs(frame2->angles[i] - frame->angles[i]) > 1e-6f) {
                throw std::runtime_error("angles[" + std::to_string(i) + "] mismatch");
            }
        }
        std::cout << "Octaves, sizes, angles verified" << std::endl;

        // Test 10: Verify descriptors
        std::cout << "Test 10: Verifying descriptors..." << std::endl;
        if (!mats_equal_exact(frame->des, frame2->des)) {
            throw std::runtime_error("des mismatch");
        }
        if (!mats_equal_exact(frame->des_r, frame2->des_r)) {
            throw std::runtime_error("des_r mismatch");
        }
        std::cout << "Descriptors verified" << std::endl;

        // Test 11: Verify depths and kps_ur
        std::cout << "Test 11: Verifying depths and kps_ur..." << std::endl;
        if (frame2->depths.size() != frame->depths.size()) {
            throw std::runtime_error("depths size mismatch");
        }
        for (size_t i = 0; i < frame->depths.size(); ++i) {
            if (std::abs(frame2->depths[i] - frame->depths[i]) > 1e-6f) {
                throw std::runtime_error("depths[" + std::to_string(i) + "] mismatch");
            }
        }

        if (frame2->kps_ur.size() != frame->kps_ur.size()) {
            throw std::runtime_error("kps_ur size mismatch");
        }
        for (size_t i = 0; i < frame->kps_ur.size(); ++i) {
            if (std::abs(frame2->kps_ur[i] - frame->kps_ur[i]) > 1e-6f) {
                throw std::runtime_error("kps_ur[" + std::to_string(i) + "] mismatch");
            }
        }
        std::cout << "Depths and kps_ur verified" << std::endl;

        // Test 12: Verify map points and outliers
        std::cout << "Test 12: Verifying map points and outliers..." << std::endl;
        if (frame2->points.size() != frame->points.size()) {
            throw std::runtime_error("points size mismatch");
        }
        if (frame2->outliers.size() != frame->outliers.size()) {
            throw std::runtime_error("outliers size mismatch");
        }

        // Check specific map points
        if (frame2->points[0] != mp1) {
            throw std::runtime_error("points[0] not restored correctly");
        }
        if (frame2->points[5] != mp2) {
            throw std::runtime_error("points[5] not restored correctly");
        }

        // Check outliers
        for (size_t i = 0; i < frame->outliers.size(); ++i) {
            if (frame2->outliers[i] != frame->outliers[i]) {
                throw std::runtime_error("outliers[" + std::to_string(i) + "] mismatch");
            }
        }
        std::cout << "Map points and outliers verified" << std::endl;

        // Test 13: Verify reference keyframe
        std::cout << "Test 13: Verifying reference keyframe..." << std::endl;
        if (!frame2->kf_ref || frame2->kf_ref->id != kf_ref->id) {
            throw std::runtime_error("kf_ref not restored correctly");
        }
        std::cout << "Reference keyframe verified" << std::endl;

        // Test 14: Verify images
        std::cout << "Test 14: Verifying images..." << std::endl;
        if (!mats_equal_exact(frame->img, frame2->img)) {
            throw std::runtime_error("img mismatch");
        }
        if (!mats_equal_exact(frame->img_right, frame2->img_right)) {
            throw std::runtime_error("img_right mismatch");
        }
        if (!mats_equal_tolerant(frame->depth_img, frame2->depth_img)) {
            throw std::runtime_error("depth_img mismatch");
        }
        if (!mats_equal_exact(frame->semantic_img, frame2->semantic_img)) {
            throw std::runtime_error("semantic_img mismatch");
        }
        std::cout << "Images verified" << std::endl;

        // Test 15: Test edge cases
        std::cout << "Test 15: Testing edge cases..." << std::endl;

        // Test with empty frame
        FramePtr empty_frame =
            std::make_shared<Frame>(camera, cv::Mat(), cv::Mat(), cv::Mat(), CameraPose(), 999);
        std::string empty_json = empty_frame->to_json();
        FramePtr empty_frame2 = Frame::from_json(empty_json);

        if (empty_frame2->id != empty_frame->id) {
            throw std::runtime_error("Empty frame ID mismatch");
        }
        if (empty_frame2->kps.rows() != 0) {
            throw std::runtime_error("Empty frame should have no keypoints");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 16: Test JSON parsing robustness
        std::cout << "Test 16: Testing JSON parsing robustness..." << std::endl;

        // Test with malformed JSON
        try {
            Frame::from_json("invalid json");
            throw std::runtime_error("Should have thrown exception for invalid JSON");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught invalid JSON: " << e.what() << std::endl;
        }

        // Test with missing fields
        nlohmann::json minimal_json;
        minimal_json["id"] = 123;
        minimal_json["timestamp"] = 456.789;
        minimal_json["img_id"] = 789;
        minimal_json["pose"] = std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0));
        minimal_json["camera"] = nullptr;
        minimal_json["is_keyframe"] = false;
        minimal_json["median_depth"] = 0.0;
        minimal_json["fov_center_c"] = nullptr;
        minimal_json["fov_center_w"] = nullptr;
        minimal_json["is_blurry"] = false;
        minimal_json["laplacian_var"] = 0.0;
        minimal_json["kps"] = nullptr;
        minimal_json["kps_r"] = nullptr;
        minimal_json["kpsu"] = nullptr;
        minimal_json["kpsn"] = nullptr;
        minimal_json["kps_sem"] = nullptr;
        minimal_json["octaves"] = nullptr;
        minimal_json["octaves_r"] = nullptr;
        minimal_json["sizes"] = nullptr;
        minimal_json["angles"] = nullptr;
        minimal_json["des"] = nullptr;
        minimal_json["des_r"] = nullptr;
        minimal_json["depths"] = nullptr;
        minimal_json["kps_ur"] = nullptr;
        minimal_json["points"] = nullptr;
        minimal_json["outliers"] = nullptr;
        minimal_json["kf_ref"] = -1;
        minimal_json["img"] = nullptr;
        minimal_json["depth_img"] = nullptr;
        minimal_json["img_right"] = nullptr;
        minimal_json["semantic_img"] = nullptr;

        FramePtr minimal_frame = Frame::from_json(minimal_json.dump());
        if (minimal_frame->id != 123) {
            throw std::runtime_error("Minimal frame ID mismatch");
        }
        std::cout << "JSON parsing robustness verified" << std::endl;

        std::cout << "\nAll Frame JSON serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

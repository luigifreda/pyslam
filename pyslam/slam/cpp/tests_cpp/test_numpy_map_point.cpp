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

#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};

    try {
        std::cout << "Starting MapPoint numpy serialization tests..." << std::endl;

        // Initialize FeatureSharedResources with default values
        init_feature_shared_info();

        // Create test camera
        std::cout << "Creating test camera..." << std::endl;
        auto camera = create_test_camera();
        std::cout << "Test camera created successfully" << std::endl;

        // Create test pose
        std::cout << "Creating test pose..." << std::endl;
        CameraPose pose = create_test_pose();
        std::cout << "Test pose created successfully" << std::endl;

        // Create test images
        std::cout << "Creating test images..." << std::endl;
        cv::Mat img, img_right, depth_img, semantic_img;
        create_test_images(img, img_right, depth_img, semantic_img);
        std::cout << "Test images created successfully" << std::endl;

        // Create Frame first
        FramePtr frame = std::make_shared<Frame>(camera, img, img_right, depth_img, pose, 123,
                                                 456.789, 789, semantic_img);

        // Create KeyFrame from Frame
        KeyFramePtr keyframe = std::make_shared<KeyFrame>(frame, img, img_right, depth_img, 2001);

        // Ensure KeyFrame has proper keypoint data for MapPoint creation
        init_keyframe_for_mappoints(keyframe);

        std::cout << "Creating MapPoint..." << std::endl;
        Eigen::Vector3d position(1.0, 2.0, 3.0);
        Eigen::Matrix<unsigned char, 3, 1> color(10, 20, 30);
        std::cout << "About to create MapPoint with keyframe ID: " << keyframe->id << std::endl;
        MapPointPtr map_point = std::make_shared<MapPoint>(position, color, keyframe, 5, 1001);
        std::cout << "MapPoint created successfully!" << std::endl;

        // Set MapPoint-specific properties
        map_point->_is_bad = false;
        map_point->_num_observations = 3;
        map_point->num_times_visible = 5;
        map_point->num_times_found = 4;
        map_point->last_frame_id_seen = 123;

        // Set geometric properties
        map_point->normal = Eigen::Vector3d(0.0, 0.0, 1.0);
        map_point->_min_distance = 1.0f;
        map_point->_max_distance = 10.0f;

        // Set visual properties
        map_point->semantic_des = cv::Mat(1, 8, CV_32F);
        map_point->des = cv::Mat(1, 32, CV_8U);
        fill_test_mat(map_point->semantic_des);
        fill_test_mat(map_point->des);

        // Set reference information
        map_point->first_kid = 2001;

        // Set update counters
        map_point->num_observations_on_last_update_des = 2;
        map_point->num_observations_on_last_update_normals = 3;
        map_point->num_observations_on_last_update_semantics = 1;

        // Set GBA properties
        map_point->pt_GBA = Eigen::Vector3d(1.1, 2.1, 3.1);
        map_point->is_pt_GBA_valid = true;
        map_point->GBA_kf_id = 3001;

        // Add observations
        auto kf1 = std::make_shared<KeyFrame>(3001);
        auto kf2 = std::make_shared<KeyFrame>(3002);
        map_point->add_observation(kf1, 10);
        map_point->add_observation(kf2, 15);

        // Add frame views
        auto frame1 = std::make_shared<Frame>(4001);
        auto frame2 = std::make_shared<Frame>(4002);
        map_point->add_frame_view(frame1, 20);
        map_point->add_frame_view(frame2, 25);

        // Set reference keyframe
        map_point->kf_ref = keyframe;

        // Set loop correction properties
        map_point->corrected_by_kf = 5001;
        map_point->corrected_reference = 5002;

        std::cout << "Created test MapPoint with comprehensive data" << std::endl;

        // Test 1: Serialize to numpy tuple
        std::cout << "Test 1: Serializing MapPoint to numpy tuple..." << std::endl;
        pybind11::tuple state_tuple = map_point->state_tuple();
        if (state_tuple.empty()) {
            throw std::runtime_error("Numpy serialization returned empty tuple");
        }
        std::cout << "Numpy serialization successful, tuple size: " << state_tuple.size()
                  << std::endl;

        // Test 2: Deserialize from numpy tuple
        std::cout << "Test 2: Deserializing MapPoint from numpy tuple..." << std::endl;
        MapPointPtr map_point2 = std::make_shared<MapPoint>();
        map_point2->restore_from_state(state_tuple);
        std::cout << "Numpy deserialization successful" << std::endl;

        // Test 3: Replace IDs with objects
        std::cout << "Test 3: Replacing IDs with objects..." << std::endl;
        std::vector<MapPointPtr> points{map_point};
        std::vector<FramePtr> frames{frame, frame1, frame2};
        std::vector<KeyFramePtr> keyframes{keyframe, kf1, kf2};
        map_point2->replace_ids_with_objects(points, frames, keyframes);
        std::cout << "ID replacement successful" << std::endl;

        // Test 4: Verify basic properties
        std::cout << "Test 4: Verifying basic properties..." << std::endl;
        if (map_point2->id != map_point->id) {
            throw std::runtime_error("ID mismatch: " + std::to_string(map_point2->id) + " vs " +
                                     std::to_string(map_point->id));
        }
        if (map_point2->_is_bad != map_point->_is_bad) {
            throw std::runtime_error("_is_bad mismatch");
        }
        if (map_point2->_num_observations != map_point->_num_observations) {
            throw std::runtime_error("_num_observations mismatch");
        }
        if (map_point2->num_times_visible != map_point->num_times_visible) {
            throw std::runtime_error("num_times_visible mismatch");
        }
        if (map_point2->num_times_found != map_point->num_times_found) {
            throw std::runtime_error("num_times_found mismatch");
        }
        if (map_point2->last_frame_id_seen != map_point->last_frame_id_seen) {
            throw std::runtime_error("last_frame_id_seen mismatch");
        }
        std::cout << "Basic properties verified" << std::endl;

        // Test 5: Verify position
        std::cout << "Test 5: Verifying position..." << std::endl;
        if (!eigen_vectors_equal(map_point2->pt(), map_point->pt())) {
            throw std::runtime_error("Position mismatch");
        }
        std::cout << "Position verified" << std::endl;

        // Test 6: Verify geometric properties
        std::cout << "Test 6: Verifying geometric properties..." << std::endl;
        if (!eigen_vectors_equal(map_point2->normal, map_point->normal)) {
            throw std::runtime_error("Normal mismatch");
        }
        if (std::abs(map_point2->_min_distance - map_point->_min_distance) > 1e-6f) {
            throw std::runtime_error("_min_distance mismatch");
        }
        if (std::abs(map_point2->_max_distance - map_point->_max_distance) > 1e-6f) {
            throw std::runtime_error("_max_distance mismatch");
        }
        std::cout << "Geometric properties verified" << std::endl;

        // Test 7: Verify visual properties
        std::cout << "Test 7: Verifying visual properties..." << std::endl;
        if (map_point2->color(0) != map_point->color(0) ||
            map_point2->color(1) != map_point->color(1) ||
            map_point2->color(2) != map_point->color(2)) {
            throw std::runtime_error("Color mismatch");
        }
        if (!mats_equal_tolerant(map_point2->semantic_des, map_point->semantic_des)) {
            throw std::runtime_error("semantic_des mismatch");
        }
        if (!mats_equal_exact(map_point2->des, map_point->des)) {
            throw std::runtime_error("des mismatch");
        }
        std::cout << "Visual properties verified" << std::endl;

        // Test 8: Verify reference information
        std::cout << "Test 8: Verifying reference information..." << std::endl;
        if (map_point2->first_kid != map_point->first_kid) {
            throw std::runtime_error("first_kid mismatch");
        }
        if (!map_point2->kf_ref || map_point2->kf_ref->id != map_point->kf_ref->id) {
            throw std::runtime_error("kf_ref not restored correctly");
        }
        std::cout << "Reference information verified" << std::endl;

        // Test 9: Verify update counters
        std::cout << "Test 9: Verifying update counters..." << std::endl;
        if (map_point2->num_observations_on_last_update_des !=
            map_point->num_observations_on_last_update_des) {
            throw std::runtime_error("num_observations_on_last_update_des mismatch");
        }
        if (map_point2->num_observations_on_last_update_normals !=
            map_point->num_observations_on_last_update_normals) {
            throw std::runtime_error("num_observations_on_last_update_normals mismatch");
        }
        if (map_point2->num_observations_on_last_update_semantics !=
            map_point->num_observations_on_last_update_semantics) {
            throw std::runtime_error("num_observations_on_last_update_semantics mismatch");
        }
        std::cout << "Update counters verified" << std::endl;

        // Test 10: Verify GBA properties
        std::cout << "Test 10: Verifying GBA properties..." << std::endl;
        if (!eigen_vectors_equal(map_point2->pt_GBA, map_point->pt_GBA)) {
            throw std::runtime_error("pt_GBA mismatch");
        }
        if (map_point2->is_pt_GBA_valid != map_point->is_pt_GBA_valid) {
            throw std::runtime_error("is_pt_GBA_valid mismatch");
        }
        if (map_point2->GBA_kf_id != map_point->GBA_kf_id) {
            throw std::runtime_error("GBA_kf_id mismatch");
        }
        std::cout << "GBA properties verified" << std::endl;

        // Test 11: Verify observations
        std::cout << "Test 11: Verifying observations..." << std::endl;
        if (map_point2->num_observations() != map_point->num_observations()) {
            throw std::runtime_error("Number of observations mismatch");
        }

        // Check specific observations
        auto obs2 = map_point2->observations();
        auto obs1 = map_point->observations();
        if (obs2.size() != obs1.size()) {
            throw std::runtime_error("Observations size mismatch");
        }

        // Verify observations are restored correctly
        bool found_kf1 = false, found_kf2 = false;
        for (const auto &obs : obs2) {
            if (obs.first->id == kf1->id && obs.second == 10)
                found_kf1 = true;
            if (obs.first->id == kf2->id && obs.second == 15)
                found_kf2 = true;
        }
        if (!found_kf1 || !found_kf2) {
            throw std::runtime_error("Observations not restored correctly");
        }
        std::cout << "Observations verified" << std::endl;

        // Test 12: Verify frame views
        std::cout << "Test 12: Verifying frame views..." << std::endl;
        auto views2 = map_point2->frame_views();
        auto views1 = map_point->frame_views();
        if (views2.size() != views1.size()) {
            throw std::runtime_error("Frame views size mismatch");
        }

        // Verify frame views are restored correctly
        bool found_frame1 = false, found_frame2 = false;
        for (const auto &view : views2) {
            if (view.first->id == frame1->id && view.second == 20)
                found_frame1 = true;
            if (view.first->id == frame2->id && view.second == 25)
                found_frame2 = true;
        }
        if (!found_frame1 || !found_frame2) {
            throw std::runtime_error("Frame views not restored correctly");
        }
        std::cout << "Frame views verified" << std::endl;

        // Test 13: Verify loop correction properties
        std::cout << "Test 13: Verifying loop correction properties..." << std::endl;
        if (map_point2->corrected_by_kf != map_point->corrected_by_kf) {
            throw std::runtime_error("corrected_by_kf mismatch");
        }
        if (map_point2->corrected_reference != map_point->corrected_reference) {
            throw std::runtime_error("corrected_reference mismatch");
        }
        std::cout << "Loop correction properties verified" << std::endl;

        // Test 14: Test edge cases
        std::cout << "Test 14: Testing edge cases..." << std::endl;

        // Test with minimal MapPoint
        MapPointPtr minimal_mp = std::make_shared<MapPoint>(9999);
        pybind11::tuple minimal_tuple = minimal_mp->state_tuple();
        MapPointPtr minimal_mp2 = std::make_shared<MapPoint>();
        minimal_mp2->restore_from_state(minimal_tuple);

        if (minimal_mp2->id != minimal_mp->id) {
            throw std::runtime_error("Minimal MapPoint ID mismatch");
        }
        if (minimal_mp2->num_observations() != 0) {
            throw std::runtime_error("Minimal MapPoint should have no observations");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 15: Test error handling
        std::cout << "Test 15: Testing error handling..." << std::endl;

        // Test with invalid tuple
        try {
            pybind11::tuple invalid_tuple = pybind11::make_tuple(1, 2, 3);
            MapPointPtr test_mp = std::make_shared<MapPoint>();
            test_mp->restore_from_state(invalid_tuple);
            throw std::runtime_error("Should have thrown exception for invalid tuple");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught invalid tuple: " << e.what() << std::endl;
        }

        std::cout << "\nAll MapPoint numpy serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
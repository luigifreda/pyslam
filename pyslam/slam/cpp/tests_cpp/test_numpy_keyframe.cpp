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
        std::cout << "Starting KeyFrame numpy serialization tests..." << std::endl;

        // Initialize FeatureSharedResources with default values
        init_feature_shared_info();

        // Create test camera
        auto camera = create_test_camera();

        // Create test pose
        CameraPose pose = create_test_pose();

        // Create test images
        cv::Mat img, img_right, depth_img, semantic_img;
        create_test_images(img, img_right, depth_img, semantic_img);

        // Create Frame first
        FramePtr frame = std::make_shared<Frame>(camera, img, img_right, depth_img, pose, 123,
                                                 456.789, 789, semantic_img);

        // Create KeyFrame from Frame
        KeyFramePtr keyframe = std::make_shared<KeyFrame>(frame, img, img_right, depth_img, 2001);

        // Set KeyFrame-specific properties
        keyframe->_is_bad = false;
        keyframe->lba_count = 5;
        keyframe->to_be_erased = false;
        keyframe->is_Tcw_GBA_valid = true;

        // Set pose relative to parent
        Eigen::Matrix4d parent_pose = Eigen::Matrix4d::Identity();
        parent_pose(0, 3) = 0.5; // translation x
        parent_pose(1, 3) = 1.0; // translation y
        parent_pose(2, 3) = 1.5; // translation z
        keyframe->_pose_Tcp = CameraPose(parent_pose);

        // Set loop closing properties
        // keyframe->g_des = cv::Mat(1, 64, CV_8U);
        // fill_test_mat(keyframe->g_des);
        keyframe->loop_query_id = 3001;
        keyframe->num_loop_words = 50;
        keyframe->loop_score = 0.85f;

        // Set relocalization properties
        keyframe->reloc_query_id = 4001;
        keyframe->num_reloc_words = 40;
        keyframe->reloc_score = 0.75f;

        // Set GBA properties
        keyframe->GBA_kf_id = 5001;
        keyframe->Tcw_GBA = Eigen::Matrix4d::Identity();
        keyframe->Tcw_GBA(0, 3) = 1.1;
        keyframe->Tcw_GBA(1, 3) = 2.1;
        keyframe->Tcw_GBA(2, 3) = 3.1;
        keyframe->Tcw_before_GBA = Eigen::Matrix4d::Identity();
        keyframe->Tcw_before_GBA(0, 3) = 1.0;
        keyframe->Tcw_before_GBA(1, 3) = 2.0;
        keyframe->Tcw_before_GBA(2, 3) = 3.0;

        // Create parent and children keyframes
        auto parent_kf = std::make_shared<KeyFrame>(3001);
        auto child1 = std::make_shared<KeyFrame>(3002);
        auto child2 = std::make_shared<KeyFrame>(3003);
        auto loop_edge = std::make_shared<KeyFrame>(3004);

        // Set KeyFrameGraph relationships
        keyframe->KeyFrameGraph::parent = parent_kf;
        keyframe->KeyFrameGraph::children.insert(child1);
        keyframe->KeyFrameGraph::children.insert(child2);
        keyframe->KeyFrameGraph::loop_edges.insert(loop_edge);
        keyframe->KeyFrameGraph::init_parent = true;
        keyframe->KeyFrameGraph::not_to_erase = true;
        keyframe->KeyFrameGraph::is_first_connection = false;

        // Set covisibility weights
        keyframe->KeyFrameGraph::connected_keyframes_weights[child1] = 10;
        keyframe->KeyFrameGraph::connected_keyframes_weights[child2] = 15;
        keyframe->KeyFrameGraph::ordered_keyframes_weights.push_back({child1, 10});
        keyframe->KeyFrameGraph::ordered_keyframes_weights.push_back({child2, 15});

        // Ensure KeyFrame has proper keypoint data for MapPoint creation
        init_keyframe_for_mappoints(keyframe);

        // Create some test map points
        auto mp1 = std::make_shared<MapPoint>(Eigen::Vector3d(1.0, 2.0, 3.0),
                                              Eigen::Matrix<unsigned char, 3, 1>(10, 20, 30),
                                              keyframe, 0, 1001);
        auto mp2 = std::make_shared<MapPoint>(Eigen::Vector3d(4.0, 5.0, 6.0),
                                              Eigen::Matrix<unsigned char, 3, 1>(40, 50, 60),
                                              keyframe, 1, 1002);

        // Add map points to keyframe
        keyframe->points[0] = mp1;
        keyframe->points[1] = mp2;

        std::cout << "Created test KeyFrame with comprehensive data" << std::endl;

        // Test 1: Serialize to numpy tuple
        std::cout << "Test 1: Serializing KeyFrame to numpy tuple..." << std::endl;
        pybind11::tuple state_tuple = keyframe->state_tuple();
        if (state_tuple.empty()) {
            throw std::runtime_error("Numpy serialization returned empty tuple");
        }
        std::cout << "Numpy serialization successful, tuple size: " << state_tuple.size()
                  << std::endl;

        // Test 2: Deserialize from numpy tuple
        std::cout << "Test 2: Deserializing KeyFrame from numpy tuple..." << std::endl;
        KeyFramePtr keyframe2 = std::make_shared<KeyFrame>(999); // Use explicit constructor
        keyframe2->restore_from_state(state_tuple);
        std::cout << "Numpy deserialization successful" << std::endl;

        // Test 3: Replace IDs with objects
        std::cout << "Test 3: Replacing IDs with objects..." << std::endl;
        std::vector<MapPointPtr> points{mp1, mp2};
        std::vector<FramePtr> frames{frame};
        std::vector<KeyFramePtr> keyframes{keyframe, parent_kf, child1, child2, loop_edge};
        keyframe2->replace_ids_with_objects(points, frames, keyframes);
        std::cout << "ID replacement successful" << std::endl;

        // Test 4: Verify basic Frame properties
        std::cout << "Test 4: Verifying basic Frame properties..." << std::endl;
        if (keyframe2->id != keyframe->id) {
            throw std::runtime_error("ID mismatch: " + std::to_string(keyframe2->id) + " vs " +
                                     std::to_string(keyframe->id));
        }
        if (std::abs(keyframe2->timestamp - keyframe->timestamp) > 1e-9) {
            throw std::runtime_error("Timestamp mismatch");
        }
        if (keyframe2->img_id != keyframe->img_id) {
            throw std::runtime_error("img_id mismatch");
        }
        if (keyframe2->is_keyframe != keyframe->is_keyframe) {
            throw std::runtime_error("is_keyframe mismatch");
        }
        std::cout << "Basic Frame properties verified" << std::endl;

        // Test 5: Verify KeyFrame-specific properties
        std::cout << "Test 5: Verifying KeyFrame-specific properties..." << std::endl;
        if (keyframe2->kid != keyframe->kid) {
            throw std::runtime_error("kid mismatch: " + std::to_string(keyframe2->kid) + " vs " +
                                     std::to_string(keyframe->kid));
        }
        if (keyframe2->_is_bad != keyframe->_is_bad) {
            throw std::runtime_error("_is_bad mismatch");
        }
        if (keyframe2->lba_count != keyframe->lba_count) {
            throw std::runtime_error("lba_count mismatch");
        }
        if (keyframe2->to_be_erased != keyframe->to_be_erased) {
            throw std::runtime_error("to_be_erased mismatch");
        }
        if (keyframe2->is_Tcw_GBA_valid != keyframe->is_Tcw_GBA_valid) {
            throw std::runtime_error("is_Tcw_GBA_valid mismatch");
        }
        std::cout << "KeyFrame-specific properties verified" << std::endl;

        // Test 6: Verify pose relative to parent
        std::cout << "Test 6: Verifying pose relative to parent..." << std::endl;
        if (!eigen_matrices_equal(keyframe2->_pose_Tcp.Tcw(), keyframe->_pose_Tcp.Tcw())) {
            throw std::runtime_error("_pose_Tcp mismatch");
        }
        std::cout << "Pose relative to parent verified" << std::endl;

        // Test 7: Verify loop closing properties
        std::cout << "Test 7: Verifying loop closing properties..." << std::endl;
        // if (!mats_equal_exact(keyframe->g_des, keyframe2->g_des)) {
        //     throw std::runtime_error("g_des mismatch");
        // }
        if (keyframe2->loop_query_id != keyframe->loop_query_id) {
            throw std::runtime_error("loop_query_id mismatch");
        }
        if (keyframe2->num_loop_words != keyframe->num_loop_words) {
            throw std::runtime_error("num_loop_words mismatch");
        }
        if (std::abs(keyframe2->loop_score - keyframe->loop_score) > 1e-6f) {
            throw std::runtime_error("loop_score mismatch");
        }
        std::cout << "Loop closing properties verified" << std::endl;

        // Test 8: Verify relocalization properties
        std::cout << "Test 8: Verifying relocalization properties..." << std::endl;
        if (keyframe2->reloc_query_id != keyframe->reloc_query_id) {
            throw std::runtime_error("reloc_query_id mismatch");
        }
        if (keyframe2->num_reloc_words != keyframe->num_reloc_words) {
            throw std::runtime_error("num_reloc_words mismatch");
        }
        if (std::abs(keyframe2->reloc_score - keyframe->reloc_score) > 1e-6f) {
            throw std::runtime_error("reloc_score mismatch");
        }
        std::cout << "Relocalization properties verified" << std::endl;

        // Test 9: Verify GBA properties
        std::cout << "Test 9: Verifying GBA properties..." << std::endl;
        if (keyframe2->GBA_kf_id != keyframe->GBA_kf_id) {
            throw std::runtime_error("GBA_kf_id mismatch");
        }
        if (!eigen_matrices_equal(keyframe2->Tcw_GBA, keyframe->Tcw_GBA)) {
            throw std::runtime_error("Tcw_GBA mismatch");
        }
        if (!eigen_matrices_equal(keyframe2->Tcw_before_GBA, keyframe->Tcw_before_GBA)) {
            throw std::runtime_error("Tcw_before_GBA mismatch");
        }
        std::cout << "GBA properties verified" << std::endl;

        // Test 10: Verify KeyFrameGraph properties
        std::cout << "Test 10: Verifying KeyFrameGraph properties..." << std::endl;
        if (keyframe2->KeyFrameGraph::init_parent != keyframe->KeyFrameGraph::init_parent) {
            throw std::runtime_error("init_parent mismatch");
        }
        if (keyframe2->KeyFrameGraph::not_to_erase != keyframe->KeyFrameGraph::not_to_erase) {
            throw std::runtime_error("not_to_erase mismatch");
        }
        if (keyframe2->KeyFrameGraph::is_first_connection !=
            keyframe->KeyFrameGraph::is_first_connection) {
            throw std::runtime_error("is_first_connection mismatch");
        }
        std::cout << "KeyFrameGraph properties verified" << std::endl;

        // Test 11: Verify parent relationship
        std::cout << "Test 11: Verifying parent relationship..." << std::endl;
        if (!keyframe2->KeyFrameGraph::parent ||
            keyframe2->KeyFrameGraph::parent->id != parent_kf->id) {
            throw std::runtime_error("Parent not restored correctly");
        }
        std::cout << "Parent relationship verified" << std::endl;

        // Test 12: Verify children relationships
        std::cout << "Test 12: Verifying children relationships..." << std::endl;
        if (keyframe2->KeyFrameGraph::children.size() != keyframe->KeyFrameGraph::children.size()) {
            throw std::runtime_error("Children size mismatch");
        }

        bool found_child1 = false, found_child2 = false;
        for (const auto &child : keyframe2->KeyFrameGraph::children) {
            if (child->id == child1->id)
                found_child1 = true;
            if (child->id == child2->id)
                found_child2 = true;
        }
        if (!found_child1 || !found_child2) {
            throw std::runtime_error("Children not restored correctly");
        }
        std::cout << "Children relationships verified" << std::endl;

        // Test 13: Verify loop edges
        std::cout << "Test 13: Verifying loop edges..." << std::endl;
        if (keyframe2->KeyFrameGraph::loop_edges.size() !=
            keyframe->KeyFrameGraph::loop_edges.size()) {
            throw std::runtime_error("Loop edges size mismatch");
        }

        bool found_loop_edge = false;
        for (const auto &edge : keyframe2->KeyFrameGraph::loop_edges) {
            if (edge->id == loop_edge->id)
                found_loop_edge = true;
        }
        if (!found_loop_edge) {
            throw std::runtime_error("Loop edge not restored correctly");
        }
        std::cout << "Loop edges verified" << std::endl;

        // Test 14: Verify connected keyframes weights
        std::cout << "Test 14: Verifying connected keyframes weights..." << std::endl;
        if (keyframe2->KeyFrameGraph::connected_keyframes_weights.size() !=
            keyframe->KeyFrameGraph::connected_keyframes_weights.size()) {
            throw std::runtime_error("Connected keyframes weights size mismatch");
        }

        bool found_weight1 = false, found_weight2 = false;
        for (const auto &pair : keyframe2->KeyFrameGraph::connected_keyframes_weights) {
            if (pair.first->id == child1->id && pair.second == 10)
                found_weight1 = true;
            if (pair.first->id == child2->id && pair.second == 15)
                found_weight2 = true;
        }
        if (!found_weight1 || !found_weight2) {
            throw std::runtime_error("Connected keyframes weights not restored correctly");
        }
        std::cout << "Connected keyframes weights verified" << std::endl;

        // Test 15: Verify ordered keyframes weights
        std::cout << "Test 15: Verifying ordered keyframes weights..." << std::endl;
        if (keyframe2->KeyFrameGraph::ordered_keyframes_weights.size() !=
            keyframe->KeyFrameGraph::ordered_keyframes_weights.size()) {
            throw std::runtime_error("Ordered keyframes weights size mismatch");
        }

        bool found_ordered1 = false, found_ordered2 = false;
        for (const auto &pair : keyframe2->KeyFrameGraph::ordered_keyframes_weights) {
            if (pair.first->id == child1->id && pair.second == 10)
                found_ordered1 = true;
            if (pair.first->id == child2->id && pair.second == 15)
                found_ordered2 = true;
        }
        if (!found_ordered1 || !found_ordered2) {
            throw std::runtime_error("Ordered keyframes weights not restored correctly");
        }
        std::cout << "Ordered keyframes weights verified" << std::endl;

        // Test 16: Verify images
        std::cout << "Test 16: Verifying images..." << std::endl;
        if (!mats_equal_exact(keyframe->img, keyframe2->img)) {
            throw std::runtime_error("img mismatch");
        }
        if (!mats_equal_exact(keyframe->img_right, keyframe2->img_right)) {
            throw std::runtime_error("img_right mismatch");
        }
        if (!mats_equal_tolerant(keyframe->depth_img, keyframe2->depth_img)) {
            throw std::runtime_error("depth_img mismatch");
        }
        if (!mats_equal_exact(keyframe->semantic_img, keyframe2->semantic_img)) {
            throw std::runtime_error("semantic_img mismatch");
        }
        std::cout << "Images verified" << std::endl;

        // Test 17: Verify keypoints and descriptors
        std::cout << "Test 17: Verifying keypoints and descriptors..." << std::endl;
        if (keyframe2->kps.rows() != keyframe->kps.rows()) {
            throw std::runtime_error("kps dimensions mismatch");
        }
        if (!mats_equal_exact(keyframe->des, keyframe2->des)) {
            throw std::runtime_error("des mismatch");
        }
        std::cout << "Keypoints and descriptors verified" << std::endl;

        // Test 18: Verify map points
        std::cout << "Test 18: Verifying map points..." << std::endl;
        if (keyframe2->points.size() != keyframe->points.size()) {
            throw std::runtime_error("points size mismatch");
        }
        if (keyframe2->points[0] != mp1) {
            throw std::runtime_error("points[0] not restored correctly");
        }
        if (keyframe2->points[1] != mp2) {
            throw std::runtime_error("points[1] not restored correctly");
        }
        std::cout << "Map points verified" << std::endl;

        // Test 19: Test edge cases
        std::cout << "Test 19: Testing edge cases..." << std::endl;

        // Test with minimal KeyFrame
        KeyFramePtr minimal_kf = std::make_shared<KeyFrame>(9999);
        pybind11::tuple minimal_tuple = minimal_kf->state_tuple();
        KeyFramePtr minimal_kf2 = std::make_shared<KeyFrame>(9998); // Use explicit constructor
        minimal_kf2->restore_from_state(minimal_tuple);

        if (minimal_kf2->kid != minimal_kf->kid) {
            throw std::runtime_error("Minimal KeyFrame kid mismatch");
        }
        if (minimal_kf2->KeyFrameGraph::children.size() != 0) {
            throw std::runtime_error("Minimal KeyFrame should have no children");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 20: Test error handling
        std::cout << "Test 20: Testing error handling..." << std::endl;

        // Test with invalid tuple
        try {
            pybind11::tuple invalid_tuple = pybind11::make_tuple(1, 2, 3);
            KeyFramePtr test_kf = std::make_shared<KeyFrame>(9997); // Use explicit constructor
            test_kf->restore_from_state(invalid_tuple);
            throw std::runtime_error("Should have thrown exception for invalid tuple");
        } catch (const std::exception &e) {
            // Expected
            std::cout << "Correctly caught invalid tuple: " << e.what() << std::endl;
        }

        std::cout << "\nAll KeyFrame numpy serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

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
#include "dictionary.h"
#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    try {
        std::cout << "Starting KeyFrame JSON serialization tests..." << std::endl;

        // Create test camera
        ConfigDict camera_config;
        ConfigDict cam_settings;
        cam_settings["Camera.width"] = 640;
        cam_settings["Camera.height"] = 480;
        cam_settings["Camera.fx"] = 525.0;
        cam_settings["Camera.fy"] = 525.0;
        cam_settings["Camera.cx"] = 320.0;
        cam_settings["Camera.cy"] = 240.0;
        cam_settings["Camera.bf"] = 0.1;
        camera_config["cam_settings"] = cam_settings;
        auto camera = std::make_shared<PinholeCamera>(camera_config);

        // Create test pose
        Eigen::Matrix4d pose_matrix = Eigen::Matrix4d::Identity();
        pose_matrix(0, 3) = 1.0; // translation x
        pose_matrix(1, 3) = 2.0; // translation y
        pose_matrix(2, 3) = 3.0; // translation z
        CameraPose pose(pose_matrix);

        // Create test images
        cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat img_right = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat depth_img = cv::Mat::zeros(480, 640, CV_32F);
        cv::Mat semantic_img = cv::Mat::zeros(480, 640, CV_8UC1);

        fill_test_mat(img);
        fill_test_mat(img_right);
        fill_test_mat(depth_img);
        fill_test_mat(semantic_img);

        // Create Frame first
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

        // Create KeyFrame from Frame
        KeyFramePtr keyframe = std::make_shared<KeyFrame>(frame, img, img_right, depth_img, 2001);

        // Set KeyFrame-specific properties
        keyframe->_is_bad = false;
        keyframe->to_be_erased = false;
        keyframe->lba_count = 3;

        // Set pose relative to parent
        Eigen::Matrix4d Tcp_matrix = Eigen::Matrix4d::Identity();
        Tcp_matrix(0, 3) = 0.5; // small translation
        Tcp_matrix(1, 3) = 0.3;
        Tcp_matrix(2, 3) = 0.1;
        keyframe->_pose_Tcp.set_from_matrix(Tcp_matrix);

        // Set loop closing properties
        // keyframe->g_des = cv::Mat(1, 64, CV_8U);
        // fill_test_mat(keyframe->g_des);
        keyframe->loop_query_id = 1001;
        keyframe->num_loop_words = 50;
        keyframe->loop_score = 0.85f;

        // Set relocalization properties
        keyframe->reloc_query_id = 2001;
        keyframe->num_reloc_words = 30;
        keyframe->reloc_score = 0.75f;

        // Set GBA properties
        keyframe->GBA_kf_id = 3001;
        keyframe->is_Tcw_GBA_valid = true;
        keyframe->Tcw_GBA = Eigen::Matrix4d::Identity();
        keyframe->Tcw_GBA(0, 3) = 1.1;
        keyframe->Tcw_GBA(1, 3) = 2.1;
        keyframe->Tcw_GBA(2, 3) = 3.1;
        keyframe->Tcw_before_GBA = Eigen::Matrix4d::Identity();
        keyframe->Tcw_before_GBA(0, 3) = 1.0;
        keyframe->Tcw_before_GBA(1, 3) = 2.0;
        keyframe->Tcw_before_GBA(2, 3) = 3.0;

        // Create additional KeyFrames for testing relationships
        auto parent_kf = std::make_shared<KeyFrame>(3001);
        auto child_kf1 = std::make_shared<KeyFrame>(3002);
        auto child_kf2 = std::make_shared<KeyFrame>(3003);
        auto loop_kf = std::make_shared<KeyFrame>(3004);
        auto connected_kf1 = std::make_shared<KeyFrame>(3005);
        auto connected_kf2 = std::make_shared<KeyFrame>(3006);

        // Set up KeyFrameGraph relationships
        keyframe->KeyFrameGraph::init_parent = true;
        keyframe->KeyFrameGraph::parent = parent_kf;
        keyframe->KeyFrameGraph::children.insert(child_kf1);
        keyframe->KeyFrameGraph::children.insert(child_kf2);
        keyframe->KeyFrameGraph::loop_edges.insert(loop_kf);
        keyframe->KeyFrameGraph::not_to_erase = true;
        keyframe->KeyFrameGraph::is_first_connection = false;

        // Set up covisibility connections
        keyframe->KeyFrameGraph::connected_keyframes_weights[connected_kf1] = 15;
        keyframe->KeyFrameGraph::connected_keyframes_weights[connected_kf2] = 12;
        keyframe->KeyFrameGraph::connected_keyframes_weights[parent_kf] = 20;

        keyframe->KeyFrameGraph::ordered_keyframes_weights.push_back({parent_kf, 20});
        keyframe->KeyFrameGraph::ordered_keyframes_weights.push_back({connected_kf1, 15});
        keyframe->KeyFrameGraph::ordered_keyframes_weights.push_back({connected_kf2, 12});

        std::cout << "Created test KeyFrame with comprehensive data" << std::endl;

        // Test 1: Serialize to JSON
        std::cout << "Test 1: Serializing KeyFrame to JSON..." << std::endl;
        const std::string json_str = keyframe->to_json();
        if (json_str.empty()) {
            throw std::runtime_error("JSON serialization returned empty string");
        }
        std::cout << "JSON serialization successful, length: " << json_str.length() << std::endl;

        // Test 2: Deserialize from JSON
        std::cout << "Test 2: Deserializing KeyFrame from JSON..." << std::endl;
        KeyFramePtr keyframe2 = KeyFrame::from_json(json_str);
        if (!keyframe2) {
            throw std::runtime_error("JSON deserialization returned null pointer");
        }
        std::cout << "JSON deserialization successful" << std::endl;

        // Test 3: Replace IDs with objects
        std::cout << "Test 3: Replacing IDs with objects..." << std::endl;
        std::vector<MapPointPtr> points{mp1, mp2};
        std::vector<FramePtr> frames{frame};
        std::vector<KeyFramePtr> keyframes{parent_kf, child_kf1,     child_kf2,
                                           loop_kf,   connected_kf1, connected_kf2};
        keyframe2->replace_ids_with_objects(points, frames, keyframes);
        std::cout << "ID replacement successful" << std::endl;

        // Test 4: Verify basic Frame properties (inherited)
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
        if (std::abs(keyframe2->median_depth - keyframe->median_depth) > 1e-6f) {
            throw std::runtime_error("median_depth mismatch");
        }
        if (keyframe2->is_blurry != keyframe->is_blurry) {
            throw std::runtime_error("is_blurry mismatch");
        }
        if (std::abs(keyframe2->laplacian_var - keyframe->laplacian_var) > 1e-6f) {
            throw std::runtime_error("laplacian_var mismatch");
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
        if (keyframe2->to_be_erased != keyframe->to_be_erased) {
            throw std::runtime_error("to_be_erased mismatch");
        }
        if (keyframe2->lba_count != keyframe->lba_count) {
            throw std::runtime_error("lba_count mismatch");
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

        // Debug g_des information
        // std::cout << "Original g_des: " << keyframe->g_des.rows << "x" << keyframe->g_des.cols
        //           << ", type: " << keyframe->g_des.type()
        //           << ", channels: " << keyframe->g_des.channels() << std::endl;
        // std::cout << "Deserialized g_des: " << keyframe2->g_des.rows << "x" <<
        // keyframe2->g_des.cols
        //           << ", type: " << keyframe2->g_des.type()
        //           << ", channels: " << keyframe2->g_des.channels() << std::endl;

        // if (keyframe->g_des.empty() && keyframe2->g_des.empty()) {
        //     std::cout << "Both g_des are empty - OK" << std::endl;
        // } else if (keyframe->g_des.empty() || keyframe2->g_des.empty()) {
        //     throw std::runtime_error("One g_des is empty, the other is not");
        // } else {
        //     // Compare first few elements for debugging
        //     std::cout << "First few elements comparison:" << std::endl;
        //     for (int i = 0; i < std::min(3, keyframe->g_des.rows); ++i) {
        //         for (int j = 0; j < std::min(3, keyframe->g_des.cols); ++j) {
        //             uint8_t orig = keyframe->g_des.at<uint8_t>(i, j);
        //             uint8_t deser = keyframe2->g_des.at<uint8_t>(i, j);
        //             std::cout << "  [" << i << "," << j << "]: " << (int)orig << " vs "
        //                       << (int)deser << std::endl;
        //         }
        //     }
        // }

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

        // Debug parent information
        std::cout << "Original parent: "
                  << (keyframe->KeyFrameGraph::parent
                          ? std::to_string(keyframe->KeyFrameGraph::parent->id)
                          : "null")
                  << std::endl;
        std::cout << "Deserialized parent: "
                  << (keyframe2->KeyFrameGraph::parent
                          ? std::to_string(keyframe2->KeyFrameGraph::parent->id)
                          : "null")
                  << std::endl;
        std::cout << "Expected parent ID: " << parent_kf->id << std::endl;

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
            if (child->id == child_kf1->id)
                found_child1 = true;
            if (child->id == child_kf2->id)
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
        bool found_loop = false;
        for (const auto &loop_edge : keyframe2->KeyFrameGraph::loop_edges) {
            if (loop_edge->id == loop_kf->id) {
                found_loop = true;
                break;
            }
        }
        if (!found_loop) {
            throw std::runtime_error("Loop edge not restored correctly");
        }
        std::cout << "Loop edges verified" << std::endl;

        // Test 14: Verify connected keyframes weights
        std::cout << "Test 14: Verifying connected keyframes weights..." << std::endl;
        if (keyframe2->KeyFrameGraph::connected_keyframes_weights.size() !=
            keyframe->KeyFrameGraph::connected_keyframes_weights.size()) {
            throw std::runtime_error("Connected keyframes weights size mismatch");
        }

        // Check specific connections
        bool found_conn1 = false, found_conn2 = false, found_parent_conn = false;
        for (const auto &pair : keyframe2->KeyFrameGraph::connected_keyframes_weights) {
            if (pair.first->id == connected_kf1->id && pair.second == 15)
                found_conn1 = true;
            if (pair.first->id == connected_kf2->id && pair.second == 12)
                found_conn2 = true;
            if (pair.first->id == parent_kf->id && pair.second == 20)
                found_parent_conn = true;
        }
        if (!found_conn1 || !found_conn2 || !found_parent_conn) {
            throw std::runtime_error("Connected keyframes weights not restored correctly");
        }
        std::cout << "Connected keyframes weights verified" << std::endl;

        // Test 15: Verify ordered keyframes weights
        std::cout << "Test 15: Verifying ordered keyframes weights..." << std::endl;
        if (keyframe2->KeyFrameGraph::ordered_keyframes_weights.size() !=
            keyframe->KeyFrameGraph::ordered_keyframes_weights.size()) {
            throw std::runtime_error("Ordered keyframes weights size mismatch");
        }

        // Check order and values
        if (keyframe2->KeyFrameGraph::ordered_keyframes_weights[0].first->id != parent_kf->id ||
            keyframe2->KeyFrameGraph::ordered_keyframes_weights[0].second != 20) {
            throw std::runtime_error("First ordered weight incorrect");
        }
        if (keyframe2->KeyFrameGraph::ordered_keyframes_weights[1].first->id != connected_kf1->id ||
            keyframe2->KeyFrameGraph::ordered_keyframes_weights[1].second != 15) {
            throw std::runtime_error("Second ordered weight incorrect");
        }
        if (keyframe2->KeyFrameGraph::ordered_keyframes_weights[2].first->id != connected_kf2->id ||
            keyframe2->KeyFrameGraph::ordered_keyframes_weights[2].second != 12) {
            throw std::runtime_error("Third ordered weight incorrect");
        }
        std::cout << "Ordered keyframes weights verified" << std::endl;

        // Test 16: Test edge cases
        std::cout << "Test 16: Testing edge cases..." << std::endl;

        // Test with KeyFrame without relationships
        KeyFramePtr isolated_kf =
            std::make_shared<KeyFrame>(frame, img, img_right, depth_img, 4001);
        std::string isolated_json = isolated_kf->to_json();
        KeyFramePtr isolated_kf2 = KeyFrame::from_json(isolated_json);

        if (isolated_kf2->kid != isolated_kf->kid) {
            throw std::runtime_error("Isolated KeyFrame kid mismatch");
        }
        if (isolated_kf2->KeyFrameGraph::parent != nullptr) {
            throw std::runtime_error("Isolated KeyFrame should have no parent");
        }
        if (!isolated_kf2->KeyFrameGraph::children.empty()) {
            throw std::runtime_error("Isolated KeyFrame should have no children");
        }
        std::cout << "Edge cases verified" << std::endl;

        // Test 17: Test JSON parsing robustness
        std::cout << "Test 17: Testing JSON parsing robustness..." << std::endl;

        // Test with malformed JSON
        try {
            KeyFrame::from_json("invalid json");
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
        minimal_json["is_keyframe"] = true;
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

        // KeyFrame-specific fields
        minimal_json["kid"] = 5001;
        minimal_json["_is_bad"] = false;
        minimal_json["lba_count"] = 0;
        minimal_json["to_be_erased"] = false;
        minimal_json["is_Tcw_GBA_valid"] = false;
        minimal_json["_pose_Tcp"] = nullptr;

        // Loop closing and relocalization fields
        // minimal_json["g_des"] = nullptr;
        minimal_json["loop_query_id"] = -1;
        minimal_json["num_loop_words"] = 0;
        minimal_json["loop_score"] = 0.0;
        minimal_json["reloc_query_id"] = -1;
        minimal_json["num_reloc_words"] = 0;
        minimal_json["reloc_score"] = 0.0;

        // GBA fields
        minimal_json["GBA_kf_id"] = 0;
        minimal_json["Tcw_GBA"] = std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0));
        minimal_json["Tcw_before_GBA"] =
            std::vector<std::vector<double>>(4, std::vector<double>(4, 0.0));

        // KeyFrameGraph fields
        minimal_json["init_parent"] = false;
        minimal_json["parent"] = nullptr;
        minimal_json["children"] = std::vector<int>();
        minimal_json["loop_edges"] = std::vector<int>();
        minimal_json["not_to_erase"] = false;
        minimal_json["connected_keyframes_weights"] = std::vector<std::vector<int>>();
        minimal_json["ordered_keyframes_weights"] = std::vector<std::vector<int>>();
        minimal_json["is_first_connection"] = true;

        KeyFramePtr minimal_kf = KeyFrame::from_json(minimal_json.dump());
        if (minimal_kf->id != 123) {
            throw std::runtime_error("Minimal KeyFrame ID mismatch");
        }
        if (minimal_kf->kid != 5001) {
            throw std::runtime_error("Minimal KeyFrame kid mismatch");
        }
        std::cout << "JSON parsing robustness verified" << std::endl;

        std::cout << "\nAll KeyFrame JSON serialization tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

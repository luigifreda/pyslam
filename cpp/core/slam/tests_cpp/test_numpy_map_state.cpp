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

#include <pybind11/embed.h>

#include "map.h"
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};

    try {
        std::cout << "Starting MapStateData numpy serialization tests..." << std::endl;

        // Test 1: Empty MapStateData serialization
        std::cout << "\nTest 1: Empty MapStateData serialization..." << std::endl;
        {
            MapStateData map_state;
            pybind11::tuple state_tuple = map_state.state_tuple();

            if (state_tuple.size() != 11) {
                throw std::runtime_error("Expected 11 elements in tuple, got " +
                                         std::to_string(state_tuple.size()));
            }

            MapStateData restored_map_state;
            restored_map_state.restore_from_state(state_tuple);

            if (!restored_map_state.poses.empty()) {
                throw std::runtime_error("Expected empty poses");
            }
            if (!restored_map_state.pose_timestamps.empty()) {
                throw std::runtime_error("Expected empty pose_timestamps");
            }
            if (!restored_map_state.fov_centers.empty()) {
                throw std::runtime_error("Expected empty fov_centers");
            }
            if (!restored_map_state.fov_centers_colors.empty()) {
                throw std::runtime_error("Expected empty fov_centers_colors");
            }
            if (!restored_map_state.points.empty()) {
                throw std::runtime_error("Expected empty points");
            }
            if (!restored_map_state.colors.empty()) {
                throw std::runtime_error("Expected empty colors");
            }
            if (!restored_map_state.semantic_colors.empty()) {
                throw std::runtime_error("Expected empty semantic_colors");
            }
            if (!restored_map_state.covisibility_graph.empty()) {
                throw std::runtime_error("Expected empty covisibility_graph");
            }
            if (!restored_map_state.spanning_tree.empty()) {
                throw std::runtime_error("Expected empty spanning_tree");
            }
            if (!restored_map_state.loops.empty()) {
                throw std::runtime_error("Expected empty loops");
            }
        }
        std::cout << "✅ Test 1 passed" << std::endl;

        // Test 2: Populated MapStateData serialization
        std::cout << "\nTest 2: Populated MapStateData serialization..." << std::endl;
        {
            MapStateData map_state;

            // Add sample poses (4x4 matrices)
            int num_poses = 3;
            for (int i = 0; i < num_poses; ++i) {
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose(0, 3) = i * 1.0; // Translation in x
                pose(1, 3) = i * 0.5;
                pose(2, 3) = i * 0.2;
                map_state.poses.push_back(pose);
                map_state.pose_timestamps.push_back(i * 0.1);
            }

            // Add sample FOV centers (Vec3d)
            map_state.fov_centers.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
            map_state.fov_centers.push_back(Eigen::Vector3d(2.0, 3.0, 4.0));
            map_state.fov_centers_colors.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
            map_state.fov_centers_colors.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));

            // Add sample points (Vec3d)
            map_state.points.push_back(Eigen::Vector3d(10.0, 20.0, 30.0));
            map_state.points.push_back(Eigen::Vector3d(11.0, 21.0, 31.0));
            map_state.points.push_back(Eigen::Vector3d(12.0, 22.0, 32.0));

            // Add sample colors (Vec3f)
            map_state.colors.push_back(Eigen::Vector3f(0.5f, 0.3f, 0.8f));
            map_state.colors.push_back(Eigen::Vector3f(0.9f, 0.1f, 0.2f));
            map_state.colors.push_back(Eigen::Vector3f(0.2f, 0.7f, 0.3f));

            // Add sample semantic colors (Vec3f)
            map_state.semantic_colors.push_back(Eigen::Vector3f(0.1f, 0.2f, 0.3f));
            map_state.semantic_colors.push_back(Eigen::Vector3f(0.4f, 0.5f, 0.6f));
            map_state.semantic_colors.push_back(Eigen::Vector3f(0.7f, 0.8f, 0.9f));

            // Add sample covisibility graph (Vec6d)
            Eigen::Matrix<double, 6, 1> vec1;
            vec1 << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;
            map_state.covisibility_graph.push_back(vec1);

            Eigen::Matrix<double, 6, 1> vec2;
            vec2 << 1.0, 0.0, 3.0, 4.0, 5.0, 6.0;
            map_state.covisibility_graph.push_back(vec2);

            // Add sample spanning tree (Vec6d)
            Eigen::Matrix<double, 6, 1> vec3;
            vec3 << 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;
            map_state.spanning_tree.push_back(vec3);

            // Add sample loops (Vec6d)
            Eigen::Matrix<double, 6, 1> vec4;
            vec4 << 20.0, 21.0, 22.0, 23.0, 24.0, 25.0;
            map_state.loops.push_back(vec4);

            // Serialize
            pybind11::tuple state_tuple = map_state.state_tuple();

            // Deserialize
            MapStateData restored_map_state;
            restored_map_state.restore_from_state(state_tuple);

            // Verify poses
            if (restored_map_state.poses.size() != num_poses) {
                throw std::runtime_error("Pose count mismatch");
            }
            if (!vectors_equal(restored_map_state.poses, map_state.poses)) {
                throw std::runtime_error("Poses mismatch");
            }

            // Verify pose timestamps
            if (!vectors_equal(restored_map_state.pose_timestamps, map_state.pose_timestamps)) {
                throw std::runtime_error("Pose timestamps mismatch");
            }

            // Verify FOV centers
            if (restored_map_state.fov_centers.size() != 2) {
                throw std::runtime_error("FOV centers count mismatch");
            }
            if (!vectors_equal(restored_map_state.fov_centers, map_state.fov_centers)) {
                throw std::runtime_error("FOV centers mismatch");
            }

            // Verify FOV center colors
            if (!vectors_equal(restored_map_state.fov_centers_colors,
                               map_state.fov_centers_colors)) {
                throw std::runtime_error("FOV center colors mismatch");
            }

            // Verify points
            if (restored_map_state.points.size() != 3) {
                throw std::runtime_error("Points count mismatch");
            }
            if (!vectors_equal(restored_map_state.points, map_state.points)) {
                throw std::runtime_error("Points mismatch");
            }

            // Verify colors
            if (!vectors_equal(restored_map_state.colors, map_state.colors)) {
                throw std::runtime_error("Colors mismatch");
            }

            // Verify semantic colors
            if (!vectors_equal(restored_map_state.semantic_colors, map_state.semantic_colors)) {
                throw std::runtime_error("Semantic colors mismatch");
            }

            // Verify covisibility graph
            if (restored_map_state.covisibility_graph.size() != 2) {
                throw std::runtime_error("Covisibility graph count mismatch");
            }
            if (!vectors_equal(restored_map_state.covisibility_graph,
                               map_state.covisibility_graph)) {
                throw std::runtime_error("Covisibility graph mismatch");
            }

            // Verify spanning tree
            if (restored_map_state.spanning_tree.size() != 1) {
                throw std::runtime_error("Spanning tree count mismatch");
            }
            if (!vectors_equal(restored_map_state.spanning_tree, map_state.spanning_tree)) {
                throw std::runtime_error("Spanning tree mismatch");
            }

            // Verify loops
            if (restored_map_state.loops.size() != 1) {
                throw std::runtime_error("Loops count mismatch");
            }
            if (!vectors_equal(restored_map_state.loops, map_state.loops)) {
                throw std::runtime_error("Loops mismatch");
            }
        }
        std::cout << "✅ Test 2 passed" << std::endl;

        // Test 3: Wrong version error handling
        std::cout << "\nTest 3: Wrong version error handling..." << std::endl;
        {
            MapStateData map_state;
            pybind11::tuple state_tuple = map_state.state_tuple();

            // Create a modified tuple with wrong version by unpacking and rebuilding
            pybind11::tuple modified_tuple =
                pybind11::make_tuple(999,            // Wrong version
                                     state_tuple[1], // poses
                                     state_tuple[2], // pose_timestamps
                                     state_tuple[3], // fov_centers
                                     state_tuple[4], // fov_centers_colors
                                     state_tuple[5], // points
                                     state_tuple[6], // colors
                                     state_tuple[7], // semantic_colors
                                     state_tuple[8], // covisibility_graph
                                     state_tuple[9], // spanning_tree
                                     state_tuple[10] // loops
                );

            // Try to restore with wrong version
            MapStateData restored_map_state;

            try {
                restored_map_state.restore_from_state(modified_tuple);
                throw std::runtime_error("Should have thrown exception for wrong version");
            } catch (const std::runtime_error &e) {
                if (std::string(e.what()).find("Unsupported MapStateData pickle version") ==
                    std::string::npos) {
                    throw std::runtime_error("Wrong error message: " + std::string(e.what()));
                }
            }
        }
        std::cout << "✅ Test 3 passed" << std::endl;

        // Test 4: Large dataset serialization
        std::cout << "\nTest 4: Large dataset serialization..." << std::endl;
        {
            MapStateData map_state;

            int num_poses = 100;
            int num_points = 1000;

            // Add many poses
            for (int i = 0; i < num_poses; ++i) {
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose(0, 3) = i * 0.5;
                pose(1, 3) = i * 0.3;
                pose(2, 3) = i * 0.1;
                map_state.poses.push_back(pose);
                map_state.pose_timestamps.push_back(i * 0.1);
            }

            // Add many points
            for (int i = 0; i < num_points; ++i) {
                map_state.points.push_back(Eigen::Vector3d(static_cast<double>(i),
                                                           static_cast<double>(i * 2),
                                                           static_cast<double>(i * 3)));
                map_state.colors.push_back(Eigen::Vector3f(static_cast<float>(i) / num_points,
                                                           static_cast<float>(i * 2) / num_points,
                                                           static_cast<float>(i * 3) / num_points));
                map_state.semantic_colors.push_back(Eigen::Vector3f(
                    static_cast<float>(i) / num_points, static_cast<float>(i * 2) / num_points,
                    static_cast<float>(i * 3) / num_points));
            }

            // Add FOV centers
            for (int i = 0; i < num_poses; ++i) {
                map_state.fov_centers.push_back(Eigen::Vector3d(
                    static_cast<double>(i), static_cast<double>(i), static_cast<double>(i)));
                map_state.fov_centers_colors.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
            }

            // Add covisibility graph
            for (int i = 0; i < num_poses / 2; ++i) {
                Eigen::Matrix<double, 6, 1> vec;
                vec << static_cast<double>(i), static_cast<double>(i + 1), 0.0, 0.0, 0.0, 0.0;
                map_state.covisibility_graph.push_back(vec);
            }

            // Serialize
            pybind11::tuple state_tuple = map_state.state_tuple();

            // Deserialize
            MapStateData restored_map_state;
            restored_map_state.restore_from_state(state_tuple);

            // Verify counts
            if (restored_map_state.poses.size() != num_poses) {
                throw std::runtime_error("Large dataset: pose count mismatch");
            }
            if (restored_map_state.points.size() != num_points) {
                throw std::runtime_error("Large dataset: points count mismatch");
            }
            if (restored_map_state.colors.size() != num_points) {
                throw std::runtime_error("Large dataset: colors count mismatch");
            }
            if (restored_map_state.fov_centers.size() != num_poses) {
                throw std::runtime_error("Large dataset: fov_centers count mismatch");
            }

            // Verify random samples
            std::vector<int> check_indices = {0, 10, 50, 90};
            for (int idx : check_indices) {
                if (idx < num_poses) {
                    if ((restored_map_state.poses[idx] - map_state.poses[idx])
                            .lpNorm<Eigen::Infinity>() > 1e-10) {
                        throw std::runtime_error("Large dataset: pose mismatch at index " +
                                                 std::to_string(idx));
                    }
                }
                if (idx < num_points) {
                    if ((restored_map_state.points[idx] - map_state.points[idx])
                            .lpNorm<Eigen::Infinity>() > 1e-10) {
                        throw std::runtime_error("Large dataset: point mismatch at index " +
                                                 std::to_string(idx));
                    }
                    if ((restored_map_state.colors[idx] - map_state.colors[idx])
                            .lpNorm<Eigen::Infinity>() > 1e-5f) {
                        throw std::runtime_error("Large dataset: color mismatch at index " +
                                                 std::to_string(idx));
                    }
                }
            }
        }
        std::cout << "✅ Test 4 passed" << std::endl;

        // Test 5: Tuple structure verification
        std::cout << "\nTest 5: Tuple structure verification..." << std::endl;
        {
            MapStateData map_state;
            map_state.points.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
            map_state.colors.push_back(Eigen::Vector3f(0.5f, 0.3f, 0.1f));

            pybind11::tuple state_tuple = map_state.state_tuple();

            if (state_tuple.size() != 11) {
                throw std::runtime_error("Expected 11 elements in tuple");
            }

            // Check version
            int version = state_tuple[0].cast<int>();
            if (version != 1) {
                throw std::runtime_error("Expected version 1, got " + std::to_string(version));
            }

            std::cout << "Version: " << version << std::endl;
        }
        std::cout << "✅ Test 5 passed" << std::endl;

        std::cout << "\n✅ All MapStateData numpy serialization tests passed successfully!"
                  << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
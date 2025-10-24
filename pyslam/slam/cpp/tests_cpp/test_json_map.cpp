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

#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "utils/test_utils.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

namespace pyslam {

void test_map_json_serialization() {
    std::cout << "Testing Map JSON serialization..." << std::endl;

    // Initialize FeatureSharedResources to avoid crashes
    test_utils::init_feature_shared_info();

    // Create a test map
    auto map = std::make_shared<Map>();

    // Create a test camera
    auto camera = test_utils::create_test_camera();

    // Create a test frame
    auto frame = FrameNewPtr(camera, cv::Mat(), cv::Mat(), cv::Mat(), CameraPose(), 0, 0.0, 0);

    // Create a test keyframe
    auto keyframe = KeyFrameNewPtr(frame);
    keyframe->kid = 0;

    // Initialize some basic data to avoid crashes
    keyframe->octaves.resize(1, 0);               // At least one octave entry
    keyframe->des = cv::Mat::zeros(1, 32, CV_8U); // Basic descriptor

    // Create a test map point
    Eigen::Vector3d point_pos(1.0, 2.0, 3.0);
    Eigen::Matrix<unsigned char, 3, 1> color;
    color << 255, 128, 64;
    auto map_point = MapPointNewPtr(point_pos, color, keyframe, 0, 0);

    // Add objects to map
    map->add_frame(frame, true);
    map->add_keyframe(keyframe);
    map->add_point(map_point);

    // Test serialization
    std::string json_str = map->to_json();
    std::cout << "Serialized JSON length: " << json_str.length() << " characters" << std::endl;

    // Test deserialization
    auto new_map = std::make_shared<Map>();
    new_map->from_json(json_str);

    // Verify basic properties
    assert(new_map->num_frames() == map->num_frames());
    assert(new_map->num_keyframes() == map->num_keyframes());
    assert(new_map->num_points() == map->num_points());

    // Verify metadata
    assert(new_map->max_frame_id == map->max_frame_id);
    assert(new_map->max_keyframe_id == map->max_keyframe_id);
    assert(new_map->max_point_id == map->max_point_id);

    std::cout << "Map JSON serialization tests passed." << std::endl;
}

} // namespace pyslam

int main() {
    try {
        pyslam::test_map_json_serialization();
        std::cout << "All Map serialization tests passed!" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

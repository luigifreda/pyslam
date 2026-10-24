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
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "utils/serialization_json.h"
#include "utils/test_utils.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace pyslam {

void test_cpp_to_python_map_serialization() {
    std::cout << "Testing C++-to-Python Map JSON serialization..." << std::endl;

    // Initialize FeatureSharedResources to avoid crashes
    test_utils::init_feature_shared_info();

    // Create a test map
    auto map = std::make_shared<Map>();

    // Create a test camera
    auto camera = test_utils::create_test_camera();

    // Enable image storage for frames (required for serialization test)
    // Note: Frame::is_store_imgs is a static member that controls whether images are stored
    Frame::is_store_imgs = true;

    // Create some keyframes with descriptors and images
    std::vector<KeyFramePtr> keyframes;
    const int num_keyframes = 5;
    const int num_kps_per_kf = 100;

    for (int i = 0; i < num_keyframes; ++i) {
        // Create a test image with some texture/patterns
        cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
        // Add checkerboard pattern
        const int checker_size = 40;
        for (int y = 0; y < 480; y += checker_size) {
            for (int x = 0; x < 640; x += checker_size) {
                if ((x / checker_size + y / checker_size) % 2 == 0) {
                    img(cv::Rect(x, y, checker_size, checker_size)) = cv::Scalar(255, 255, 255);
                }
            }
        }
        // Add some noise for more features
        cv::Mat noise(480, 640, CV_8UC3);
        cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
        img = img + noise;

        // Create a frame with the image
        CameraPose pose;
        auto frame = FrameNewPtr(camera, img, cv::Mat(), cv::Mat(), pose, i, i * 0.1, i);

        // Initialize keypoints and descriptors
        frame->kps.resize(num_kps_per_kf, 2);
        frame->kpsu.resize(num_kps_per_kf, 2);
        frame->kpsn.resize(num_kps_per_kf, 2);
        frame->octaves.resize(num_kps_per_kf);
        frame->sizes.resize(num_kps_per_kf);
        frame->angles.resize(num_kps_per_kf);
        frame->des = cv::Mat::zeros(num_kps_per_kf, 32, CV_8U);
        frame->points.resize(num_kps_per_kf, nullptr);

        // Fill with test data
        std::mt19937 gen(i); // Deterministic seed
        std::uniform_real_distribution<float> kp_dist(0.0f, 640.0f);
        std::uniform_int_distribution<int> octave_dist(0, 7);
        std::uniform_real_distribution<float> size_dist(5.0f, 15.0f);
        std::uniform_real_distribution<float> angle_dist(0.0f, 360.0f);

        for (int j = 0; j < num_kps_per_kf; ++j) {
            frame->kps(j, 0) = kp_dist(gen);
            frame->kps(j, 1) = kp_dist(gen);
            frame->kpsu(j, 0) = frame->kps(j, 0);
            frame->kpsu(j, 1) = frame->kps(j, 1);
            frame->kpsn(j, 0) = frame->kps(j, 0);
            frame->kpsn(j, 1) = frame->kps(j, 1);
            frame->octaves[j] = octave_dist(gen);
            frame->sizes[j] = size_dist(gen);
            frame->angles[j] = angle_dist(gen);

            // Create ORB descriptors (32 bytes each)
            for (int k = 0; k < 32; ++k) {
                frame->des.at<unsigned char>(j, k) =
                    static_cast<unsigned char>((i * 1000 + j * 7 + k * 13) % 256);
            }
        }

        // Create keyframe
        auto keyframe = KeyFrameNewPtr(frame);
        keyframe->kid = i;
        keyframe->is_keyframe = true;
        keyframe->_is_bad = false;

        // Add to map
        map->add_frame(frame, true);
        map->add_keyframe(keyframe);
        keyframes.push_back(keyframe);
    }

    // Create map points and associate them with keyframes
    const int num_points = 20;
    std::vector<MapPointPtr> map_points;

    for (int i = 0; i < num_points; ++i) {
        // Select a keyframe to associate this point with
        int kf_idx = i % num_keyframes;
        auto kf = keyframes[kf_idx];

        // Select a keypoint index in this keyframe
        int kp_idx = i % num_kps_per_kf;

        // Random 3D position
        std::mt19937 gen(i);
        std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
        Eigen::Vector3d pt(pos_dist(gen), pos_dist(gen), pos_dist(gen));

        // Random color
        std::uniform_int_distribution<int> color_dist(0, 255);
        Eigen::Matrix<unsigned char, 3, 1> color;
        color << static_cast<unsigned char>(color_dist(gen)),
            static_cast<unsigned char>(color_dist(gen)),
            static_cast<unsigned char>(color_dist(gen));

        // Create map point
        auto mp = MapPointNewPtr(pt, color, kf, kp_idx, i);

        // Create descriptor
        mp->des = cv::Mat::zeros(1, 32, CV_8U);
        for (int j = 0; j < 32; ++j) {
            mp->des.at<unsigned char>(0, j) = static_cast<unsigned char>((i * 7 + j * 13) % 256);
        }

        // Add to map
        map->add_point(mp);
        map_points.push_back(mp);

        // CRITICAL: Associate the map point with the keyframe at the specific keypoint index
        // Ensure points array is large enough
        if (kf->points.size() <= static_cast<size_t>(kp_idx)) {
            kf->points.resize(kp_idx + 1, nullptr);
        }
        kf->points[kp_idx] = mp;

        // Also add observation to map point (bidirectional relationship)
        mp->add_observation(kf, kp_idx);
    }

    // Update max IDs
    map->max_frame_id = 10;
    map->max_keyframe_id = 10;
    map->max_point_id = 20;

    // Sync static ID counters
    FrameBase::set_id(10);
    MapPointBase::set_id(20);

    std::cout << "Created map with " << map->num_frames() << " frames, " << map->num_keyframes()
              << " keyframes, " << map->num_points() << " points" << std::endl;

    // Verify associations before saving
    std::cout << "\nVerifying map point associations..." << std::endl;
    for (const auto &kf : keyframes) {
        auto kf_points = kf->get_points();
        int num_associated = 0;
        for (const auto &p : kf_points) {
            if (p && !p->is_bad()) {
                num_associated++;
            }
        }
        std::cout << "  Keyframe " << kf->id << ": " << num_associated
                  << " map points associated out of " << kf_points.size() << " keypoints"
                  << std::endl;
    }

    // Save map to JSON file
    std::string output_dir = std::string(__FILE__);
    size_t last_slash = output_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        output_dir = output_dir.substr(0, last_slash);
    }
    output_dir += "/../tests_py/test_data";

    // Create directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    std::string output_file = output_dir + "/cpp_saved_map.json";
    std::cout << "\nSaving map to " << output_file << "..." << std::endl;

    std::string json_str = map->to_json();
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << json_str;
        file.close();
        std::cout << "Map saved successfully. File size: " << json_str.length() << " characters"
                  << std::endl;
    } else {
        throw std::runtime_error("Failed to open file for writing: " + output_file);
    }

    std::cout << "\nMap summary:" << std::endl;
    std::cout << "  Frames: " << map->num_frames() << std::endl;
    std::cout << "  Keyframes: " << map->num_keyframes() << std::endl;
    std::cout << "  Points: " << map->num_points() << std::endl;
    std::cout << "  max_frame_id: " << map->max_frame_id << std::endl;
    std::cout << "  max_keyframe_id: " << map->max_keyframe_id << std::endl;
    std::cout << "  max_point_id: " << map->max_point_id << std::endl;

    std::cout << "\nTest map saved to: " << output_file << std::endl;
    std::cout << "You can now run the Python test to load this map." << std::endl;
}

} // namespace pyslam

int main() {
    try {
        pyslam::test_cpp_to_python_map_serialization();
        std::cout << "\nC++-to-Python map serialization test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

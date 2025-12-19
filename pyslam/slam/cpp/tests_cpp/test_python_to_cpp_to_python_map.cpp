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
 * but WITHOUT EVEN THE IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License for more details.
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
#include <sstream>
#include <string>

namespace pyslam {

void test_python_to_cpp_to_python_map_serialization() {
    std::cout << "Testing Python-to-C++-to-Python Map JSON serialization..." << std::endl;

    // Initialize FeatureSharedResources to avoid crashes
    test_utils::init_feature_shared_info();

    // Find the Python-saved map file
    std::string map_file;
    bool found = false;

    // First, check environment variable
    const char *env_path = std::getenv("PYSLAM_TEST_MAP_FILE");
    if (env_path != nullptr) {
        std::ifstream test_file(env_path);
        if (test_file.good()) {
            map_file = env_path;
            found = true;
            test_file.close();
        }
    }

    // If not found, try paths relative to source file
    if (!found) {
        std::string source_file = __FILE__;
        size_t last_slash = source_file.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            source_file = source_file.substr(0, last_slash);
        }

        std::vector<std::string> possible_paths = {
            source_file + "/../tests_py/test_data/python_saved_map.json",
            source_file + "/../../tests_py/test_data/python_saved_map.json",
            "pyslam/slam/cpp/tests_py/test_data/python_saved_map.json",
        };

        for (const auto &path : possible_paths) {
            std::ifstream test_file(path);
            if (test_file.good()) {
                map_file = path;
                found = true;
                test_file.close();
                break;
            }
        }
    }

    if (!found) {
        std::cerr << "ERROR: Python-saved map file not found." << std::endl;
        if (env_path != nullptr) {
            std::cerr << "  Environment variable PYSLAM_TEST_MAP_FILE: " << env_path << std::endl;
        }
        std::cerr << "  Tried paths relative to source file and executable directory." << std::endl;
        std::cerr
            << "Please run test_python_to_cpp_map_serialization.py first to generate the test map."
            << std::endl;
        throw std::runtime_error("Test map file not found");
    }

    std::cout << "Loading Python-saved map from: " << map_file << std::endl;

    // Read JSON file
    std::ifstream file(map_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open map file: " + map_file);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    file.close();

    std::cout << "JSON file size: " << json_str.length() << " characters" << std::endl;

    // Load map from JSON (Python → C++)
    auto map = std::make_shared<Map>();
    try {
        map->from_json(json_str);
        std::cout << "Map loaded successfully from Python JSON!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: Failed to load map: " << e.what() << std::endl;
        throw;
    }

    // Verify basic properties
    int num_frames = map->num_frames();
    int num_keyframes = map->num_keyframes();
    int num_points = map->num_points();

    std::cout << "\nLoaded map statistics:" << std::endl;
    std::cout << "  Frames: " << num_frames << std::endl;
    std::cout << "  Keyframes: " << num_keyframes << std::endl;
    std::cout << "  Points: " << num_points << std::endl;
    std::cout << "  max_frame_id: " << map->max_frame_id << std::endl;
    std::cout << "  max_keyframe_id: " << map->max_keyframe_id << std::endl;
    std::cout << "  max_point_id: " << map->max_point_id << std::endl;

    // Verify we have data
    assert(num_keyframes > 0 && "Expected at least one keyframe");
    assert(num_points > 0 && "Expected at least one map point");

    // Save map back to JSON (C++ → Python)
    std::string output_dir = std::string(__FILE__);
    size_t last_slash = output_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        output_dir = output_dir.substr(0, last_slash);
    }
    output_dir += "/../tests_py/test_data";

    // Create directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    std::string output_file = output_dir + "/python_to_cpp_to_python_map.json";
    std::cout << "\nSaving map to " << output_file << " (C++ → Python)..." << std::endl;

    std::string output_json_str = map->to_json();
    std::ofstream out_file(output_file);
    if (out_file.is_open()) {
        out_file << output_json_str;
        out_file.close();
        std::cout << "Map saved successfully. File size: " << output_json_str.length()
                  << " characters" << std::endl;
    } else {
        throw std::runtime_error("Failed to open file for writing: " + output_file);
    }

    std::cout << "\nTest map saved to: " << output_file << std::endl;
    std::cout << "You can now run the Python test to load this map and verify the round-trip."
              << std::endl;

    std::cout << "\nPython-to-C++-to-Python map serialization test completed successfully!"
              << std::endl;
}

} // namespace pyslam

int main() {
    try {
        pyslam::test_python_to_cpp_to_python_map_serialization();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

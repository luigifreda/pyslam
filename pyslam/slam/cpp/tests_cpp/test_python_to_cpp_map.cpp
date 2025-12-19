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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace pyslam {

void test_python_to_cpp_map_loading() {
    std::cout << "Testing Python-to-C++ map loading..." << std::endl;

    // Initialize FeatureSharedResources to avoid crashes
    test_utils::init_feature_shared_info();

    // Path to Python-saved map
    std::string map_file;
    bool found = false;

    // Try to get path from environment variable first (set by test script)
    const char *env_path = std::getenv("PYSLAM_TEST_MAP_FILE");
    if (env_path != nullptr) {
        std::ifstream test_file(env_path);
        if (test_file.good()) {
            test_file.close();
            map_file = env_path;
            found = true;
        } else {
            test_file.close();
        }
    }

    if (!found) {
        // Compute path relative to source file location
        std::string source_file = __FILE__;
        std::string source_dir = source_file.substr(0, source_file.find_last_of("/\\"));
        std::string tests_py_dir = source_dir + "/../tests_py";
        std::string test_data_file = tests_py_dir + "/test_data/python_saved_map.json";

        // Try multiple possible paths (relative to source, relative to executable, absolute)
        std::vector<std::string> possible_paths = {
            test_data_file,                                   // Relative to source file
            "../tests_py/test_data/python_saved_map.json",    // Relative to build/tests_cpp
            "../../tests_py/test_data/python_saved_map.json", // From build/tests_cpp (alternative)
            "pyslam/slam/cpp/tests_py/test_data/python_saved_map.json", // From project root
        };

        for (const auto &path : possible_paths) {
            std::ifstream test_file(path);
            if (test_file.good()) {
                map_file = path;
                found = true;
                test_file.close();
                break;
            }
            test_file.close();
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

    std::cout << "Loading map from: " << map_file << std::endl;

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

    // Load map from JSON
    auto map = std::make_shared<Map>();
    try {
        map->from_json(json_str);
        std::cout << "Map loaded successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: Failed to load map: " << e.what() << std::endl;
        throw;
    }

    // Verify basic properties
    int num_frames = map->num_frames();
    int num_keyframes = map->num_keyframes();
    int num_points = map->num_points();

    std::cout << "Loaded map statistics:" << std::endl;
    std::cout << "  Frames: " << num_frames << std::endl;
    std::cout << "  Keyframes: " << num_keyframes << std::endl;
    std::cout << "  Points: " << num_points << std::endl;
    std::cout << "  max_frame_id: " << map->max_frame_id << std::endl;
    std::cout << "  max_keyframe_id: " << map->max_keyframe_id << std::endl;
    std::cout << "  max_point_id: " << map->max_point_id << std::endl;

    // Verify we have data
    assert(num_keyframes > 0 && "Map should have at least one keyframe");
    assert(num_points > 0 && "Map should have at least one map point");

    // Verify keyframes_map is populated
    assert(map->keyframes_map.size() == static_cast<size_t>(num_keyframes) &&
           "keyframes_map should have same size as keyframes");

    std::cout << "  keyframes_map size: " << map->keyframes_map.size() << std::endl;

    // Verify FrameBase::_id and MapPointBase::_id are restored
    int frame_base_id = FrameBase::next_id();
    int mappoint_base_id = MapPointBase::next_id();

    std::cout << "  FrameBase::_id: " << frame_base_id << std::endl;
    std::cout << "  MapPointBase::_id: " << mappoint_base_id << std::endl;

    assert(frame_base_id == map->max_frame_id && "FrameBase::_id should match max_frame_id");
    assert(mappoint_base_id == map->max_point_id && "MapPointBase::_id should match max_point_id");

    // Verify keyframes have descriptors
    auto keyframes = map->get_keyframes_vector();
    int keyframes_with_des = 0;
    int keyframes_with_valid_des = 0;
    int keyframes_with_matching_des_kps = 0;
    int total_descriptors = 0;

    for (const auto &kf : keyframes) {
        if (kf && !kf->des.empty() && kf->des.rows > 0) {
            keyframes_with_des++;

            // Check descriptor validity (not all zeros, correct type, correct dimensions)
            bool has_valid_des = true;
            if (kf->des.type() != CV_8U) {
                std::cerr << "WARNING: Keyframe " << kf->id
                          << " has descriptors with wrong type (expected CV_8U, got "
                          << kf->des.type() << ")" << std::endl;
                has_valid_des = false;
            }

            // Check descriptor dimensions (should be N x 32 for ORB)
            if (kf->des.cols != 32) {
                std::cerr << "WARNING: Keyframe " << kf->id
                          << " has descriptors with wrong width (expected 32, got " << kf->des.cols
                          << ")" << std::endl;
                has_valid_des = false;
            }

            // Check if descriptors match keypoint count
            int num_kps = kf->kps.rows();
            int num_des = kf->des.rows;
            if (num_kps > 0 && num_des != num_kps) {
                std::cerr << "WARNING: Keyframe " << kf->id
                          << " has mismatched descriptor/keypoint count (kps: " << num_kps
                          << ", des: " << num_des << ")" << std::endl;
                has_valid_des = false;
            } else if (num_kps == num_des && num_kps > 0) {
                keyframes_with_matching_des_kps++;
            }

            // Check if descriptors are all zeros (invalid)
            if (has_valid_des && num_des > 0) {
                cv::Scalar sum = cv::sum(kf->des);
                if (sum[0] == 0 && sum[1] == 0 && sum[2] == 0 && sum[3] == 0) {
                    std::cerr << "WARNING: Keyframe " << kf->id
                              << " has all-zero descriptors (likely invalid)" << std::endl;
                    has_valid_des = false;
                }
            }

            if (has_valid_des) {
                keyframes_with_valid_des++;
                total_descriptors += num_des;
            }
        }
    }

    std::cout << "  Keyframes with descriptors: " << keyframes_with_des << " / " << num_keyframes
              << std::endl;
    std::cout << "  Keyframes with valid descriptors: " << keyframes_with_valid_des << " / "
              << num_keyframes << std::endl;
    std::cout << "  Keyframes with matching descriptor/keypoint counts: "
              << keyframes_with_matching_des_kps << " / " << num_keyframes << std::endl;
    std::cout << "  Total descriptors across all keyframes: " << total_descriptors << std::endl;

    // Verify at least some keyframes have descriptors (for relocalization)
    if (keyframes_with_des == 0) {
        std::cerr << "WARNING: No keyframes have descriptors loaded!" << std::endl;
        std::cerr << "This will cause relocalization to fail." << std::endl;
    } else if (keyframes_with_valid_des == 0) {
        std::cerr << "WARNING: No keyframes have valid descriptors!" << std::endl;
        std::cerr << "This will cause relocalization to fail." << std::endl;
    } else if (keyframes_with_valid_des < num_keyframes) {
        std::cerr << "WARNING: Only " << keyframes_with_valid_des << " / " << num_keyframes
                  << " keyframes have valid descriptors!" << std::endl;
    }

    // Verify keyframes have keypoints
    int keyframes_with_kps = 0;
    for (const auto &kf : keyframes) {
        if (kf && kf->kps.rows() > 0) {
            keyframes_with_kps++;
        }
    }

    std::cout << "  Keyframes with keypoints: " << keyframes_with_kps << " / " << num_keyframes
              << std::endl;

    // Verify map points have descriptors
    auto points = map->get_points();
    int points_with_des = 0;
    int points_with_valid_des = 0;
    int total_point_descriptors = 0;

    for (const auto &mp : points) {
        if (mp && !mp->des.empty() && mp->des.rows > 0) {
            points_with_des++;

            // Check descriptor validity
            bool has_valid_des = true;
            if (mp->des.type() != CV_8U) {
                std::cerr << "WARNING: MapPoint " << mp->id
                          << " has descriptor with wrong type (expected CV_8U, got "
                          << mp->des.type() << ")" << std::endl;
                has_valid_des = false;
            }

            // Check descriptor dimensions (should be 1 x 32 for ORB)
            if (mp->des.cols != 32) {
                std::cerr << "WARNING: MapPoint " << mp->id
                          << " has descriptor with wrong width (expected 32, got " << mp->des.cols
                          << ")" << std::endl;
                has_valid_des = false;
            }

            // Check if descriptor is all zeros (invalid)
            if (has_valid_des && mp->des.rows > 0) {
                cv::Scalar sum = cv::sum(mp->des);
                if (sum[0] == 0 && sum[1] == 0 && sum[2] == 0 && sum[3] == 0) {
                    std::cerr << "WARNING: MapPoint " << mp->id
                              << " has all-zero descriptor (likely invalid)" << std::endl;
                    has_valid_des = false;
                }
            }

            if (has_valid_des) {
                points_with_valid_des++;
                total_point_descriptors += mp->des.rows;
            }
        }
    }

    std::cout << "  Map points with descriptors: " << points_with_des << " / " << num_points
              << std::endl;
    std::cout << "  Map points with valid descriptors: " << points_with_valid_des << " / "
              << num_points << std::endl;
    std::cout << "  Total descriptors across all map points: " << total_point_descriptors
              << std::endl;

    if (points_with_des > 0 && points_with_valid_des == 0) {
        std::cerr << "WARNING: No map points have valid descriptors!" << std::endl;
    } else if (points_with_valid_des < points_with_des) {
        std::cerr << "WARNING: Only " << points_with_valid_des << " / " << points_with_des
                  << " map points have valid descriptors!" << std::endl;
    }

    // Test that we can access keyframes via keyframes_map
    int accessible_keyframes = 0;
    for (const auto &kf : keyframes) {
        if (kf) {
            auto it = map->keyframes_map.find(kf->id);
            if (it != map->keyframes_map.end() && it->second == kf) {
                accessible_keyframes++;
            }
        }
    }

    std::cout << "  Accessible keyframes via keyframes_map: " << accessible_keyframes << " / "
              << num_keyframes << std::endl;
    assert(accessible_keyframes == num_keyframes &&
           "All keyframes should be accessible via keyframes_map");

    // Verify camera is loaded correctly
    if (!keyframes.empty() && keyframes[0]) {
        auto camera = keyframes[0]->camera;
        if (camera) {
            std::cout << "  Sample keyframe camera: " << camera->width << "x" << camera->height
                      << std::endl;
            assert(camera->width > 0 && camera->height > 0 &&
                   "Camera should have valid dimensions");
        }
    }

    // Verify images are loaded (critical for relocalization via compute_frame_matches)
    int keyframes_with_img = 0;
    for (const auto &kf : keyframes) {
        if (kf && !kf->img.empty() && kf->img.rows > 0 && kf->img.cols > 0) {
            keyframes_with_img++;
        }
    }
    std::cout << "  Keyframes with images: " << keyframes_with_img << " / " << num_keyframes
              << std::endl;

    // Warn if no images loaded (will cause relocalization to fail)
    if (keyframes_with_img == 0) {
        std::cerr << "WARNING: No keyframes have images loaded!" << std::endl;
        std::cerr << "This will cause compute_frame_matches to fail (returns 0 compared pairs)."
                  << std::endl;
        std::cerr << "Relocalization requires images for feature matching." << std::endl;
    } else {
        // Verify image dimensions match camera
        if (!keyframes.empty() && keyframes[0]) {
            auto kf = keyframes[0];
            if (kf->camera && !kf->img.empty()) {
                assert(kf->img.rows == kf->camera->height && kf->img.cols == kf->camera->width &&
                       "Image dimensions should match camera dimensions");
            }
        }
    }

    // Verify keyframes are not marked as bad (critical for relocalization)
    int bad_keyframes = 0;
    for (const auto &kf : keyframes) {
        if (kf && kf->is_bad()) {
            bad_keyframes++;
        }
    }
    std::cout << "  Bad keyframes: " << bad_keyframes << " / " << num_keyframes << std::endl;
    if (bad_keyframes == num_keyframes) {
        std::cerr << "WARNING: All keyframes are marked as bad!" << std::endl;
        std::cerr << "This will cause compute_frame_matches to skip all frames (returns 0 compared "
                     "pairs)."
                  << std::endl;
    }

    // Summary of relocalization prerequisites
    std::cout << "\nRelocalization prerequisites check:" << std::endl;
    bool has_valid_descriptors = keyframes_with_valid_des > 0;
    bool has_matching_des_kps = keyframes_with_matching_des_kps > 0;
    bool has_keypoints = keyframes_with_kps > 0;
    bool has_images = keyframes_with_img > 0;
    bool has_good_keyframes = bad_keyframes < num_keyframes;

    std::cout << "  ✓ Valid descriptors: " << (has_valid_descriptors ? "OK" : "MISSING") << " ("
              << keyframes_with_valid_des << "/" << num_keyframes << " keyframes)" << std::endl;
    std::cout << "  ✓ Matching descriptor/keypoint counts: "
              << (has_matching_des_kps ? "OK" : "MISSING") << " ("
              << keyframes_with_matching_des_kps << "/" << num_keyframes << " keyframes)"
              << std::endl;
    std::cout << "  ✓ Keypoints: " << (has_keypoints ? "OK" : "MISSING") << " ("
              << keyframes_with_kps << "/" << num_keyframes << " keyframes)" << std::endl;
    std::cout << "  ✓ Images: " << (has_images ? "OK" : "MISSING") << " (" << keyframes_with_img
              << "/" << num_keyframes << " keyframes)" << std::endl;
    std::cout << "  ✓ Good keyframes: " << (has_good_keyframes ? "OK" : "MISSING") << " ("
              << (num_keyframes - bad_keyframes) << "/" << num_keyframes << " keyframes)"
              << std::endl;

    if (!has_valid_descriptors || !has_matching_des_kps || !has_keypoints || !has_images ||
        !has_good_keyframes) {
        std::cerr << "\nWARNING: Some relocalization prerequisites are missing!" << std::endl;
        std::cerr << "Relocalization may fail with 'compute_frame_matches: #compared pairs: 0'"
                  << std::endl;
        if (!has_valid_descriptors) {
            std::cerr << "  - Missing or invalid descriptors in keyframes" << std::endl;
        }
        if (!has_matching_des_kps) {
            std::cerr << "  - Descriptor/keypoint count mismatch in keyframes" << std::endl;
        }
    } else {
        std::cout << "  ✓ All relocalization prerequisites met!" << std::endl;
        std::cout << "  ✓ Total descriptors available: " << total_descriptors << std::endl;
    }

    // ========================================================================
    // Verify Map Point Associations with Keyframes
    // ========================================================================
    // This is critical for relocalization - keyframes must have map points
    // at keypoint indices so prepare_input_data_for_pnpsolver can work
    std::cout << "\nMap Point Association Check:" << std::endl;

    // Debug: Check if _points_id_data was populated during deserialization
    std::cout << "  Debug: Checking _points_id_data in keyframes..." << std::endl;
    for (const auto &kf : keyframes) {
        if (!kf)
            continue;
        // Note: _points_id_data is protected, so we can't access it directly
        // But we can check if points array is populated
        auto kf_points = kf->get_points();
        std::cout << "    Keyframe " << kf->id << ": points array size = " << kf_points.size()
                  << std::endl;
    }

    int keyframes_with_points = 0;
    int total_keyframe_points = 0;
    int keyframes_with_valid_points = 0;
    int total_valid_keyframe_points = 0;

    for (const auto &kf : keyframes) {
        if (!kf)
            continue;

        auto kf_points = kf->get_points();
        if (!kf_points.empty()) {
            keyframes_with_points++;
            total_keyframe_points += kf_points.size();

            // Count valid (non-null, non-bad) map points
            int valid_points = 0;
            for (const auto &mp : kf_points) {
                if (mp && !mp->is_bad()) {
                    valid_points++;
                }
            }

            if (valid_points > 0) {
                keyframes_with_valid_points++;
                total_valid_keyframe_points += valid_points;
            }
        }
    }

    std::cout << "  Keyframes with points array: " << keyframes_with_points << " / "
              << num_keyframes << std::endl;
    std::cout << "  Total map point associations: " << total_keyframe_points << std::endl;
    std::cout << "  Keyframes with valid map points: " << keyframes_with_valid_points << " / "
              << num_keyframes << std::endl;
    std::cout << "  Total valid map point associations: " << total_valid_keyframe_points
              << std::endl;

    if (keyframes_with_valid_points == 0) {
        std::cerr << "\nWARNING: No keyframes have valid map point associations!" << std::endl;
        std::cerr << "This will cause prepare_input_data_for_pnpsolver to return 0 correspondences."
                  << std::endl;
        std::cerr << "Relocalization will fail even if feature matching finds many matches."
                  << std::endl;
    } else if (keyframes_with_valid_points < num_keyframes) {
        std::cerr << "\nWARNING: Only " << keyframes_with_valid_points << " / " << num_keyframes
                  << " keyframes have valid map point associations!" << std::endl;
    }

    // Test that we can actually retrieve map points from keyframes at keypoint indices
    // This simulates what prepare_input_data_for_pnpsolver does
    std::cout << "\nTesting map point retrieval (simulating prepare_input_data_for_pnpsolver):"
              << std::endl;
    int test_keyframes_with_retrievable_points = 0;
    int total_retrievable_points = 0;

    for (const auto &kf : keyframes) {
        if (!kf || kf->kps.rows() == 0)
            continue;

        auto kf_points = kf->get_points();
        if (kf_points.empty())
            continue;

        // Test retrieving map points at keypoint indices (like prepare_input_data_for_pnpsolver
        // does)
        int retrievable_points = 0;
        int num_kps = kf->kps.rows();
        int num_points = kf_points.size();
        int max_idx = std::min(num_kps, num_points);

        for (int i = 0; i < max_idx; ++i) {
            auto mp = kf->get_point_match(i);
            if (mp && !mp->is_bad()) {
                retrievable_points++;
            }
        }

        if (retrievable_points > 0) {
            test_keyframes_with_retrievable_points++;
            total_retrievable_points += retrievable_points;
        }
    }

    std::cout << "  Keyframes with retrievable map points: "
              << test_keyframes_with_retrievable_points << " / " << num_keyframes << std::endl;
    std::cout << "  Total retrievable map points: " << total_retrievable_points << std::endl;

    if (test_keyframes_with_retrievable_points == 0) {
        std::cerr << "\nERROR: Cannot retrieve map points from keyframes at keypoint indices!"
                  << std::endl;
        std::cerr
            << "This means prepare_input_data_for_pnpsolver will always return 0 correspondences."
            << std::endl;
        std::cerr << "Relocalization will fail even with perfect feature matching." << std::endl;
        throw std::runtime_error(
            "Map point associations not properly restored - relocalization will fail");
    } else {
        std::cout << "  ✓ Map points can be retrieved from keyframes at keypoint indices"
                  << std::endl;
        std::cout << "  ✓ prepare_input_data_for_pnpsolver should work correctly" << std::endl;
    }

    // Verify map points have observations linked to keyframes
    std::cout << "\nMap Point Observations Check:" << std::endl;
    int points_with_observations = 0;
    int total_observations = 0;

    for (const auto &mp : points) {
        if (!mp || mp->is_bad())
            continue;

        int num_obs = mp->num_observations();
        if (num_obs > 0) {
            points_with_observations++;
            total_observations += num_obs;
        }
    }

    std::cout << "  Map points with observations: " << points_with_observations << " / "
              << num_points << std::endl;
    std::cout << "  Total observations: " << total_observations << std::endl;

    if (points_with_observations == 0) {
        std::cerr << "WARNING: No map points have observations linked to keyframes!" << std::endl;
    }

    std::cout << "\nPython-to-C++ map loading tests passed!" << std::endl;
}

} // namespace pyslam

int main() {
    try {
        pyslam::test_python_to_cpp_map_loading();
        std::cout << "\nAll Python-to-C++ serialization tests passed!" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

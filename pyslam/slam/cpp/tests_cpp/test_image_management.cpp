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
/*
 * Test code for ImageColorExtractor class and extract_mean_colors function
 *
 * This test file demonstrates usage of the image color extraction utilities
 * from pyslam/slam/cpp/utils/image_processing.h
 */

#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// Include the header we're testing
#include "utils/image_processing.h"

using namespace pyslam::utils;
using namespace cv;

// Test helper functions
void print_test_header(const std::string &test_name) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "TEST: " << test_name << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void print_test_result(bool passed, const std::string &test_name) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// Test 1: Basic functionality of extract_mean_colors function
void test_extract_mean_colors_function() {
    print_test_header("extract_mean_colors Function Tests");

    // Create a test image (10x10, 3 channels)
    Mat test_img = Mat::zeros(10, 10, CV_8UC3);

    // Fill with a gradient pattern
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            test_img.at<cv::Vec3b>(y, x) = cv::Vec3b(x * 25, y * 25, (x + y) * 12);
        }
    }

    // Test points (center of image)
    pyslam::MatNx2d test_points(3, 2);
    test_points << 5, 5, // center
        3, 3,            // top-left area
        7, 7;            // bottom-right area

    // Test with 3-channel image
    auto colors_3ch = extract_mean_colors<3>(test_img, test_points, 1);

    bool test1_passed = (colors_3ch.size() == 3);
    print_test_result(test1_passed, "extract_mean_colors returns correct number of colors");

    // Test with 1-channel image
    Mat gray_img;
    cvtColor(test_img, gray_img, COLOR_BGR2GRAY);

    auto colors_1ch = extract_mean_colors<1>(gray_img, test_points, 1);

    bool test2_passed = (colors_1ch.size() == 3);
    print_test_result(test2_passed, "extract_mean_colors works with grayscale image");

    // Test edge case - points outside image bounds
    pyslam::MatNx2d edge_points(2, 2);
    edge_points << -1, -1, // outside bounds
        15, 15;            // outside bounds

    auto edge_colors = extract_mean_colors<3>(test_img, edge_points, 1, Vec3f(255, 255, 255));

    bool test3_passed =
        (edge_colors[0] == cv::Vec3f(255, 255, 255) && edge_colors[1] == cv::Vec3f(255, 255, 255));
    print_test_result(test3_passed, "extract_mean_colors handles out-of-bounds points");
}

// Test 2: ImageColorExtractor class basic functionality
void test_image_color_extractor_basic() {
    print_test_header("ImageColorExtractor Class Basic Tests");

    // Create test image
    Mat test_img = Mat::zeros(8, 8, CV_8UC3);

    // Fill with known pattern
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            test_img.at<cv::Vec3b>(y, x) = cv::Vec3b(100, 150, 200);
        }
    }

    // Test constructor
    ImageColorExtractor extractor(test_img, 1, Vec3f(0, 0, 0));

    bool test1_passed = extractor.is_valid();
    print_test_result(test1_passed, "ImageColorExtractor constructor creates valid extractor");

    bool test2_passed =
        (extractor.height() == 8 && extractor.width() == 8 && extractor.channels() == 3);
    print_test_result(test2_passed, "ImageColorExtractor reports correct image dimensions");

    // Test single point extraction
    cv::Vec3f color = extractor.extract_mean_color<true>(4, 4);
    bool test3_passed = (color[0] == 100.0f && color[1] == 150.0f && color[2] == 200.0f);
    print_test_result(test3_passed, "ImageColorExtractor extracts correct color for single point");
}

// Test 3: Different point types with ImageColorExtractor
void test_different_point_types() {
    print_test_header("ImageColorExtractor with Different Point Types");

    // Create test image
    Mat test_img = Mat::zeros(6, 6, CV_8UC3);
    test_img.setTo(cv::Vec3b(50, 100, 150));

    ImageColorExtractor extractor(test_img, 0); // delta=0 for exact pixel

    // Test with cv::Point2f
    Point2f cv_point(3.0f, 3.0f);
    Vec3f color1 = extractor.extract_mean_color<cv::Point2f, true>(cv_point);

    bool test1_passed = (color1[0] == 50.0f && color1[1] == 100.0f && color1[2] == 150.0f);
    print_test_result(test1_passed, "ImageColorExtractor works with cv::Point2f");

    // Test with Eigen::Vector2d
    Eigen::Vector2d eigen_point(2.0, 2.0);
    Vec3f color2 = extractor.extract_mean_color<Eigen::Vector2d, true>(eigen_point);

    bool test2_passed = (color2[0] == 50.0f && color2[1] == 100.0f && color2[2] == 150.0f);
    print_test_result(test2_passed, "ImageColorExtractor works with Eigen::Vector2d");

    // Test with std::array
    std::array<double, 2> array_point = {1.0, 1.0};
    Vec3f color3 = extractor.extract_mean_color<std::array<double, 2>, true>(array_point);

    bool test3_passed = (color3[0] == 50.0f && color3[1] == 100.0f && color3[2] == 150.0f);
    print_test_result(test3_passed, "ImageColorExtractor works with std::array<double, 2>");
}

// Test 4: Multiple points extraction with ImageColorExtractor
void test_multiple_points_extraction() {
    print_test_header("ImageColorExtractor Multiple Points Extraction");

    // Create test image with different colored regions
    Mat test_img = Mat::zeros(10, 10, CV_8UC3);

    // Region 1: Red
    Rect region1(0, 0, 5, 5);
    test_img(region1).setTo(Vec3b(255, 0, 0));

    // Region 2: Green
    Rect region2(5, 0, 5, 5);
    test_img(region2).setTo(Vec3b(0, 255, 0));

    // Region 3: Blue
    Rect region3(0, 5, 5, 5);
    test_img(region3).setTo(Vec3b(0, 0, 255));

    // Region 4: White
    Rect region4(5, 5, 5, 5);
    test_img(region4).setTo(Vec3b(255, 255, 255));

    ImageColorExtractor extractor(test_img, 1);

    // Test points in different regions
    pyslam::MatNx2d test_points(4, 2);
    test_points << 2, 2, // Red region
        7, 2,            // Green region
        2, 7,            // Blue region
        7, 7;            // White region

    auto colors = extractor.extract_mean_colors<pyslam::MatNx2d, true>(test_points);

    bool test1_passed = (colors.size() == 4);
    print_test_result(test1_passed, "Multiple points extraction returns correct number of colors");

    // Check if colors are approximately correct (allowing for averaging)
    bool test2_passed =
        (colors[0][2] < colors[0][0] && colors[0][2] < colors[0][1]); // Red dominant
    bool test3_passed =
        (colors[1][0] < colors[1][1] && colors[1][2] < colors[1][1]); // Green dominant
    bool test4_passed =
        (colors[2][0] < colors[2][2] && colors[2][1] < colors[2][2]); // Blue dominant
    bool test5_passed =
        (colors[3][0] > 200 && colors[3][1] > 200 && colors[3][2] > 200); // White-ish

    print_test_result(test2_passed, "Red region color extraction");
    print_test_result(test3_passed, "Green region color extraction");
    print_test_result(test4_passed, "Blue region color extraction");
    print_test_result(test5_passed, "White region color extraction");
}

// Test 5: Edge cases and error handling
void test_edge_cases() {
    print_test_header("Edge Cases and Error Handling");

    // Test with invalid image type
    Mat invalid_img = Mat::zeros(5, 5, CV_32FC3);
    ImageColorExtractor extractor_invalid(invalid_img, 1);

    bool test1_passed = !extractor_invalid.is_valid();
    print_test_result(test1_passed, "ImageColorExtractor handles invalid image type");

    // Test with grayscale image
    Mat gray_img = Mat::zeros(6, 6, CV_8UC1);
    gray_img.setTo(128);

    ImageColorExtractor extractor_gray(gray_img, 1);

    bool test2_passed = extractor_gray.is_valid();
    print_test_result(test2_passed, "ImageColorExtractor works with grayscale image");

    // Test grayscale color extraction
    Vec3f gray_color = extractor_gray.extract_mean_color<false>(3, 3);
    bool test3_passed = (gray_color[0] == 128.0f && gray_color[1] == 0.0f && gray_color[2] == 0.0f);
    print_test_result(test3_passed, "Grayscale color extraction works correctly");

    // Test out-of-bounds points
    Vec3f out_of_bounds = extractor_gray.extract_mean_color<false>(-1, -1);
    bool test4_passed = (out_of_bounds == Vec3f(0, 0, 0)); // Should return default color
    print_test_result(test4_passed, "Out-of-bounds points return default color");
}

// Test 6: Performance test with larger image
void test_performance() {
    print_test_header("Performance Test");

    // Create larger test image
    Mat large_img = Mat::zeros(100, 100, CV_8UC3);

    // Fill with random colors
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            large_img.at<Vec3b>(y, x) = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        }
    }

    ImageColorExtractor extractor(large_img, 2);

    // Create many test points
    pyslam::MatNx2d many_points(1000, 2);
    for (int i = 0; i < 1000; ++i) {
        many_points(i, 0) = rand() % 100;
        many_points(i, 1) = rand() % 100;
    }

    // Time the extraction
    auto start = std::chrono::high_resolution_clock::now();
    auto colors = extractor.extract_mean_colors<pyslam::MatNx2d, true>(many_points);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    bool test1_passed = (colors.size() == 1000);
    print_test_result(test1_passed, "Performance test: extracted colors for 1000 points");

    std::cout << "Performance: " << duration.count() << " ms for 1000 points" << std::endl;
}

// Test 7: get_xy function with different point types
void test_get_xy_function() {
    print_test_header("get_xy Function Tests");

    // Test cv::Point2f
    Point2f cv_point(10.5f, 20.3f);
    auto [x1, y1] = get_xy<Point2f, float>(cv_point);
    bool test1_passed = (x1 == 10.5f && y1 == 20.3f);
    print_test_result(test1_passed, "get_xy works with cv::Point2f");

    // Test Eigen::Vector2d
    Eigen::Vector2d eigen_point(15.7, 25.9);
    auto [x2, y2] = get_xy<Eigen::Vector2d, double>(eigen_point);
    bool test2_passed = (x2 == 15.7 && y2 == 25.9);
    print_test_result(test2_passed, "get_xy works with Eigen::Vector2d");

    // Test std::array
    std::array<float, 2> array_point = {30.1f, 40.2f};
    auto [x3, y3] = get_xy<std::array<float, 2>, float>(array_point);
    bool test3_passed = (x3 == 30.1f && y3 == 40.2f);
    print_test_result(test3_passed, "get_xy works with std::array<float, 2>");
}

int main() {
    std::cout << "Starting ImageColorExtractor and extract_mean_colors Tests" << std::endl;
    std::cout << "========================================================" << std::endl;

    try {
        test_extract_mean_colors_function();
        test_image_color_extractor_basic();
        test_different_point_types();
        test_multiple_points_extraction();
        test_edge_cases();
        test_performance();
        test_get_xy_function();

        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ALL TESTS COMPLETED" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
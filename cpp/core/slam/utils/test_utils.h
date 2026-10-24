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

#pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "camera.h"
#include "camera_pose.h"
#include "dictionary.h"
#include "feature_shared_resources.h"
#include "frame.h"
#include "keyframe.h"

namespace pyslam {
namespace test_utils {

// ===============================
// Matrix Comparison Utilities
// ===============================

/**
 * Compare two cv::Mat objects for exact equality (bit-for-bit)
 */
inline bool mats_equal_exact(const cv::Mat &a, const cv::Mat &b) {
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    if (a.empty())
        return b.empty();
    const cv::Mat ac = a.isContinuous() ? a : a.clone();
    const cv::Mat bc = b.isContinuous() ? b : b.clone();
    const size_t bytes = ac.total() * ac.elemSize();
    return bytes == bc.total() * bc.elemSize() && std::memcmp(ac.data, bc.data, bytes) == 0;
}

/**
 * Compare two cv::Mat objects with tolerance for floating-point matrices
 */
inline bool mats_equal_tolerant(const cv::Mat &a, const cv::Mat &b, double tolerance = 1e-6) {
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    if (a.empty())
        return b.empty();

    // For floating-point matrices, use element-wise comparison with tolerance
    if (a.type() == CV_32F) {
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < a.cols; ++j) {
                if (std::abs(a.at<float>(i, j) - b.at<float>(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    } else if (a.type() == CV_64F) {
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < a.cols; ++j) {
                if (std::abs(a.at<double>(i, j) - b.at<double>(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    } else {
        // For non-floating-point types, use exact comparison
        return mats_equal_exact(a, b);
    }
}

/**
 * Compare two Eigen matrices with tolerance
 */
inline bool eigen_matrices_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
                                 double tolerance = 1e-12) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        return false;
    return (a - b).lpNorm<Eigen::Infinity>() < tolerance;
}

/**
 * Compare two Eigen vectors with tolerance
 */
inline bool eigen_vectors_equal(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                                double tolerance = 1e-12) {
    return (a - b).lpNorm<Eigen::Infinity>() < tolerance;
}

// ===============================
// Test Data Generation Utilities
// ===============================

/**
 * Fill a cv::Mat with deterministic test data
 */
inline void fill_test_mat(cv::Mat &m) {
    // Fill with deterministic pattern
    const int channels = m.channels();
    for (int r = 0; r < m.rows; ++r) {
        switch (m.depth()) {
        case CV_8U: {
            uint8_t *row = m.ptr<uint8_t>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    int val = ((r * 131 + c * 17 + ch * 7) % 256);
                    row[c * channels + ch] = static_cast<uint8_t>(val);
                }
            }
            break;
        }
        case CV_32F: {
            float *row = m.ptr<float>(r);
            for (int c = 0; c < m.cols; ++c) {
                for (int ch = 0; ch < channels; ++ch) {
                    float val = ((r * 131 + c * 17 + ch * 7) % 1000) / 1000.0f;
                    row[c * channels + ch] = val;
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported matrix type");
        }
    }
}

// ===============================
// Test Setup Utilities
// ===============================

/**
 * Initialize FeatureSharedResources with default values for testing
 */
inline void init_feature_shared_info() {
    FeatureSharedResources::num_levels = 8;
    FeatureSharedResources::scale_factors.resize(FeatureSharedResources::num_levels);
    FeatureSharedResources::scale_factor = 1.2f;
    for (int i = 0; i < FeatureSharedResources::num_levels; ++i) {
        FeatureSharedResources::scale_factors[i] =
            std::pow(FeatureSharedResources::scale_factor, i);
    }
}

/**
 * Create a test camera with standard parameters
 */
inline CameraPtr create_test_camera() {
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
    return std::make_shared<PinholeCamera>(camera_config);
}

/**
 * Create test images with deterministic data
 */
inline void create_test_images(cv::Mat &img, cv::Mat &img_right, cv::Mat &depth_img,
                               cv::Mat &semantic_img) {
    img = cv::Mat::zeros(480, 640, CV_8UC3);
    img_right = cv::Mat::zeros(480, 640, CV_8UC3);
    depth_img = cv::Mat::zeros(480, 640, CV_32F);
    semantic_img = cv::Mat::zeros(480, 640, CV_8UC1);

    // fill_test_mat(img);
    // fill_test_mat(img_right);
    // fill_test_mat(depth_img);
    // fill_test_mat(semantic_img);
}

/**
 * Create a test pose matrix
 */
inline CameraPose create_test_pose() {
    Eigen::Matrix4d pose_matrix = Eigen::Matrix4d::Identity();
    pose_matrix(0, 3) = 1.0; // translation x
    pose_matrix(1, 3) = 2.0; // translation y
    pose_matrix(2, 3) = 3.0; // translation z
    return CameraPose(pose_matrix);
}

/**
 * Initialize KeyFrame with proper keypoint data for MapPoint creation
 */
inline void init_keyframe_for_mappoints(KeyFramePtr keyframe, int num_kps = 10) {
    std::cout << "Initializing KeyFrame for MapPoints..." << std::endl;

    // Initialize basic keypoint data
    std::cout << "Resizing kps..." << std::endl;
    keyframe->kps.resize(num_kps, 2);
    keyframe->kps_r.resize(num_kps, 2);
    keyframe->kpsu.resize(num_kps, 2);
    keyframe->kpsn.resize(num_kps, 2);

    std::cout << "Resizing octaves and sizes..." << std::endl;
    keyframe->octaves.resize(num_kps);
    keyframe->sizes.resize(num_kps);
    keyframe->points.resize(num_kps, nullptr); // Initialize with null pointers

    std::cout << "Creating descriptor matrix..." << std::endl;
    keyframe->des = cv::Mat::zeros(num_kps, 32, CV_8U); // Create proper descriptor matrix

    std::cout << "Setting keyframe flag..." << std::endl;
    // Set keyframe flag
    keyframe->is_keyframe = true;

    for (int i = 0; i < num_kps; ++i) {
        keyframe->octaves[i] = i % 4;
        keyframe->sizes[i] = 10.0f + i * 2.0f;
        keyframe->kps.row(i) = Eigen::RowVector2f(i * 10.0f, i * 10.0f);
        keyframe->kps_r.row(i) = Eigen::RowVector2f(i * 10.0f, i * 10.0f);
        keyframe->kpsu.row(i) = Eigen::RowVector2f(i * 10.0f, i * 10.0f);
        keyframe->kpsn.row(i) = Eigen::RowVector2f(i * 10.0f, i * 10.0f);

        // Initialize descriptor row
        for (int j = 0; j < 32; ++j) {
            keyframe->des.at<unsigned char>(i, j) =
                static_cast<unsigned char>((i * 7 + j * 13) % 256);
        }
    }
}

// ===============================
// Test Assertion Utilities
// ===============================

/**
 * Assert that two matrices are equal (exact or tolerant based on type)
 */
inline void assert_mats_equal(const cv::Mat &a, const cv::Mat &b,
                              const std::string &name = "matrices", double tolerance = 1e-6) {
    bool equal = false;
    if (a.type() == CV_32F || a.type() == CV_64F) {
        equal = mats_equal_tolerant(a, b, tolerance);
    } else {
        equal = mats_equal_exact(a, b);
    }

    if (!equal) {
        throw std::runtime_error(name + " mismatch");
    }
}

/**
 * Assert that two Eigen matrices are equal
 */
inline void assert_eigen_matrices_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
                                        const std::string &name = "matrices",
                                        double tolerance = 1e-12) {
    if (!eigen_matrices_equal(a, b, tolerance)) {
        throw std::runtime_error(name + " mismatch");
    }
}

/**
 * Assert that two Eigen vectors are equal
 */
inline void assert_eigen_vectors_equal(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                                       const std::string &name = "vectors",
                                       double tolerance = 1e-12) {
    if (!eigen_vectors_equal(a, b, tolerance)) {
        throw std::runtime_error(name + " mismatch");
    }
}

/**
 * Simple assertion utility for boolean conditions
 */
inline void expect_true(bool cond, const char *msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

/**
 * Simple assertion utility for boolean conditions with string message
 */
inline void expect_true(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

// ===============================

// Helper function to check if two Mat4d vectors are equal
bool vectors_equal(const std::vector<Eigen::Matrix4d> &a, const std::vector<Eigen::Matrix4d> &b,
                   double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] - b[i]).lpNorm<Eigen::Infinity>() > tolerance)
            return false;
    }
    return true;
}

// Helper function to check if two Vec3d vectors are equal
bool vectors_equal(const std::vector<Eigen::Vector3d> &a, const std::vector<Eigen::Vector3d> &b,
                   double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] - b[i]).lpNorm<Eigen::Infinity>() > tolerance)
            return false;
    }
    return true;
}

// Helper function to check if two Vec3f vectors are equal
bool vectors_equal(const std::vector<Eigen::Vector3f> &a, const std::vector<Eigen::Vector3f> &b,
                   float tolerance = 1e-6f) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] - b[i]).lpNorm<Eigen::Infinity>() > tolerance)
            return false;
    }
    return true;
}

// Helper function to check if two Vec6d vectors are equal
bool vectors_equal(const std::vector<Eigen::Matrix<double, 6, 1>> &a,
                   const std::vector<Eigen::Matrix<double, 6, 1>> &b, double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] - b[i]).lpNorm<Eigen::Infinity>() > tolerance)
            return false;
    }
    return true;
}

// Helper function to check if two double vectors are equal
bool vectors_equal(const std::vector<double> &a, const std::vector<double> &b,
                   double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance)
            return false;
    }
    return true;
}

} // namespace test_utils
} // namespace pyslam

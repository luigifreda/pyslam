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

#include "dictionary.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pyslam {

// Camera types enum
enum class CameraType { NONE = 0, PINHOLE = 1 };

// Sensor types enum
enum class SensorType { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

// Utility functions
double fov2focal(double fov, int pixels);
double focal2fov(double focal, int pixels);

// CameraUtils class
class CameraUtils {
  public:
    // Backproject 2d image points (pixels) into 3D points by using depth and
    // intrinsic K Input: uv: array [N,2], depth: array [N], K: array [3,3]
    // Output: xyz: array [N,3]
    static std::vector<Eigen::Vector3d> backproject_3d(const std::vector<cv::Point2f> &uv,
                                                       const std::vector<double> &depth,
                                                       const Eigen::Matrix3d &K);

    static std::vector<Eigen::Vector3d> backproject_3d_numba(const std::vector<cv::Point2f> &uv,
                                                             const std::vector<double> &depth,
                                                             const Eigen::Matrix3d &Kinv);

    // project a 3D point or an array of 3D points (w.r.t. camera frame), of shape
    // [Nx3] out: Nx2 image points, [Nx1] array of map point depths
    static std::pair<std::vector<cv::Point2f>, std::vector<double>>
    project(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K);

    static std::pair<std::vector<cv::Point2f>, std::vector<double>>
    project_numba(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K);

    // stereo-project a 3D point or an array of 3D points (w.r.t. camera frame),
    // of shape [Nx3] (assuming rectified stereo images) out: Nx3 image points,
    // [Nx1] array of map point depths
    static std::pair<std::vector<cv::Point3f>, std::vector<double>>
    project_stereo(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K, double bf);

    static std::pair<std::vector<cv::Point3f>, std::vector<double>>
    project_stereo_numba(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K,
                         double bf);

    // in: uvs [Nx2]
    // out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points
    // on z=1 plane)
    static std::vector<Eigen::Vector2d> unproject_points(const std::vector<cv::Point2f> &uvs,
                                                         const Eigen::Matrix3d &Kinv);

    static std::vector<Eigen::Vector2d> unproject_points_numba(const std::vector<cv::Point2f> &uvs,
                                                               const Eigen::Matrix3d &Kinv);

    // in: uvs [Nx2], depths [Nx1]
    // out: xcs array [Nx3] of backprojected 3D points
    static std::vector<Eigen::Vector3d> unproject_points_3d(const std::vector<cv::Point2f> &uvs,
                                                            const std::vector<double> &depths,
                                                            const Eigen::Matrix3d &Kinv);

    static std::vector<Eigen::Vector3d>
    unproject_points_3d_numba(const std::vector<cv::Point2f> &uvs,
                              const std::vector<double> &depths, const Eigen::Matrix3d &Kinv);

    // input: [Nx2] array of uvs, [Nx1] of zs
    // output: [Nx1] array of visibility flags
    static std::vector<bool> are_in_image_numba(const std::vector<cv::Point2f> &uvs,
                                                const std::vector<double> &zs, double u_min,
                                                double u_max, double v_min, double v_max);
};

// Base object class for camera info management - matches Python CameraBase
// exactly
class CameraBase {
  public:
    // Camera parameters
    CameraType type;
    int width, height;
    double fx, fy, cx, cy;
    std::vector<double> D; // distortion coefficients [k1, k2, p1, p2, k3]
    bool is_distorted;
    int fps;
    double bf, b;                      // stereo parameters
    double u_min, u_max, v_min, v_max; // image bounds
    bool initialized;

    // Constructor - matches Python __init__ exactly
    CameraBase();

    // Destructor
    virtual ~CameraBase() = default;

    // Copy constructor and assignment
    CameraBase(const CameraBase &other);
    CameraBase &operator=(const CameraBase &other);
};

// Camera class - inherits from CameraBase, matches Python Camera exactly
class Camera : public CameraBase {
  public:
    // Additional parameters
    double fovx, fovy;      // field of view in x and y directions
    SensorType sensor_type; // sensor type (monocular, stereo, RGBD)
    double depth_factor;    // depth map values factor
    double depth_threshold; // close/far threshold

    // Constructor - matches Python __init__ exactly
    Camera(const ConfigDict &config);

    // Destructor
    virtual ~Camera() = default;

    // Copy constructor and assignment
    Camera(const Camera &other);
    Camera &operator=(const Camera &other);

    virtual std::pair<std::vector<cv::Point2f>, std::vector<double>>
    project(const std::vector<Eigen::Vector3d> &xcs) const {
        return std::make_pair(std::vector<cv::Point2f>(), std::vector<double>());
    }
    virtual std::pair<std::vector<cv::Point3f>, std::vector<double>>
    project_stereo(const std::vector<Eigen::Vector3d> &xcs) const {
        return std::make_pair(std::vector<cv::Point3f>(), std::vector<double>());
    }

    // Methods
    bool is_stereo() const;
    std::string to_json() const;
    void init_from_json(const std::string &json_str);
    bool is_in_image(const cv::Point2f &uv, double z) const;
    std::vector<bool> are_in_image(const std::vector<cv::Point2f> &uvs,
                                   const std::vector<double> &zs) const;
    Eigen::Matrix4d get_render_projection_matrix(double znear = 0.01, double zfar = 100.0) const;
    void set_fovx(double fovx);
    void set_fovy(double fovy);
};

// PinholeCamera class - inherits from Camera, matches Python PinholeCamera
// exactly
class PinholeCamera : public Camera {
  public:
    // Intrinsic matrices
    Eigen::Matrix3d K;    // intrinsic matrix
    Eigen::Matrix3d Kinv; // inverse intrinsic matrix

    // Constructor
    PinholeCamera(const ConfigDict &config);

    // Destructor
    ~PinholeCamera() = default;

    // Copy constructor and assignment
    PinholeCamera(const PinholeCamera &other);
    PinholeCamera &operator=(const PinholeCamera &other);

    // Methods
    void init();

    std::pair<std::vector<cv::Point2f>, std::vector<double>>
    project(const std::vector<Eigen::Vector3d> &xcs) const;

    std::pair<std::vector<cv::Point3f>, std::vector<double>>
    project_stereo(const std::vector<Eigen::Vector3d> &xcs) const;

    Eigen::Vector2d unproject(const cv::Point2f &uv) const;
    Eigen::Vector3d unproject_3d(double u, double v, double depth) const;
    std::vector<Eigen::Vector2d> unproject_points(const std::vector<cv::Point2f> &uvs) const;
    std::vector<Eigen::Vector3d> unproject_points_3d(const std::vector<cv::Point2f> &uvs,
                                                     const std::vector<double> &depths) const;
    std::vector<cv::Point2f> undistort_points(const std::vector<cv::Point2f> &uvs) const;
    void undistort_image_bounds();
    std::string to_json() const;
    static PinholeCamera from_json(const std::string &json_str);

  private:
    // Helper methods
    void compute_intrinsic_matrices();
    void compute_fov();
    void update_distortion_flag();
};

} // namespace pyslam

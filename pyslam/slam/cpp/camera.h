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

#include "eigen_aliases.h"
#include "utils/dictionary.h"

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

// generic extractor for 2D point types
template <typename Elem> inline std::pair<double, double> get_xy(const Elem &p) {
    if constexpr (std::is_same_v<Elem, cv::Point2f> || std::is_same_v<Elem, cv::Point2d>) {
        return {static_cast<double>(p.x), static_cast<double>(p.y)};
    } else if constexpr (std::is_same_v<Elem, Eigen::Vector2d>) {
        return {p.x(), p.y()};
    } else if constexpr (std::is_same_v<Elem, Eigen::Vector2f>) {
        return {static_cast<double>(p.x()), static_cast<double>(p.y())};
    } else if constexpr (std::is_same_v<Elem, Eigen::Matrix<double, 2, 1>>) {
        return {p(0), p(1)};
    } else if constexpr (std::is_same_v<Elem, Eigen::Matrix<float, 2, 1>>) {
        return {static_cast<double>(p(0)), static_cast<double>(p(1))};
    } else if constexpr (std::is_same_v<Elem, std::array<double, 2>> ||
                         std::is_same_v<Elem, std::array<float, 2>>) {
        return {static_cast<double>(p[0]), static_cast<double>(p[1])};
    } else {
        static_assert(sizeof(Elem) == 0, "get_xy: unsupported 2D point element type");
    }
}

// CameraUtils class
class CameraUtils {

  public:
    // project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    // out:
    // [Nx2] image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    static std::pair<MatNx2d, VecNd> project_points(MatNx3dRef xcs, Mat3dRef K);

    // ------------------------------------------------------------------------
    // stereo-project an array of 3D points (w.r.t. camera frame),
    // of shape [Nx3] (assuming rectified stereo images)
    // out:
    // [Nx3] image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64, (3,3) float64, float64)
    static std::pair<MatNx3d, VecNd> project_points_stereo(MatNx3dRef xcs, Mat3dRef K,
                                                           const double bf);

    // ------------------------------------------------------------------------
    // in: uvs [Nx2]
    // out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points
    // on z=1 plane)
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    static MatNx2d unproject_points(MatNx2dRef uvs, Mat3dRef Kinv);

    // ------------------------------------------------------------------------
    // Backproject 2d image points (pixels) into 3D points by using depth and intrinsics Kinv
    // in: uvs [Nx2], depths [Nx1], Kinv: array [3,3]
    // out: xcs array [Nx3] of backprojected 3D points
    template <typename VecPoints2>
    static std::vector<Eigen::Vector3d> unproject_points_3d(const VecPoints2 &uvs,
                                                            const std::vector<double> &depths,
                                                            const Eigen::Matrix3d &Kinv) {
        using Elem = typename VecPoints2::value_type;

        const size_t N = std::min(uvs.size(), depths.size());
        std::vector<Eigen::Vector3d> result;
        result.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            auto [u, v] = get_xy(uvs[i]);
            const Eigen::Vector3d uvh(u, v, 1.0);
            result.push_back(depths[i] * (Kinv * uvh));
        }
        return result;
    }

    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    static MatNx3d unproject_points_3d(MatNx2dRef uvs, Mat3dRef Kinv, VecNdRef depths);

    // ------------------------------------------------------------------------

    // input: [Nx2] array of uvs, [Nx1] of zs
    // output: [Nx1] array of visibility flags
    static std::vector<bool> are_in_image(MatNx2dRef uvs, VecNdRef zs, double u_min, double u_max,
                                          double v_min, double v_max) {

        std::vector<bool> result;
        result.reserve(uvs.rows());

        for (int i = 0; i < uvs.rows(); ++i) {
            bool in_image = (uvs(i, 0) >= u_min) && (uvs(i, 0) < u_max) && (uvs(i, 1) >= v_min) &&
                            (uvs(i, 1) < v_max) && (zs(i) > 0);
            result.push_back(in_image);
        }

        return result;
    }
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

    // Constructor
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

    // Intrinsic matrices
    Eigen::Matrix3d K;    // intrinsic matrix
    Eigen::Matrix3d Kinv; // inverse intrinsic matrix

    // Constructor
    Camera(const ConfigDict &config);

    // Destructor
    virtual ~Camera() = default;

    // Copy constructor and assignment
    Camera(const Camera &other);
    Camera &operator=(const Camera &other);

    // ------------------------------------------------------------------------
    // - unproject a 2D image point into a 3D point on the z=1 plane
    //   out: 3D point
    Eigen::Vector2d unproject_point(const double u, const double v) const;
    template <typename T> Eigen::Vector2d unproject_point(const Eigen::Matrix<T, 2, 1> &uv) const {
        const double x = (static_cast<double>(uv(0)) - cx) / fx;
        const double y = (static_cast<double>(uv(1)) - cy) / fy;
        return Eigen::Vector2d(x, y);
    }

    // - unproject a 2D image point into a 3D point
    //   out: 3D point
    Eigen::Vector3d unproject_point_3d(const double u, const double v, const double depth) const;
    template <typename T>
    Eigen::Vector3d unproject_point_3d(const Eigen::Matrix<T, 2, 1> &uv, const double depth) const {
        const double x = depth * (static_cast<double>(uv(0)) - cx) / fx;
        const double y = depth * (static_cast<double>(uv(1)) - cy) / fy;
        return Eigen::Vector3d(x, y, depth);
    }

    // ------------------------------------------------------------------------
    // - project a 3D point (w.r.t. camera frame) into a 2D image point
    virtual std::pair<Eigen::Vector2d, double> project_point(const Eigen::Vector3d &xcs) const {
        return std::make_pair(
            Eigen::Vector2d(fx * xcs.x() / xcs.z() + cx, fy * xcs.y() / xcs.z() + cy), xcs.z());
    }

    // - project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   out: Nx2 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    virtual std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const {
        return CameraUtils::project_points(xcs, K);
    }

    // ------------------------------------------------------------------------
    // - stereo-project a 3D point (w.r.t. camera frame) into a 3D image point (stereo)
    //   out: 3D image point
    virtual Eigen::Vector3d project_point_stereo(const Eigen::Vector3d &xcs) const {
        const double u = fx * xcs.x() / xcs.z() + cx;
        const double v = fy * xcs.y() / xcs.z() + cy;
        const double ur = u - bf / xcs.z();
        return Eigen::Vector3d(u, v, ur);
    }

    // - stereo-project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   (assuming rectified stereo images)
    //   out: Nx3 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    virtual std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const {
        return CameraUtils::project_points_stereo(xcs, K, bf);
    }

    // ------------------------------------------------------------------------
    void compute_intrinsic_matrices();

    // ------------------------------------------------------------------------
    // Methods
    bool is_stereo() const;
    std::string to_json() const;
    void init_from_json(const std::string &json_str);
    bool is_in_image(const Eigen::Vector2d &uv, double z) const;
    std::vector<bool> are_in_image(MatNx2dRef uvs, VecNdRef zs) const {
        return CameraUtils::are_in_image(uvs, zs, u_min, u_max, v_min, v_max);
    }
    Eigen::Matrix4d get_render_projection_matrix(double znear = 0.01, double zfar = 100.0) const;
    void set_fovx(double fovx);
    void set_fovy(double fovy);

    // ------------------------------------------------------------------------
    // - undistort a 2D image point or an array of 2D image points
    //   out: 2D image points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx2d undistort_points(MatNx2dRef uvs) const { return MatNx2d(); }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1 plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx2d unproject_points(MatNx2dRef uvs) const { return MatNx2d(); }
};

// PinholeCamera class - inherits from Camera, matches Python PinholeCamera
// exactly
class PinholeCamera : public Camera {
  public:
    // Constructor
    PinholeCamera(const ConfigDict &config);

    // Destructor
    ~PinholeCamera() = default;

    // Copy constructor and assignment
    PinholeCamera(const PinholeCamera &other);
    PinholeCamera &operator=(const PinholeCamera &other);

    // Methods
    void init();

    // ------------------------------------------------------------------------
    // - project a 3D point (w.r.t. camera frame) into a 2D image point
    std::pair<Eigen::Vector2d, double> project_point(const Eigen::Vector3d &xcs) const {
        return std::make_pair(
            Eigen::Vector2d(fx * xcs.x() / xcs.z() + cx, fy * xcs.y() / xcs.z() + cy), xcs.z());
    }

    // - project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   out: Nx2 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const {
        return CameraUtils::project_points(xcs, K);
    }

    // ------------------------------------------------------------------------
    // - stereo-project a 3D point (w.r.t. camera frame) into a 2D image point
    virtual Eigen::Vector3d project_point_stereo(const Eigen::Vector3d &xcs) const {
        const double u = fx * xcs.x() / xcs.z() + cx;
        const double v = fy * xcs.y() / xcs.z() + cy;
        const double ur = u - bf / xcs.z();
        return Eigen::Vector3d(u, v, ur);
    }

    // - stereo-project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   (assuming rectified stereo images)
    //   out: Nx3 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const {
        return CameraUtils::project_points_stereo(xcs, K, bf);
    }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1 plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx2d unproject_points(MatNx2dRef uvs) const {
        return CameraUtils::unproject_points(uvs, Kinv);
    }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1 plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const {
        return CameraUtils::unproject_points_3d(uvs, Kinv, depths);
    }

    // ------------------------------------------------------------------------
    // - undistort a 2D image point or an array of 2D image points
    //   out: 2D image points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx2d undistort_points(MatNx2dRef uvs) const;

    // ------------------------------------------------------------------------
    // - undistort the image bounds
    void undistort_image_bounds();

    // ------------------------------------------------------------------------
    // - convert to JSON
    std::string to_json() const;
    static PinholeCamera from_json(const std::string &json_str);

  private:
    // Helper methods
    void compute_fov();
    void update_distortion_flag();
};

// Helper function to create camera from JSON with proper type detection
Camera *create_camera_from_json(const nlohmann::json &camera_json);

} // namespace pyslam

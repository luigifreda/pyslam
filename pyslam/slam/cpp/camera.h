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
#include "eigen_aliases.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "smart_pointers.h"
#ifndef NDEBUG
#include "utils/messages.h"
#endif

#ifdef USE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

namespace pyslam {

// Camera types enum
enum class CameraType { NONE = 0, PINHOLE = 1 }; // keep it consistent with Python CameraType

// Sensor types enum
enum class SensorType {
    MONOCULAR = 0,
    STEREO = 1,
    RGBD = 2
}; // keep it consistent with Python SensorType

// Utility function to get sensor type
SensorType get_sensor_type(const std::string &sensor_type);

// Utility functions
double fov2focal(double fov, int pixels);
double focal2fov(double focal, int pixels);

constexpr double kMinZ = 1e-10;

// CameraUtils class
class CameraUtils {

  public:
    // project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    // out:
    // [Nx2] image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    template <typename Scalar>
    static std::pair<MatNx2<Scalar>, VecN<Scalar>> project_points(MatNx3Ref<Scalar> xcs,
                                                                  Mat3Ref<Scalar> K);
    // ------------------------------------------------------------------------
    // stereo-project an array of 3D points (w.r.t. camera frame),
    // of shape [Nx3] (assuming rectified stereo images)
    // out:
    // [Nx3] image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64, (3,3) float64, float64)
    template <typename Scalar>
    static std::pair<MatNx3<Scalar>, VecN<Scalar>>
    project_points_stereo(MatNx3Ref<Scalar> xcs, Mat3Ref<Scalar> K, const Scalar bf);

    // ------------------------------------------------------------------------
    // in: uvs [Nx2]
    // out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points
    // on z=1 plane)
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    template <typename Scalar>
    static MatNx2<Scalar> unproject_points(MatNx2Ref<Scalar> uvs, Mat3Ref<Scalar> Kinv);

    // ------------------------------------------------------------------------
    // Backproject 2d image points (pixels) into 3D points by using depth and intrinsics Kinv
    // in: uvs [Nx2], depths [Nx1], Kinv: array [3,3]
    // out: xcs array [Nx3] of backprojected 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    template <typename Scalar>
    static MatNx3<Scalar> unproject_points_3d(MatNx2Ref<Scalar> uvs, VecNRef<Scalar> depths,
                                              Mat3Ref<Scalar> Kinv) {
        const int N = static_cast<int>(uvs.rows());
        MatNx3<Scalar> result(N, 3);

        for (int i = 0; i < N; ++i) {
            const Vec3<Scalar> uv_homogeneous(uvs(i, 0), uvs(i, 1), 1.0);
            const Vec3<Scalar> uv_scaled = uv_homogeneous * depths(i); // Scale by depth first
            const Vec3<Scalar> p = Kinv * uv_scaled; // Then apply inverse intrinsics
            result(i, 0) = p.x();
            result(i, 1) = p.y();
            result(i, 2) = p.z(); // Use the computed z coordinate
        }
        return std::move(result);
    }

    // ------------------------------------------------------------------------

    // input: [Nx2] array of uvs, [Nx1] of zs
    // output: [Nx1] array of visibility flags
    template <typename Scalar>
    static std::vector<bool> are_in_image(MatNxMRef<Scalar> uvs, VecNRef<Scalar> zs,
                                          const Scalar u_min, const Scalar u_max,
                                          const Scalar v_min, const Scalar v_max) {

        std::vector<bool> result;
        result.reserve(static_cast<size_t>(uvs.rows()));

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
    CameraType type = CameraType::NONE;
    int width = -1, height = -1;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    std::vector<double> D; // distortion coefficients [k1, k2, p1, p2, k3]
    bool is_distorted = false;
    int fps = -1;
    double bf = -1.0;
    double b = -1.0;                                               // stereo parameters
    double u_min = -1.0, u_max = -1.0, v_min = -1.0, v_max = -1.0; // image bounds
    bool initialized = false;

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
    using SharedPtr = std::shared_ptr<Camera>;

  public:
    // Additional parameters
    double fovx = 0.0, fovy = 0.0;                  // field of view in x and y directions
    SensorType sensor_type = SensorType::MONOCULAR; // sensor type (monocular, stereo, RGBD)
    double depth_factor = 1.0;                      // depth map values factor
    double depth_threshold = std::numeric_limits<double>::infinity(); // close/far threshold

    // Intrinsic matrices
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();    // intrinsic matrix
    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Zero(); // inverse intrinsic matrix

    cv::Mat K_cv;
    cv::Mat D_cv;
    int cv_depth = std::numeric_limits<int>::max();

    // Constructor
    Camera();
    Camera(const ConfigDict &config);

    // Destructor
    virtual ~Camera() = default;

    // Copy constructor, assignment, move constructor, move assignment
    Camera(const Camera &other);
    Camera &operator=(const Camera &other);
    Camera(Camera &&other) noexcept;
    Camera &operator=(Camera &&other) noexcept;

    // ------------------------------------------------------------------------
    // - unproject a 2D image point into a 3D point on the z=1 plane
    //   out: 3D point
    template <typename T> Vec2<T> unproject_point(const T u, const T v) const {
        const T x = (u - cx) / fx;
        const T y = (v - cy) / fy;
        return Vec2<T>(x, y);
    }
    template <typename T> Vec2<T> unproject_point(const Vec2Ref<T> uv) const {
        const T x = (static_cast<T>(uv(0)) - cx) / fx;
        const T y = (static_cast<T>(uv(1)) - cy) / fy;
        return Vec2<T>(x, y);
    }

    // - unproject a 2D image point into a 3D point
    //   out: 3D point
    template <typename T> Vec3<T> unproject_point_3d(const T u, const T v, const T depth) const {
        const T x = depth * (u - cx) / fx;
        const T y = depth * (v - cy) / fy;
        return Vec3<T>(x, y, depth);
    }

    template <typename T> Vec3<T> unproject_point_3d(const Vec2Ref<T> uv, const T depth) const {
        const T x = depth * (static_cast<T>(uv(0)) - cx) / fx;
        const T y = depth * (static_cast<T>(uv(1)) - cy) / fy;
        return Vec3<T>(x, y, depth);
    }

    // ------------------------------------------------------------------------
    // - project a 3D point (w.r.t. camera frame) into a 2D image point
    template <typename Scalar>
    std::pair<Vec2<Scalar>, Scalar> project_point_template(Vec3Ref<Scalar> xcs) const {
        return std::make_pair(
            Vec2<Scalar>(fx * xcs.x() / xcs.z() + cx, fy * xcs.y() / xcs.z() + cy), xcs.z());
    }
    virtual std::pair<Vec2<float>, float> project_point(Vec3Ref<float> xcs) const {
        return project_point_template<float>(xcs);
    }
    virtual std::pair<Vec2<double>, double> project_point(Vec3Ref<double> xcs) const {
        return project_point_template<double>(xcs);
    }

    // - project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   out: Nx2 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    virtual std::pair<MatNx2f, VecNf> project(MatNx3fRef xcs) const {
        return CameraUtils::project_points<float>(xcs, K.template cast<float>());
    }
    virtual std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const {
        return CameraUtils::project_points<double>(xcs, K);
    }

    // ------------------------------------------------------------------------
    // - stereo-project a 3D point (w.r.t. camera frame) into a 3D image point (stereo)
    //   out: 3D image point
    template <typename Scalar>
    inline std::pair<Vec3<Scalar>, Scalar>
    project_point_stereo_template(Vec3Ref<Scalar> xcs) const {
        const Scalar z = xcs.z();
#ifndef NDEBUG
        if (z <= kMinZ) {
            MSG_RED_WARN("project_point_stereo_template: Depth is less than minimum depth in "
                         "project_point_stereo");
            return std::make_pair(Vec3<Scalar>(-1.0, -1.0, -1.0), static_cast<Scalar>(-1.0));
        }
#endif
        const Scalar inv_z = 1.0 / z;
        const Scalar u = fx * xcs.x() * inv_z + cx;
        const Scalar v = fy * xcs.y() * inv_z + cy;
        const Scalar ur = u - bf * inv_z;
        return std::make_pair(Vec3<Scalar>(u, v, ur), z);
    }

    virtual std::pair<Vec3<float>, float> project_point_stereo(Vec3Ref<float> xcs) const {
        return project_point_stereo_template<float>(xcs);
    }
    virtual std::pair<Vec3<double>, double> project_point_stereo(Vec3Ref<double> xcs) const {
        return project_point_stereo_template<double>(xcs);
    }

    // - stereo-project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   (assuming rectified stereo images)
    //   out: Nx3 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    virtual std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const {
        return CameraUtils::project_points_stereo<double>(xcs, K, bf);
    }
    virtual std::pair<MatNx3f, VecNf> project_stereo(MatNx3fRef xcs) const {
        return CameraUtils::project_points_stereo<float>(xcs, K.template cast<float>(), bf);
    }

    // ------------------------------------------------------------------------
    void set_intrinsic_matrices();
    void update_cv_matrices(int cv_depth = CV_64F);
    template <typename Scalar> void set_cv_matrices();

    // ------------------------------------------------------------------------
    // Methods
    bool is_stereo() const;
    std::string to_json() const;
    void init_from_json(const std::string &json_str);

    template <typename Scalar> bool is_in_image(Vec2Ref<Scalar> uv, const Scalar z) const {
        return (uv.x() >= u_min) && (uv.x() < u_max) && (uv.y() >= v_min) && (uv.y() < v_max) &&
               (z > 0);
    }

    template <typename Scalar>
    std::vector<bool> are_in_image(MatNxMRef<Scalar> uvs, VecNRef<Scalar> zs) const {
        return CameraUtils::are_in_image(uvs, zs, static_cast<Scalar>(u_min),
                                         static_cast<Scalar>(u_max), static_cast<Scalar>(v_min),
                                         static_cast<Scalar>(v_max));
    }

    Eigen::Matrix4d get_render_projection_matrix(double znear = 0.01, double zfar = 100.0) const;
    void set_fovx(double fovx);
    void set_fovy(double fovy);

    // ------------------------------------------------------------------------
    // - undistort a 2D image point or an array of 2D image points
    //   out: 2D image points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    virtual MatNx2d undistort_points(MatNx2dRef uvs) { return MatNx2d(); }
    virtual MatNx2f undistort_points(MatNx2fRef uvs) { return MatNx2f(); }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1
    // plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    virtual MatNx2d unproject_points(MatNx2dRef uvs) const { return MatNx2d(); }
    virtual MatNx2f unproject_points(MatNx2fRef uvs) const { return MatNx2f(); }

    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1
    // plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    virtual MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const { return MatNx3d(); }
    virtual MatNx3f unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const { return MatNx3f(); }

#ifdef USE_PYTHON
    // Numpy serialization
    pybind11::tuple state_tuple() const;              // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &); // fills this object from the tuple
#endif
};

// PinholeCamera class - inherits from Camera, matches Python PinholeCamera
// exactly
class PinholeCamera : public Camera {
  public:
    // Constructor
    PinholeCamera();
    PinholeCamera(const ConfigDict &config);

    // Destructor
    ~PinholeCamera() = default;

    // Copy constructor, assignment, move constructor, move assignment
    PinholeCamera(const PinholeCamera &other);
    PinholeCamera &operator=(const PinholeCamera &other);
    PinholeCamera(PinholeCamera &&other) noexcept;
    PinholeCamera &operator=(PinholeCamera &&other) noexcept;

    // Methods
    void init();

    // ------------------------------------------------------------------------
    // - project a 3D point (w.r.t. camera frame) into a 2D image point
    std::pair<Vec2<float>, float> project_point(Vec3Ref<float> xcs) const override {
        return project_point_template<float>(xcs);
    }
    std::pair<Vec2<double>, double> project_point(Vec3Ref<double> xcs) const override {
        return project_point_template<double>(xcs);
    }

    // - project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   out: Nx2 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const override {
        return CameraUtils::project_points<double>(xcs, K);
    }
    std::pair<MatNx2f, VecNf> project(MatNx3fRef xcs) const override {
        return CameraUtils::project_points<float>(xcs, K.template cast<float>());
    }

    // ------------------------------------------------------------------------
    // - stereo-project a 3D point (w.r.t. camera frame) into a 2D image point
    std::pair<Vec3<float>, float> project_point_stereo(Vec3Ref<float> xcs) const override {
        return project_point_stereo_template<float>(xcs);
    }
    std::pair<Vec3<double>, double> project_point_stereo(Vec3Ref<double> xcs) const override {
        return project_point_stereo_template<double>(xcs);
    }

    // - stereo-project an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    //   (assuming rectified stereo images)
    //   out: Nx3 image points, [Nx1] array of map point depths
    // Zero-copy friendly overload (NumPy C-contiguous (N,3) float64)
    std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const override {
        return CameraUtils::project_points_stereo<double>(xcs, K, bf);
    }
    std::pair<MatNx3f, VecNf> project_stereo(MatNx3fRef xcs) const override {
        return CameraUtils::project_points_stereo<float>(xcs, K.template cast<float>(), bf);
    }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1
    // plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx2d unproject_points(MatNx2dRef uvs) const override {
        return CameraUtils::unproject_points<double>(uvs, Kinv);
    }
    MatNx2f unproject_points(MatNx2fRef uvs) const override {
        return CameraUtils::unproject_points<float>(uvs, Kinv.template cast<float>());
    }

    // ------------------------------------------------------------------------
    // - unproject a 2D image point or an array of 2D image points into a 3D point on the z=1
    // plane
    //   out: 3D points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const override {
        return CameraUtils::unproject_points_3d<double>(uvs, depths, Kinv);
    }
    MatNx3f unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const override {
        return CameraUtils::unproject_points_3d<float>(uvs, depths, Kinv.template cast<float>());
    }

    // ------------------------------------------------------------------------
    // - undistort a 2D image point or an array of 2D image points
    //   out: 2D image points
    // Zero-copy friendly overload (NumPy C-contiguous (N,2) float64, (3,3) float64)
    template <typename Scalar> MatNx2<Scalar> undistort_points_template(MatNx2Ref<Scalar> uvs);

    MatNx2d undistort_points(MatNx2dRef uvs) override {
        return undistort_points_template<double>(uvs);
    }
    MatNx2f undistort_points(MatNx2fRef uvs) override {
        return undistort_points_template<float>(uvs);
    }

    // ------------------------------------------------------------------------
    // - undistort the image bounds
    void undistort_image_bounds();

    // ------------------------------------------------------------------------
    // - convert to JSON
    std::string to_json() const;
    static PinholeCamera from_json(const std::string &json_str);

#ifdef USE_PYTHON
    // Numpy serialization
    pybind11::tuple state_tuple() const;              // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &); // fills this object from the tuple
#endif

  private:
    // Helper methods
    void compute_fov();
    void update_distortion_flag();
};

// Helper function to create camera from JSON with proper type detection
CameraPtr create_camera_from_json(const nlohmann::json &camera_json);

} // namespace pyslam

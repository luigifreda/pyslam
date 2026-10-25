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

#include "camera_model_math.h"
#include "camera_rig_geometry.h"
#include "utils/messages.h"
#include "utils/serialization_json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pyslam {

// Utility function to get sensor type
SensorType get_sensor_type(const std::string &sensor_str) {
    SensorType sensor_type = SensorType::MONOCULAR;
    // Convert to lowercase
    std::string sensor_lower = sensor_str;
    std::transform(sensor_lower.begin(), sensor_lower.end(), sensor_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (sensor_lower == "monocular" || sensor_lower == "mono") {
        sensor_type = SensorType::MONOCULAR;
    }
    if (sensor_lower == "stereo") {
        sensor_type = SensorType::STEREO;
    } else if (sensor_lower == "rgbd") {
        sensor_type = SensorType::RGBD;
    }
    return sensor_type;
}

// Utility functions
double fov2focal(double fov, int pixels) {
    return static_cast<double>(pixels) / (2.0 * std::tan(fov / 2.0));
}

double focal2fov(double focal, int pixels) {
    return 2.0 * std::atan(static_cast<double>(pixels) / (2.0 * focal));
}

// ================================================================
// CameraUtils Implementation
// ================================================================

template <typename Scalar>
std::pair<MatNx2<Scalar>, VecN<Scalar>> CameraUtils::project_points(MatNx3Ref<Scalar> xcs,
                                                                    Mat3Ref<Scalar> K) {
    const int N = static_cast<int>(xcs.rows());
    MatNx2<Scalar> uv(N, 2);
    VecN<Scalar> z(N);

    // Single pass with fixed-size math. Do not hold xcs * K^T in an `auto`:
    // Eigen keeps it as a lazy product and re-evaluates it on every access.
    const Mat3<Scalar> K_ = K;
    const Scalar min_z = static_cast<Scalar>(kMinZ);
    for (int i = 0; i < N; ++i) {
        const Vec3<Scalar> q = K_ * xcs.row(i).transpose();
        z(i) = q.z();
        if (q.z() > min_z) {
            uv(i, 0) = q.x() / q.z();
            uv(i, 1) = q.y() / q.z();
        } else {
            uv(i, 0) = static_cast<Scalar>(-1.0);
            uv(i, 1) = static_cast<Scalar>(-1.0);
        }
    }

    return {std::move(uv), std::move(z)};
}

template <typename Scalar>
std::pair<MatNx3<Scalar>, VecN<Scalar>>
CameraUtils::project_points_stereo(MatNx3Ref<Scalar> xcs, Mat3Ref<Scalar> K, const Scalar bf) {
    const int N = static_cast<int>(xcs.rows());
    MatNx3<Scalar> projections(N, 3);
    VecN<Scalar> z(N);

    // Single pass with fixed-size math (see project_points).
    const Mat3<Scalar> K_ = K;
    const Scalar min_z = static_cast<Scalar>(kMinZ);
    for (int i = 0; i < N; ++i) {
        const Vec3<Scalar> q = K_ * xcs.row(i).transpose();
        z(i) = q.z();
        if (q.z() > min_z) {
            const Scalar u = q.x() / q.z();
            projections(i, 0) = u;
            projections(i, 1) = q.y() / q.z();
            projections(i, 2) = u - bf / q.z();
        } else {
            projections(i, 0) = static_cast<Scalar>(-1.0);
            projections(i, 1) = static_cast<Scalar>(-1.0);
            projections(i, 2) = static_cast<Scalar>(-1.0);
        }
    }
    return std::make_pair(std::move(projections), std::move(z));
}

template <typename Scalar>
MatNx2<Scalar> CameraUtils::unproject_points(MatNx2Ref<Scalar> uvs, Mat3Ref<Scalar> Kinv) {
    const int N = static_cast<int>(uvs.rows());
    MatNx2<Scalar> result(N, 2);

    for (int i = 0; i < N; ++i) {
        const Vec3<Scalar> uv_homogeneous(uvs(i, 0), uvs(i, 1), 1.0);
        const Vec3<Scalar> p = Kinv * uv_homogeneous;
        result(i, 0) = p.x();
        result(i, 1) = p.y();
    }

    return std::move(result);
}

// ================================================================
// CameraBase Implementation
// ================================================================

CameraBase::CameraBase()
    : type(CameraType::NONE), width(0), height(0), fx(0.0), fy(0.0), cx(0.0), cy(0.0),
      is_distorted(false), fps(0), bf(0.0), b(0.0), u_min(0.0), u_max(0.0), v_min(0.0), v_max(0.0),
      initialized(false) {}

CameraBase::CameraBase(const CameraBase &other)
    : type(other.type), width(other.width), height(other.height), fx(other.fx), fy(other.fy),
      cx(other.cx), cy(other.cy), D(other.D), is_distorted(other.is_distorted), fps(other.fps),
      bf(other.bf), b(other.b), u_min(other.u_min), u_max(other.u_max), v_min(other.v_min),
      v_max(other.v_max), initialized(other.initialized) {}

CameraBase &CameraBase::operator=(const CameraBase &other) {
    if (this != &other) {
        type = other.type;
        width = other.width;
        height = other.height;
        fx = other.fx;
        fy = other.fy;
        cx = other.cx;
        cy = other.cy;
        D = other.D;
        is_distorted = other.is_distorted;
        fps = other.fps;
        bf = other.bf;
        b = other.b;
        u_min = other.u_min;
        u_max = other.u_max;
        v_min = other.v_min;
        v_max = other.v_max;
        initialized = other.initialized;
    }
    return *this;
}

// ================================================================
// Camera Implementation
// ================================================================

Camera::Camera()
    : CameraBase(), fovx(0.0), fovy(0.0), sensor_type(SensorType::MONOCULAR), depth_factor(1.0),
      depth_threshold(std::numeric_limits<double>::infinity()) {}

Camera::Camera(const ConfigDict &config)
    : CameraBase(), fovx(0.0), fovy(0.0), sensor_type(SensorType::MONOCULAR), depth_factor(1.0),
      depth_threshold(std::numeric_limits<double>::infinity()) {

    if (config.is_empty()) {
        return;
    }

    const auto &cam_settings = config.get("cam_settings", ConfigDict());
    const auto &dataset_settings = config.get("dataset_settings", ConfigDict());

    // Extract parameters from config
    width = cam_settings.get("Camera.width", 0);
    if (width <= 0) {
        width = dataset_settings.get("Camera.w", 0);
    }
    height = cam_settings.get("Camera.height", 0);
    if (height <= 0) {
        height = dataset_settings.get("Camera.h", 0);
    }

    fx = cam_settings.get("Camera.fx", 0.0);
    // fy should be read from cam_settings like the other intrinsics
    fy = cam_settings.get("Camera.fy", 0.0);
    cx = cam_settings.get("Camera.cx", 0.0);
    cy = cam_settings.get("Camera.cy", 0.0);

    // Read distortion coefficients (k1, k2, p1, p2, k3) individually from config
    // Consistent with the Python implementation in config.py which ALWAYS uses 5 elements
    D.clear();
    double k1 = cam_settings.get("Camera.k1", 0.0);
    double k2 = cam_settings.get("Camera.k2", 0.0);
    double p1 = cam_settings.get("Camera.p1", 0.0);
    double p2 = cam_settings.get("Camera.p2", 0.0);
    double k3 = cam_settings.get("Camera.k3", 0.0); // Default to 0 if not present like Python
    // Always use 5 coefficients to match Python's np.array([k1, k2, p1, p2, k3])
    D = {k1, k2, p1, p2, k3};

    fps = cam_settings.get("Camera.fps", 30);

    if (width == 0 || height == 0) {
        throw std::runtime_error("Camera: Expecting the fields Camera.width and "
                                 "Camera.height in the camera config file");
    }

    set_intrinsic_matrices();
    update_cv_matrices(CV_64F);

    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);

    // Sensor type
    std::string sensor_str = dataset_settings.get<std::string>("sensor_type", "mono");
    sensor_type = get_sensor_type(sensor_str);

    // For stereo cameras, distortion coefficients are set to zeros
    // (images are rectified, so no distortion)
    if (sensor_type == SensorType::STEREO && !cam_settings.has("Stereo.right")) {
        D = {0.0, 0.0, 0.0, 0.0, 0.0};
        MSG_WARN("Using stereo camera, images are automatically rectified, "
                 "and DistCoef is set to [0,0,0,0,0]");
    }

    // Check if distorted
    double norm = 0.0;
    for (double d : D) {
        norm += d * d;
    }
    is_distorted = std::sqrt(norm) > 1e-10;

    // Stereo parameters
    if (cam_settings.has("Camera.bf") && sensor_type != SensorType::MONOCULAR) {
        bf = cam_settings.get("Camera.bf", 0.0);
        // Avoid division by zero if fx is missing/zero
        b = (fx != 0.0) ? (bf / fx) : 0.0;
        std::cout << "Camera: bf = " << bf << ", b = " << b
                  << ", sensor_type = " << (int)sensor_type << std::endl;
    }
    if (sensor_type == SensorType::STEREO and bf <= 0) {
        throw std::runtime_error("Camera: Expecting the field Camera.bf in the camera config file");
    }

    // Depth factor
    depth_factor = 1.0;
    if (cam_settings.has("DepthMapFactor")) {
        depth_factor = 1.0 / cam_settings.get("DepthMapFactor", 1.0);
    }
    if (sensor_type == SensorType::RGBD and depth_factor <= 0) {
        throw std::runtime_error(
            "Camera: Expecting the field DepthMapFactor in the camera config file");
    }

    // Depth threshold
    depth_threshold = std::numeric_limits<double>::infinity();
    if (cam_settings.has("ThDepth") && sensor_type != SensorType::MONOCULAR) {
        double th_depth = cam_settings.get("ThDepth", 0.0);
        if (bf <= 0) {
            throw std::runtime_error(
                "Camera: Expecting the field ThDepth in the camera config file");
        }
        // Only compute if fx is valid; otherwise keep the default (infinity)
        if (fx != 0.0)
            depth_threshold = bf * th_depth / fx;
    }
    if ((sensor_type == SensorType::RGBD || sensor_type == SensorType::STEREO) &&
        depth_threshold == std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("Camera: Expecting the field ThDepth in the camera config file");
    }
}

Camera::Camera(const Camera &other)
    : CameraBase(other), fovx(other.fovx), fovy(other.fovy), sensor_type(other.sensor_type),
      depth_factor(other.depth_factor), depth_threshold(other.depth_threshold), K(other.K),
      Kinv(other.Kinv), right_camera(other.right_camera), T_right_left(other.T_right_left) {}

Camera &Camera::operator=(const Camera &other) {
    if (this != &other) {
        CameraBase::operator=(other);
        fovx = other.fovx;
        fovy = other.fovy;
        sensor_type = other.sensor_type;
        depth_factor = other.depth_factor;
        depth_threshold = other.depth_threshold;
        K = other.K;
        Kinv = other.Kinv;
        right_camera = other.right_camera;
        T_right_left = other.T_right_left;
    }
    return *this;
}

Camera::Camera(Camera &&other) noexcept
    : CameraBase(std::move(other)), fovx(other.fovx), fovy(other.fovy),
      sensor_type(other.sensor_type), depth_factor(other.depth_factor),
      depth_threshold(other.depth_threshold), K(std::move(other.K)), Kinv(std::move(other.Kinv)),
      right_camera(std::move(other.right_camera)), T_right_left(other.T_right_left) {}

Camera &Camera::operator=(Camera &&other) noexcept {
    if (this != &other) {
        CameraBase::operator=(std::move(other));
        fovx = other.fovx;
        fovy = other.fovy;
        sensor_type = other.sensor_type;
        depth_factor = other.depth_factor;
        depth_threshold = other.depth_threshold;
        K = std::move(other.K);
        Kinv = std::move(other.Kinv);
        right_camera = std::move(other.right_camera);
        T_right_left = other.T_right_left;
    }
    return *this;
}

std::pair<MatNx3d, VecNd> Camera::triangulate_stereo(MatNx2dRef left, MatNx2dRef right, double max_error) const {
    if (!right_camera || left.rows() != right.rows())
        throw std::invalid_argument("triangulate_stereo requires a rig and paired pixel arrays");
    MatNx3d points = MatNx3d::Zero(left.rows(), 3);
    VecNd valid = VecNd::Zero(left.rows());
    for (int i = 0; i < left.rows(); ++i) {
        Vec3d X;
        if (!triangulate_stereo_point(left.row(i).transpose(), right.row(i).transpose(), max_error, X))
            continue;
        points.row(i) = X.transpose();
        valid[i] = 1.0;
    }
    return {std::move(points), std::move(valid)};
}

bool Camera::triangulate_stereo_point(const Vec2d &ul, const Vec2d &ur, double max_error,
                                      Vec3d &X) const {
    return camera_model::triangulate_stereo_point(*this, ul, ur, max_error, X);
}

double Camera::bearing_to_plane_pixel_distance(const Vec3d &bearing, const Vec3d &unit_normal) const {
    return camera_model::bearing_to_plane_pixel_distance(*this, bearing, unit_normal);
}

bool Camera::project_stereo_with_jacobian(Vec3dRef Xc, Vec3d &uvur, Eigen::Matrix3d *J) const {
    return camera_model::project_stereo_with_jacobian(*this, Vec3d(Xc), uvur, J);
}

void Camera::set_intrinsic_matrices() {
    // Compute intrinsic matrix K
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    // Compute inverse intrinsic matrix explicitly
    Kinv << 1.0 / fx, 0.0, -cx / fx, 0.0, 1.0 / fy, -cy / fy, 0.0, 0.0, 1.0;
}

void Camera::update_cv_matrices(int cv_depth_) {
    if (cv_depth_ == cv_depth) {
        return;
    }
    cv_depth = cv_depth_;
    if (cv_depth == CV_64F) {
        set_cv_matrices<double>();
    } else if (cv_depth == CV_32F) {
        set_cv_matrices<float>();
    } else {
        MSG_ERROR("Camera: Unsupported cv_depth: " + std::to_string(cv_depth));
        throw std::runtime_error("Camera: Unsupported cv_depth: " + std::to_string(cv_depth));
    }
}

template <typename Scalar> void Camera::set_cv_matrices() {
    int depth;
    if constexpr (std::is_same_v<Scalar, double>) {
        depth = CV_64F;
    } else if constexpr (std::is_same_v<Scalar, float>) {
        depth = CV_32F;
    } else {
        MSG_ERROR("Camera: Unsupported scalar type: " + std::to_string(sizeof(Scalar)));
        throw std::runtime_error("Camera: Unsupported scalar type: " +
                                 std::to_string(sizeof(Scalar)));
    }

    // clang-format off
    K_cv = (cv::Mat_<Scalar>(3, 3) << 
    static_cast<Scalar>(fx), 0.0, static_cast<Scalar>(cx), 
    0.0, static_cast<Scalar>(fy), static_cast<Scalar>(cy), 
    0.0, 0.0, 1.0);
    if (D.size() >= 8) {
        D_cv = (cv::Mat_<Scalar>(8, 1) << static_cast<Scalar>(D[0]), static_cast<Scalar>(D[1]),
                static_cast<Scalar>(D[2]), static_cast<Scalar>(D[3]), static_cast<Scalar>(D[4]),
                static_cast<Scalar>(D[5]), static_cast<Scalar>(D[6]), static_cast<Scalar>(D[7]));
    } else if (D.size() == 5) {
        D_cv = (cv::Mat_<Scalar>(5, 1) << static_cast<Scalar>(D[0]), static_cast<Scalar>(D[1]),
                static_cast<Scalar>(D[2]), static_cast<Scalar>(D[3]), static_cast<Scalar>(D[4]));
    } else if (D.size() >= 4) {
        D_cv = (cv::Mat_<Scalar>(4, 1) << static_cast<Scalar>(D[0]), static_cast<Scalar>(D[1]),
                static_cast<Scalar>(D[2]), static_cast<Scalar>(D[3]));
    } else {
        D_cv = cv::Mat::zeros(4, 1, depth);
    }
    // clang-format on
}

bool Camera::is_stereo() const { return bf > 0 && sensor_type != SensorType::MONOCULAR; }

Eigen::Matrix4d Camera::get_render_projection_matrix(double znear, double zfar) const {
    double W = width, H = height;
    double left = ((2 * cx - W) / W - 1.0) * W / 2.0;
    double right = ((2 * cx - W) / W + 1.0) * W / 2.0;
    double top = ((2 * cy - H) / H + 1.0) * H / 2.0;
    double bottom = ((2 * cy - H) / H - 1.0) * H / 2.0;

    left = znear / fx * left;
    right = znear / fx * right;
    top = znear / fy * top;
    bottom = znear / fy * bottom;

    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    double z_sign = 1.0;
    P(0, 0) = 2.0 * znear / (right - left);
    P(1, 1) = 2.0 * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);

    return P;
}

void Camera::set_fovx(double fovx) {
    fx = fov2focal(fovx, width);
    this->fovx = fovx;
}

void Camera::set_fovy(double fovy) {
    fy = fov2focal(fovy, height);
    this->fovy = fovy;
}

namespace {

template <typename Scalar>
void check_batch_size(int count, int uv_rows, int depth_size, int valid_size, const char *name) {
    if (uv_rows != count || depth_size != count || valid_size != count) {
        throw std::invalid_argument(std::string(name) +
                                    ": output buffers must have one row per input point");
    }
}

template <typename Scalar>
void project_into_impl(const Camera &cam, MatNx3Ref<Scalar> xcs, Eigen::Ref<MatNx2<Scalar>> uv,
                       Eigen::Ref<VecN<Scalar>> depth, Eigen::Ref<VecN<Scalar>> valid) {
    const int count = static_cast<int>(xcs.rows());
    check_batch_size<Scalar>(count, static_cast<int>(uv.rows()), static_cast<int>(depth.size()),
                             static_cast<int>(valid.size()), "Camera::project_into");
    for (int i = 0; i < count; ++i) {
        Vec3<Scalar> X;
        X << xcs(i, 0), xcs(i, 1), xcs(i, 2);
        Vec2<Scalar> projected;
        const bool ok =
            cam.project_with_jacobian(X, projected, static_cast<Eigen::Matrix<Scalar, 2, 3> *>(nullptr));
        uv(i, 0) = projected[0];
        uv(i, 1) = projected[1];
        depth(i) = X[2];
        valid(i) = ok ? Scalar(1) : Scalar(0);
    }
}

template <typename Scalar>
void unproject_bearings_impl(const Camera &cam, MatNx2Ref<Scalar> uvs,
                             Eigen::Ref<MatNx3<Scalar>> bearings, Eigen::Ref<VecN<Scalar>> valid) {
    const int count = static_cast<int>(uvs.rows());
    if (static_cast<int>(bearings.rows()) != count || static_cast<int>(valid.size()) != count) {
        throw std::invalid_argument(
            "Camera::unproject_bearings_into: output buffers must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        Vec2<Scalar> uv;
        uv << uvs(i, 0), uvs(i, 1);
        Vec3<Scalar> bearing;
        const bool ok =
            cam.unproject_bearing(uv, bearing, static_cast<Eigen::Matrix<Scalar, 3, 2> *>(nullptr));
        bearings(i, 0) = bearing[0];
        bearings(i, 1) = bearing[1];
        bearings(i, 2) = bearing[2];
        valid(i) = ok ? Scalar(1) : Scalar(0);
    }
}

} // namespace

bool Camera::project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                                   Eigen::Matrix<double, 2, 3> *J_uv_Xc) const {
    return camera_model::project_pinhole(fx, fy, cx, cy, Vec3d(Xc), uv, J_uv_Xc);
}

bool Camera::project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                                   Eigen::Matrix<float, 2, 3> *J_uv_Xc) const {
    return camera_model::project_pinhole(static_cast<float>(fx), static_cast<float>(fy),
                                         static_cast<float>(cx), static_cast<float>(cy), Vec3f(Xc),
                                         uv, J_uv_Xc);
}

bool Camera::unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                               Eigen::Matrix<double, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_pinhole_bearing(fx, fy, cx, cy, Vec2d(uv), bearing, J_bearing_uv);
}

bool Camera::unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                               Eigen::Matrix<float, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_pinhole_bearing(static_cast<float>(fx), static_cast<float>(fy),
                                                   static_cast<float>(cx), static_cast<float>(cy),
                                                   Vec2f(uv), bearing, J_bearing_uv);
}

void Camera::project_into(MatNx3dRef xcs, Eigen::Ref<MatNx2d> uv, Eigen::Ref<VecNd> depth,
                          Eigen::Ref<VecNd> valid) const {
    project_into_impl<double>(*this, xcs, uv, depth, valid);
}

void Camera::project_into(MatNx3fRef xcs, Eigen::Ref<MatNx2f> uv, Eigen::Ref<VecNf> depth,
                          Eigen::Ref<VecNf> valid) const {
    project_into_impl<float>(*this, xcs, uv, depth, valid);
}

void Camera::unproject_bearings_into(MatNx2dRef uvs, Eigen::Ref<MatNx3d> bearings,
                                     Eigen::Ref<VecNd> valid) const {
    unproject_bearings_impl<double>(*this, uvs, bearings, valid);
}

void Camera::unproject_bearings_into(MatNx2fRef uvs, Eigen::Ref<MatNx3f> bearings,
                                     Eigen::Ref<VecNf> valid) const {
    unproject_bearings_impl<float>(*this, uvs, bearings, valid);
}

bool Camera::reprojection_error_jacobians(Vec3dRef Xc, const Eigen::Matrix<double, 3, 6> &dXc_dpose,
                                          const Eigen::Matrix3d &dXc_dpoint,
                                          Eigen::Matrix<double, 2, 6> &J_error_pose,
                                          Eigen::Matrix<double, 2, 3> &J_error_point,
                                          Vec2d *uv) const {
    Vec2d projected;
    Eigen::Matrix<double, 2, 3> J_uv_Xc;
    const bool valid = project_with_jacobian(Xc, projected, &J_uv_Xc);
    if (uv) {
        *uv = projected;
    }
    if (!valid) J_uv_Xc.setZero();
    camera_model::reprojection_error_jacobians(J_uv_Xc, dXc_dpose, dXc_dpoint, J_error_pose,
                                               J_error_point);
    return valid;
}

bool Camera::g2o_reprojection_error_jacobians(Vec3dRef Xc, const Eigen::Matrix3d &Rcw,
                                              Eigen::Matrix<double, 2, 6> &J_error_pose,
                                              Eigen::Matrix<double, 2, 3> &J_error_point,
                                              Vec2d *uv) const {
    Vec2d projected;
    Eigen::Matrix<double, 2, 3> J_uv_Xc;
    const bool valid = project_with_jacobian(Xc, projected, &J_uv_Xc);
    if (uv) {
        *uv = projected;
    }
    if (!valid) {
        J_error_pose.setZero();
        J_error_point.setZero();
        return false;
    }
    camera_model::g2o_reprojection_error_jacobians(J_uv_Xc, Vec3d(Xc), Rcw, J_error_pose,
                                                   J_error_point);
    return valid;
}

// ================================================================
// PinholeCamera Implementation
// ================================================================

PinholeCamera::PinholeCamera() : Camera() { type = CameraType::PINHOLE; }

// PinholeCamera Implementation
PinholeCamera::PinholeCamera(const ConfigDict &config) : Camera(config) {
    type = CameraType::PINHOLE;

    if (config.is_empty()) {
        return;
    }

    u_min = 0.0;
    u_max = width;
    v_min = 0.0;
    v_max = height;

    init();
}

PinholeCamera::PinholeCamera(const PinholeCamera &other) : Camera(other) {
    u_min = other.u_min;
    u_max = other.u_max;
    v_min = other.v_min;
    v_max = other.v_max;
}

PinholeCamera &PinholeCamera::operator=(const PinholeCamera &other) {
    if (this != &other) {
        Camera::operator=(other);
        u_min = other.u_min;
        u_max = other.u_max;
        v_min = other.v_min;
        v_max = other.v_max;
    }
    return *this;
}

PinholeCamera::PinholeCamera(PinholeCamera &&other) noexcept : Camera(std::move(other)) {
    u_min = other.u_min;
    u_max = other.u_max;
    v_min = other.v_min;
    v_max = other.v_max;
}

PinholeCamera &PinholeCamera::operator=(PinholeCamera &&other) noexcept {
    if (this != &other) {
        Camera::operator=(std::move(other));
        u_min = other.u_min;
        u_max = other.u_max;
        v_min = other.v_min;
        v_max = other.v_max;
    }
    return *this;
}

void PinholeCamera::init() {
    if (!initialized) {
        initialized = true;
        undistort_image_bounds();
    }
}

template <typename Scalar>
MatNx2<Scalar> PinholeCamera::undistort_points_template(MatNx2Ref<Scalar> uvs) {
    if (!is_distorted) {
        return uvs;
    }

    int depth = CV_32F;
    if constexpr (std::is_same_v<Scalar, double>) {
        depth = CV_64F;
    }
    update_cv_matrices(depth);

    const int num_points = static_cast<int>(uvs.rows());

    cv::Mat uvs_mat(num_points, 2, depth);
    for (int i = 0; i < num_points; ++i) {
        uvs_mat.at<Scalar>(i, 0) = uvs(i, 0);
        uvs_mat.at<Scalar>(i, 1) = uvs(i, 1);
    }
    uvs_mat.reshape(2);
    cv::undistortPoints(uvs_mat, uvs_mat, K_cv, D_cv, cv::Mat(), K_cv);
    uvs_mat.reshape(1);

    // Convert back to NumPy array
    MatNx2<Scalar> result(uvs.rows(), 2);
    for (int i = 0; i < num_points; ++i) {
        result(i, 0) = uvs_mat.at<Scalar>(i, 0);
        result(i, 1) = uvs_mat.at<Scalar>(i, 1);
    }

    return result;
}

void PinholeCamera::undistort_image_bounds() {
    MatNx2d uv_bounds(4, 2);
    uv_bounds << u_min, v_min, u_min, v_max, u_max, v_min, u_max, v_max;

    if (is_distorted) {
        uv_bounds = undistort_points(uv_bounds);
    }

    u_min = std::min({uv_bounds(0, 0), uv_bounds(1, 0)});
    u_max = std::max({uv_bounds(2, 0), uv_bounds(3, 0)});
    v_min = std::min({uv_bounds(0, 1), uv_bounds(2, 1)});
    v_max = std::max({uv_bounds(1, 1), uv_bounds(3, 1)});
}

void PinholeCamera::compute_fov() {
    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);
}

void PinholeCamera::update_distortion_flag() {
    is_distorted = false;

    if (!D.empty()) {
        double norm = 0.0;
        for (double d : D) {
            norm += d * d;
        }
        is_distorted = std::sqrt(norm) > 1e-10;
    }
}

//===============================================================================================
// Explicit template instantiations for CameraUtils methods used in Python bindings
//===============================================================================================

template MatNx2<float> pyslam::CameraUtils::unproject_points<float>(MatNx2Ref<float> uvs,
                                                                    Mat3Ref<float> Kinv);
template MatNx2<double> pyslam::CameraUtils::unproject_points<double>(MatNx2Ref<double> uvs,
                                                                      Mat3Ref<double> Kinv);

} // namespace pyslam

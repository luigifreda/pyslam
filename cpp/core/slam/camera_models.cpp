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

#include "camera_models.h"

#include "camera_model_math.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace pyslam {
namespace {

template <typename Scalar> Scalar distortion_coeff(const std::vector<double> &D, size_t index) {
    return static_cast<Scalar>(index < D.size() ? D[index] : 0.0);
}

void refresh_model(Camera &cam, CameraType type, bool distorted) {
    cam.type = type;
    cam.is_distorted = distorted;
    cam.set_intrinsic_matrices();
    cam.u_min = 0.0;
    cam.u_max = static_cast<double>(cam.width);
    cam.v_min = 0.0;
    cam.v_max = static_cast<double>(cam.height);
    cam.initialized = true;
    cam.cv_depth = -1;
    cam.update_cv_matrices(CV_64F);
}

[[noreturn]] void reject_stereo(const char *model) {
    throw std::logic_error(std::string(model) +
                           " does not support rectified stereo projection");
}

template <typename Scalar>
std::pair<Vec3<Scalar>, Scalar> project_depth_observation(const Camera &cam, Vec3Ref<Scalar> X) {
    if (!cam.is_native_stereo() && cam.sensor_type != SensorType::RGBD) reject_stereo("Camera");
    const Vec3d point = X.template cast<double>();
    Vec3d uv;
    const bool valid = cam.project_stereo_with_jacobian(point, uv);
    return {uv.template cast<Scalar>(), valid ? X.z() : std::numeric_limits<Scalar>::quiet_NaN()};
}

template <typename Scalar>
std::pair<MatNx3<Scalar>, VecN<Scalar>> project_depth_rows(const Camera &cam, MatNx3Ref<Scalar> X) {
    MatNx3<Scalar> uv(X.rows(), 3);
    VecN<Scalar> depth(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        const Vec3<Scalar> point = X.row(i).transpose();
        auto projected = project_depth_observation<Scalar>(cam, point);
        uv.row(i) = projected.first.transpose();
        depth[i] = projected.second;
    }
    return {std::move(uv), std::move(depth)};
}

template <typename Scalar>
std::pair<MatNx2<Scalar>, VecN<Scalar>> project_rows(const Camera &cam, MatNx3Ref<Scalar> xcs) {
    const int N = static_cast<int>(xcs.rows());
    MatNx2<Scalar> uv(N, 2);
    VecN<Scalar> depth(N);
    for (int i = 0; i < N; ++i) {
        Vec3<Scalar> X;
        X << xcs(i, 0), xcs(i, 1), xcs(i, 2);
        Vec2<Scalar> projected;
        const bool ok = cam.project_with_jacobian(
            X, projected, static_cast<Eigen::Matrix<Scalar, 2, 3> *>(nullptr));
        uv(i, 0) = projected[0];
        uv(i, 1) = projected[1];
        depth(i) = ok ? X[2] : std::numeric_limits<Scalar>::quiet_NaN();
    }
    return {std::move(uv), std::move(depth)};
}

template <typename Scalar>
MatNx2<Scalar> bearings_to_z1(const Camera &cam, MatNx2Ref<Scalar> uvs) {
    const int N = static_cast<int>(uvs.rows());
    MatNx2<Scalar> out(N, 2);
    for (int i = 0; i < N; ++i) {
        Vec2<Scalar> uv;
        uv << uvs(i, 0), uvs(i, 1);
        Vec3<Scalar> bearing;
        const bool ok = cam.unproject_bearing(uv, bearing,
                                              static_cast<Eigen::Matrix<Scalar, 3, 2> *>(nullptr));
        if (!ok || std::abs(bearing.z()) <= Scalar(1e-12)) {
            out(i, 0) = std::numeric_limits<Scalar>::quiet_NaN();
            out(i, 1) = std::numeric_limits<Scalar>::quiet_NaN();
        } else {
            out(i, 0) = bearing.x() / bearing.z();
            out(i, 1) = bearing.y() / bearing.z();
        }
    }
    return out;
}

template <typename Scalar>
MatNx2<Scalar> undistort_to_pinhole(const Camera &cam, MatNx2Ref<Scalar> uvs) {
    const int N = static_cast<int>(uvs.rows());
    MatNx2<Scalar> out(N, 2);
    const Scalar fx = static_cast<Scalar>(cam.fx);
    const Scalar fy = static_cast<Scalar>(cam.fy);
    const Scalar cx = static_cast<Scalar>(cam.cx);
    const Scalar cy = static_cast<Scalar>(cam.cy);
    for (int i = 0; i < N; ++i) {
        Vec2<Scalar> uv;
        uv << uvs(i, 0), uvs(i, 1);
        Vec3<Scalar> bearing;
        const bool ok = cam.unproject_bearing(uv, bearing,
                                              static_cast<Eigen::Matrix<Scalar, 3, 2> *>(nullptr));
        if (!ok || std::abs(bearing.z()) <= Scalar(1e-12)) {
            out(i, 0) = std::numeric_limits<Scalar>::quiet_NaN();
            out(i, 1) = std::numeric_limits<Scalar>::quiet_NaN();
        } else {
            out(i, 0) = fx * bearing.x() / bearing.z() + cx;
            out(i, 1) = fy * bearing.y() / bearing.z() + cy;
        }
    }
    return out;
}

template <typename Scalar>
MatNx3<Scalar> bearings_to_depth(const Camera &cam, MatNx2Ref<Scalar> uvs, VecNRef<Scalar> depths) {
    const MatNx2<Scalar> xy = bearings_to_z1<Scalar>(cam, uvs);
    const int N = static_cast<int>(uvs.rows());
    MatNx3<Scalar> out(N, 3);
    for (int i = 0; i < N; ++i) {
        out(i, 0) = depths(i) * xy(i, 0);
        out(i, 1) = depths(i) * xy(i, 1);
        out(i, 2) = depths(i);
    }
    return out;
}

bool distortion_is_nonzero(const std::vector<double> &D) {
    double norm = 0.0;
    for (double value : D) {
        norm += value * value;
    }
    return std::sqrt(norm) > 1e-10;
}

#define PYSLAM_DEFINE_CAMERA_MODEL_DISPATCH(ClassName)                                             \
    std::pair<Vec2<float>, float> ClassName::project_point(Vec3Ref<float> xcs) const {             \
        Vec2f uv;                                                                                  \
        const bool ok = project_with_jacobian(xcs, uv, nullptr);                                   \
        const float depth = ok ? xcs.z() : std::numeric_limits<float>::quiet_NaN();                \
        return {uv, depth};                                                                        \
    }                                                                                              \
    std::pair<Vec2<double>, double> ClassName::project_point(Vec3Ref<double> xcs) const {          \
        Vec2d uv;                                                                                  \
        const bool ok = project_with_jacobian(xcs, uv, nullptr);                                   \
        const double depth = ok ? xcs.z() : std::numeric_limits<double>::quiet_NaN();              \
        return {uv, depth};                                                                        \
    }                                                                                              \
    std::pair<MatNx2d, VecNd> ClassName::project(MatNx3dRef xcs) const {                           \
        return project_rows<double>(*this, xcs);                                                   \
    }                                                                                              \
    std::pair<MatNx2f, VecNf> ClassName::project(MatNx3fRef xcs) const {                           \
        return project_rows<float>(*this, xcs);                                                    \
    }                                                                                              \
    std::pair<Vec3<float>, float> ClassName::project_point_stereo(Vec3Ref<float> X) const {          \
        return project_depth_observation<float>(*this, X);                                         \
    }                                                                                              \
    std::pair<Vec3<double>, double> ClassName::project_point_stereo(Vec3Ref<double> X) const {       \
        return project_depth_observation<double>(*this, X);                                        \
    }                                                                                              \
    std::pair<MatNx3d, VecNd> ClassName::project_stereo(MatNx3dRef X) const { return project_depth_rows<double>(*this, X); } \
    std::pair<MatNx3f, VecNf> ClassName::project_stereo(MatNx3fRef X) const { return project_depth_rows<float>(*this, X); } \
    MatNx2d ClassName::unproject_points(MatNx2dRef uvs) const {                                    \
        return bearings_to_z1<double>(*this, uvs);                                                 \
    }                                                                                              \
    MatNx2f ClassName::unproject_points(MatNx2fRef uvs) const {                                    \
        return bearings_to_z1<float>(*this, uvs);                                                  \
    }                                                                                              \
    MatNx3d ClassName::unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const {                \
        return bearings_to_depth<double>(*this, uvs, depths);                                      \
    }                                                                                              \
    MatNx3f ClassName::unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const {                \
        return bearings_to_depth<float>(*this, uvs, depths);                                       \
    }                                                                                              \
    MatNx2d ClassName::undistort_points(MatNx2dRef uvs) {                                          \
        return undistort_to_pinhole<double>(*this, uvs);                                           \
    }                                                                                              \
    MatNx2f ClassName::undistort_points(MatNx2fRef uvs) {                                          \
        return undistort_to_pinhole<float>(*this, uvs);                                            \
    }

} // namespace

// ---------------------------------------------------------------------------
// KannalaBrandt4Camera
// ---------------------------------------------------------------------------

KannalaBrandt4Camera::KannalaBrandt4Camera() : Camera() { type = CameraType::KANNALA_BRANDT4; }

KannalaBrandt4Camera::KannalaBrandt4Camera(const ConfigDict &config) : Camera(config) {
    type = CameraType::KANNALA_BRANDT4;
    if (config.is_empty()) {
        return;
    }
    const auto &cam_settings = config.get("cam_settings", ConfigDict());
    D = {cam_settings.get("Camera.k1", 0.0), cam_settings.get("Camera.k2", 0.0),
         cam_settings.get("Camera.k3", 0.0), cam_settings.get("Camera.k4", 0.0)};
    refresh_model(*this, CameraType::KANNALA_BRANDT4, true);
}

KannalaBrandt4Camera::KannalaBrandt4Camera(int image_width, int image_height, double focal_x,
                                           double focal_y, double principal_x, double principal_y,
                                           double k1, double k2, double k3, double k4)
    : Camera() {
    width = image_width;
    height = image_height;
    fx = focal_x;
    fy = focal_y;
    cx = principal_x;
    cy = principal_y;
    D = {k1, k2, k3, k4};
    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);
    refresh_model(*this, CameraType::KANNALA_BRANDT4, true);
}

bool KannalaBrandt4Camera::project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                                                 Eigen::Matrix<double, 2, 3> *J_uv_Xc) const {
    return camera_model::project_kannala_brandt4(
        fx, fy, cx, cy, distortion_coeff<double>(D, 0), distortion_coeff<double>(D, 1),
        distortion_coeff<double>(D, 2), distortion_coeff<double>(D, 3), Vec3d(Xc), uv, J_uv_Xc);
}

bool KannalaBrandt4Camera::project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                                                 Eigen::Matrix<float, 2, 3> *J_uv_Xc) const {
    return camera_model::project_kannala_brandt4(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        distortion_coeff<float>(D, 2), distortion_coeff<float>(D, 3), Vec3f(Xc), uv, J_uv_Xc);
}

bool KannalaBrandt4Camera::unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                                             Eigen::Matrix<double, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_kannala_brandt4_bearing(
        fx, fy, cx, cy, distortion_coeff<double>(D, 0), distortion_coeff<double>(D, 1),
        distortion_coeff<double>(D, 2), distortion_coeff<double>(D, 3), Vec2d(uv), bearing,
        J_bearing_uv);
}

bool KannalaBrandt4Camera::unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                                             Eigen::Matrix<float, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_kannala_brandt4_bearing(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        distortion_coeff<float>(D, 2), distortion_coeff<float>(D, 3), Vec2f(uv), bearing,
        J_bearing_uv);
}

PYSLAM_DEFINE_CAMERA_MODEL_DISPATCH(KannalaBrandt4Camera)

KannalaBrandt4Camera KannalaBrandt4Camera::from_json(const std::string &json_str) {
    KannalaBrandt4Camera camera;
    camera.init_from_json(json_str);
    refresh_model(camera, CameraType::KANNALA_BRANDT4, true);
    return camera;
}

// ---------------------------------------------------------------------------
// PinholeRadtan8Camera
// ---------------------------------------------------------------------------

PinholeRadtan8Camera::PinholeRadtan8Camera() : Camera() { type = CameraType::PINHOLE_RADTAN8; }

PinholeRadtan8Camera::PinholeRadtan8Camera(const ConfigDict &config) : Camera(config) {
    type = CameraType::PINHOLE_RADTAN8;
    if (config.is_empty()) {
        return;
    }
    const auto &cam_settings = config.get("cam_settings", ConfigDict());
    D = {cam_settings.get("Camera.k1", 0.0), cam_settings.get("Camera.k2", 0.0),
         cam_settings.get("Camera.p1", 0.0), cam_settings.get("Camera.p2", 0.0),
         cam_settings.get("Camera.k3", 0.0), cam_settings.get("Camera.k4", 0.0),
         cam_settings.get("Camera.k5", 0.0), cam_settings.get("Camera.k6", 0.0)};
    refresh_model(*this, CameraType::PINHOLE_RADTAN8, distortion_is_nonzero(D));
}

PinholeRadtan8Camera::PinholeRadtan8Camera(int image_width, int image_height, double focal_x,
                                           double focal_y, double principal_x, double principal_y,
                                           double k1, double k2, double p1, double p2, double k3,
                                           double k4, double k5, double k6)
    : Camera() {
    width = image_width;
    height = image_height;
    fx = focal_x;
    fy = focal_y;
    cx = principal_x;
    cy = principal_y;
    D = {k1, k2, p1, p2, k3, k4, k5, k6};
    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);
    refresh_model(*this, CameraType::PINHOLE_RADTAN8, distortion_is_nonzero(D));
}

bool PinholeRadtan8Camera::project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                                                 Eigen::Matrix<double, 2, 3> *J_uv_Xc) const {
    return camera_model::project_pinhole_radtan8(
        fx, fy, cx, cy, distortion_coeff<double>(D, 0), distortion_coeff<double>(D, 1),
        distortion_coeff<double>(D, 2), distortion_coeff<double>(D, 3),
        distortion_coeff<double>(D, 4), distortion_coeff<double>(D, 5),
        distortion_coeff<double>(D, 6), distortion_coeff<double>(D, 7), Vec3d(Xc), uv, J_uv_Xc);
}

bool PinholeRadtan8Camera::project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                                                 Eigen::Matrix<float, 2, 3> *J_uv_Xc) const {
    return camera_model::project_pinhole_radtan8(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        distortion_coeff<float>(D, 2), distortion_coeff<float>(D, 3), distortion_coeff<float>(D, 4),
        distortion_coeff<float>(D, 5), distortion_coeff<float>(D, 6), distortion_coeff<float>(D, 7),
        Vec3f(Xc), uv, J_uv_Xc);
}

bool PinholeRadtan8Camera::unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                                             Eigen::Matrix<double, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_pinhole_radtan8_bearing(
        fx, fy, cx, cy, distortion_coeff<double>(D, 0), distortion_coeff<double>(D, 1),
        distortion_coeff<double>(D, 2), distortion_coeff<double>(D, 3),
        distortion_coeff<double>(D, 4), distortion_coeff<double>(D, 5),
        distortion_coeff<double>(D, 6), distortion_coeff<double>(D, 7), Vec2d(uv), bearing,
        J_bearing_uv);
}

bool PinholeRadtan8Camera::unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                                             Eigen::Matrix<float, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_pinhole_radtan8_bearing(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        distortion_coeff<float>(D, 2), distortion_coeff<float>(D, 3), distortion_coeff<float>(D, 4),
        distortion_coeff<float>(D, 5), distortion_coeff<float>(D, 6), distortion_coeff<float>(D, 7),
        Vec2f(uv), bearing, J_bearing_uv);
}

PYSLAM_DEFINE_CAMERA_MODEL_DISPATCH(PinholeRadtan8Camera)

PinholeRadtan8Camera PinholeRadtan8Camera::from_json(const std::string &json_str) {
    PinholeRadtan8Camera camera;
    camera.init_from_json(json_str);
    refresh_model(camera, CameraType::PINHOLE_RADTAN8, distortion_is_nonzero(camera.D));
    return camera;
}

// ---------------------------------------------------------------------------
// DoubleSphereCamera
// ---------------------------------------------------------------------------

DoubleSphereCamera::DoubleSphereCamera() : Camera() {
    type = CameraType::DOUBLE_SPHERE;
    D = {0.0, 0.5};
}

DoubleSphereCamera::DoubleSphereCamera(const ConfigDict &config) : Camera(config) {
    type = CameraType::DOUBLE_SPHERE;
    if (config.is_empty()) {
        D = {0.0, 0.5};
        return;
    }
    const auto &cam_settings = config.get("cam_settings", ConfigDict());
    D = {cam_settings.get("Camera.xi", 0.0), cam_settings.get("Camera.alpha", 0.5)};
    refresh_model(*this, CameraType::DOUBLE_SPHERE, true);
}

DoubleSphereCamera::DoubleSphereCamera(int image_width, int image_height, double focal_x,
                                       double focal_y, double principal_x, double principal_y,
                                       double xi, double alpha)
    : Camera() {
    width = image_width;
    height = image_height;
    fx = focal_x;
    fy = focal_y;
    cx = principal_x;
    cy = principal_y;
    D = {xi, alpha};
    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);
    refresh_model(*this, CameraType::DOUBLE_SPHERE, true);
}

bool DoubleSphereCamera::project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                                               Eigen::Matrix<double, 2, 3> *J_uv_Xc) const {
    return camera_model::project_double_sphere(fx, fy, cx, cy, distortion_coeff<double>(D, 0),
                                               distortion_coeff<double>(D, 1), Vec3d(Xc), uv,
                                               J_uv_Xc);
}

bool DoubleSphereCamera::project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                                               Eigen::Matrix<float, 2, 3> *J_uv_Xc) const {
    return camera_model::project_double_sphere(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        Vec3f(Xc), uv, J_uv_Xc);
}

bool DoubleSphereCamera::unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                                           Eigen::Matrix<double, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_double_sphere_bearing(
        fx, fy, cx, cy, distortion_coeff<double>(D, 0), distortion_coeff<double>(D, 1), Vec2d(uv),
        bearing, J_bearing_uv);
}

bool DoubleSphereCamera::unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                                           Eigen::Matrix<float, 3, 2> *J_bearing_uv) const {
    return camera_model::unproject_double_sphere_bearing(
        static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx),
        static_cast<float>(cy), distortion_coeff<float>(D, 0), distortion_coeff<float>(D, 1),
        Vec2f(uv), bearing, J_bearing_uv);
}

PYSLAM_DEFINE_CAMERA_MODEL_DISPATCH(DoubleSphereCamera)

DoubleSphereCamera DoubleSphereCamera::from_json(const std::string &json_str) {
    DoubleSphereCamera camera;
    camera.init_from_json(json_str);
    refresh_model(camera, CameraType::DOUBLE_SPHERE, true);
    return camera;
}

#undef PYSLAM_DEFINE_CAMERA_MODEL_DISPATCH

} // namespace pyslam

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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace pyslam {

// Utility functions
double fov2focal(double fov, int pixels) {
    return static_cast<double>(pixels) / (2.0 * std::tan(fov / 2.0));
}

double focal2fov(double focal, int pixels) {
    return 2.0 * std::atan(static_cast<double>(pixels) / (2.0 * focal));
}

// CameraUtils Implementation
std::vector<Eigen::Vector3d> CameraUtils::backproject_3d(const std::vector<cv::Point2f> &uv,
                                                         const std::vector<double> &depth,
                                                         const Eigen::Matrix3d &K) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(uv.size());

    Eigen::Matrix3d Kinv = K.inverse();

    for (size_t i = 0; i < uv.size() && i < depth.size(); ++i) {
        Eigen::Vector3d uv_homogeneous(uv[i].x, uv[i].y, 1.0);
        Eigen::Vector3d p3d = depth[i] * (Kinv * uv_homogeneous);
        result.push_back(p3d);
    }

    return result;
}

std::vector<Eigen::Vector3d> CameraUtils::backproject_3d_numba(const std::vector<cv::Point2f> &uv,
                                                               const std::vector<double> &depth,
                                                               const Eigen::Matrix3d &Kinv) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(uv.size());

    for (size_t i = 0; i < uv.size() && i < depth.size(); ++i) {
        Eigen::Vector3d uv_homogeneous(uv[i].x, uv[i].y, 1.0);
        Eigen::Vector3d p3d = depth[i] * (Kinv * uv_homogeneous);
        result.push_back(p3d);
    }

    return result;
}

std::pair<std::vector<cv::Point2f>, std::vector<double>>
CameraUtils::project(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K) {
    std::vector<cv::Point2f> projections;
    std::vector<double> depths;
    projections.reserve(xcs.size());
    depths.reserve(xcs.size());

    for (const auto &xc : xcs) {
        Eigen::Vector3d proj = K * xc;
        double z = proj.z();
        depths.push_back(z);

        if (z > 1e-6) {
            projections.push_back(
                cv::Point2f(static_cast<float>(proj.x() / z), static_cast<float>(proj.y() / z)));
        } else {
            projections.push_back(cv::Point2f(-1.0f, -1.0f)); // Invalid point
        }
    }

    return std::make_pair(projections, depths);
}

std::pair<std::vector<cv::Point2f>, std::vector<double>>
CameraUtils::project_numba(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K) {
    return project(xcs, K); // Same implementation for now
}

std::pair<std::vector<cv::Point3f>, std::vector<double>>
CameraUtils::project_stereo(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K,
                            double bf) {
    std::vector<cv::Point3f> projections;
    std::vector<double> depths;
    projections.reserve(xcs.size());
    depths.reserve(xcs.size());

    for (const auto &xc : xcs) {
        Eigen::Vector3d proj = K * xc;
        double z = proj.z();
        depths.push_back(z);

        if (z > 1e-6) {
            double u = proj.x() / z;
            double v = proj.y() / z;
            double ur = u - bf / z;
            projections.push_back(
                cv::Point3f(static_cast<float>(u), static_cast<float>(v), static_cast<float>(ur)));
        } else {
            projections.push_back(cv::Point3f(-1.0f, -1.0f, -1.0f)); // Invalid point
        }
    }

    return std::make_pair(projections, depths);
}

std::pair<std::vector<cv::Point3f>, std::vector<double>>
CameraUtils::project_stereo_numba(const std::vector<Eigen::Vector3d> &xcs, const Eigen::Matrix3d &K,
                                  double bf) {
    return project_stereo(xcs, K, bf); // Same implementation for now
}

std::vector<Eigen::Vector2d> CameraUtils::unproject_points(const std::vector<cv::Point2f> &uvs,
                                                           const Eigen::Matrix3d &Kinv) {
    std::vector<Eigen::Vector2d> result;
    result.reserve(uvs.size());

    for (const auto &uv : uvs) {
        Eigen::Vector3d uv_homogeneous(uv.x, uv.y, 1.0);
        Eigen::Vector3d p = Kinv * uv_homogeneous;
        result.push_back(Eigen::Vector2d(p.x(), p.y()));
    }

    return result;
}

std::vector<Eigen::Vector2d>
CameraUtils::unproject_points_numba(const std::vector<cv::Point2f> &uvs,
                                    const Eigen::Matrix3d &Kinv) {
    return unproject_points(uvs, Kinv); // Same implementation for now
}

std::vector<Eigen::Vector3d> CameraUtils::unproject_points_3d(const std::vector<cv::Point2f> &uvs,
                                                              const std::vector<double> &depths,
                                                              const Eigen::Matrix3d &Kinv) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(uvs.size());

    for (size_t i = 0; i < uvs.size() && i < depths.size(); ++i) {
        Eigen::Vector3d uv_homogeneous(uvs[i].x, uvs[i].y, 1.0);
        Eigen::Vector3d p = depths[i] * (Kinv * uv_homogeneous);
        result.push_back(p);
    }

    return result;
}

std::vector<Eigen::Vector3d>
CameraUtils::unproject_points_3d_numba(const std::vector<cv::Point2f> &uvs,
                                       const std::vector<double> &depths,
                                       const Eigen::Matrix3d &Kinv) {
    return unproject_points_3d(uvs, depths, Kinv); // Same implementation for now
}

std::vector<bool> CameraUtils::are_in_image_numba(const std::vector<cv::Point2f> &uvs,
                                                  const std::vector<double> &zs, double u_min,
                                                  double u_max, double v_min, double v_max) {
    std::vector<bool> result;
    result.reserve(uvs.size());

    for (size_t i = 0; i < uvs.size() && i < zs.size(); ++i) {
        bool in_image = (uvs[i].x >= u_min) && (uvs[i].x < u_max) && (uvs[i].y >= v_min) &&
                        (uvs[i].y < v_max) && (zs[i] > 0);
        result.push_back(in_image);
    }

    return result;
}

// CameraBase Implementation
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

// Camera Implementation
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
    height = cam_settings.get("Camera.height", 0);

    fx = cam_settings.get("Camera.fx", 0.0);
    // fy should be read from cam_settings like the other intrinsics
    fy = cam_settings.get("Camera.fy", 0.0);
    cx = cam_settings.get("Camera.cx", 0.0);
    cy = cam_settings.get("Camera.cy", 0.0);
    D = cam_settings.get("Camera.DistCoef", std::vector<double>{});
    fps = cam_settings.get("Camera.fps", 30);

    fovx = focal2fov(fx, width);
    fovy = focal2fov(fy, height);

    // Check if distorted
    double norm = 0.0;
    for (double d : D) {
        norm += d * d;
    }
    is_distorted = std::sqrt(norm) > 1e-10;

    // Sensor type
    std::string sensor_str = dataset_settings.get<std::string>("sensor_type", "monocular");
    if (sensor_str == "stereo") {
        sensor_type = SensorType::STEREO;
    } else if (sensor_str == "rgbd") {
        sensor_type = SensorType::RGBD;
    } else {
        sensor_type = SensorType::MONOCULAR;
    }

    // Stereo parameters
    if (cam_settings.has("Camera.bf")) {
        bf = cam_settings.get("Camera.bf", 0.0);
        // Avoid division by zero if fx is missing/zero
        b = (fx != 0.0) ? (bf / fx) : 0.0;
    }

    // Depth factor
    if (cam_settings.has("DepthMapFactor")) {
        depth_factor = 1.0 / cam_settings.get("DepthMapFactor", 1.0);
    }

    // Depth threshold
    if (cam_settings.has("ThDepth")) {
        double th_depth = cam_settings.get("ThDepth", 0.0);
        // Only compute if fx is valid; otherwise keep the default (infinity)
        if (fx != 0.0)
            depth_threshold = bf * th_depth / fx;
    }
}

Camera::Camera(const Camera &other)
    : CameraBase(other), fovx(other.fovx), fovy(other.fovy), sensor_type(other.sensor_type),
      depth_factor(other.depth_factor), depth_threshold(other.depth_threshold) {}

Camera &Camera::operator=(const Camera &other) {
    if (this != &other) {
        CameraBase::operator=(other);
        fovx = other.fovx;
        fovy = other.fovy;
        sensor_type = other.sensor_type;
        depth_factor = other.depth_factor;
        depth_threshold = other.depth_threshold;
    }
    return *this;
}

bool Camera::is_stereo() const { return bf > 0; }

std::string Camera::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"type\": " << static_cast<int>(type) << ", ";
    oss << "\"width\": " << width << ", ";
    oss << "\"height\": " << height << ", ";
    oss << "\"fx\": " << fx << ", ";
    oss << "\"fy\": " << fy << ", ";
    oss << "\"cx\": " << cx << ", ";
    oss << "\"cy\": " << cy << ", ";
    oss << "\"D\": [";
    for (size_t i = 0; i < D.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << D[i];
    }
    oss << "], ";
    oss << "\"fps\": " << fps << ", ";
    oss << "\"bf\": " << bf << ", ";
    oss << "\"b\": " << b << ", ";
    oss << "\"depth_factor\": " << depth_factor << ", ";
    oss << "\"depth_threshold\": " << depth_threshold << ", ";
    oss << "\"is_distorted\": " << (is_distorted ? "true" : "false") << ", ";
    oss << "\"u_min\": " << u_min << ", ";
    oss << "\"u_max\": " << u_max << ", ";
    oss << "\"v_min\": " << v_min << ", ";
    oss << "\"v_max\": " << v_max << ", ";
    oss << "\"initialized\": " << (initialized ? "true" : "false");
    oss << "}";
    return oss.str();
}

void Camera::init_from_json(const std::string &json_str) {
    // TODO: Implement JSON parsing
    // This would involve parsing the JSON string and setting all the parameters
}

bool Camera::is_in_image(const cv::Point2f &uv, double z) const {
    return (uv.x >= u_min) && (uv.x < u_max) && (uv.y >= v_min) && (uv.y < v_max) && (z > 0);
}

std::vector<bool> Camera::are_in_image(const std::vector<cv::Point2f> &uvs,
                                       const std::vector<double> &zs) const {
    return CameraUtils::are_in_image_numba(uvs, zs, u_min, u_max, v_min, v_max);
}

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

// PinholeCamera Implementation
PinholeCamera::PinholeCamera(const ConfigDict &config) : Camera(config) {
    type = CameraType::PINHOLE;

    if (config.is_empty()) {
        return;
    }

    compute_intrinsic_matrices();

    if (width == 0 || height == 0) {
        throw std::runtime_error("Camera: Expecting the fields Camera.width and "
                                 "Camera.height in the camera config file");
    }

    u_min = 0.0;
    u_max = width;
    v_min = 0.0;
    v_max = height;

    init();
}

PinholeCamera::PinholeCamera(const PinholeCamera &other)
    : Camera(other), K(other.K), Kinv(other.Kinv) {}

PinholeCamera &PinholeCamera::operator=(const PinholeCamera &other) {
    if (this != &other) {
        Camera::operator=(other);
        K = other.K;
        Kinv = other.Kinv;
    }
    return *this;
}

void PinholeCamera::init() {
    if (!initialized) {
        initialized = true;
        undistort_image_bounds();
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<double>>
PinholeCamera::project(const std::vector<Eigen::Vector3d> &xcs) const {
    return CameraUtils::project_numba(xcs, K);
}

std::pair<std::vector<cv::Point3f>, std::vector<double>>
PinholeCamera::project_stereo(const std::vector<Eigen::Vector3d> &xcs) const {
    return CameraUtils::project_stereo_numba(xcs, K, bf);
}

Eigen::Vector2d PinholeCamera::unproject(const cv::Point2f &uv) const {
    double x = (uv.x - cx) / fx;
    double y = (uv.y - cy) / fy;
    return Eigen::Vector2d(x, y);
}

Eigen::Vector3d PinholeCamera::unproject_3d(double u, double v, double depth) const {
    double x = depth * (u - cx) / fx;
    double y = depth * (v - cy) / fy;
    return Eigen::Vector3d(x, y, depth);
}

std::vector<Eigen::Vector2d>
PinholeCamera::unproject_points(const std::vector<cv::Point2f> &uvs) const {
    return CameraUtils::unproject_points_numba(uvs, Kinv);
}

std::vector<Eigen::Vector3d>
PinholeCamera::unproject_points_3d(const std::vector<cv::Point2f> &uvs,
                                   const std::vector<double> &depths) const {
    return CameraUtils::unproject_points_3d_numba(uvs, depths, Kinv);
}

std::vector<cv::Point2f>
PinholeCamera::undistort_points(const std::vector<cv::Point2f> &uvs) const {
    if (!is_distorted) {
        return uvs;
    }

    // Convert to OpenCV format
    cv::Mat uvs_mat(uvs);
    uvs_mat = uvs_mat.reshape(1, static_cast<int>(uvs.size()));

    // Create intrinsic matrix for OpenCV
    cv::Mat K_cv = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // Create distortion coefficients
    cv::Mat D_cv;
    if (D.size() >= 4) {
        D_cv = (cv::Mat_<double>(1, 4) << D[0], D[1], D[2], D[3]);
    } else {
        D_cv = cv::Mat::zeros(1, 4, CV_64F);
    }

    // Undistort points
    cv::Mat undistorted_uvs;
    cv::undistortPoints(uvs_mat, undistorted_uvs, K_cv, D_cv, cv::Mat(), K_cv);

    // Convert back to vector
    std::vector<cv::Point2f> result;
    result.reserve(uvs.size());

    for (int i = 0; i < undistorted_uvs.rows; ++i) {
        double u = undistorted_uvs.at<double>(i, 0);
        double v = undistorted_uvs.at<double>(i, 1);
        result.push_back(cv::Point2f(static_cast<float>(u), static_cast<float>(v)));
    }

    return result;
}

void PinholeCamera::undistort_image_bounds() {
    std::vector<cv::Point2f> uv_bounds = {cv::Point2f(u_min, v_min), cv::Point2f(u_min, v_max),
                                          cv::Point2f(u_max, v_min), cv::Point2f(u_max, v_max)};

    if (is_distorted) {
        uv_bounds = undistort_points(uv_bounds);
    }

    u_min = std::min({uv_bounds[0].x, uv_bounds[1].x});
    u_max = std::max({uv_bounds[2].x, uv_bounds[3].x});
    v_min = std::min({uv_bounds[0].y, uv_bounds[2].y});
    v_max = std::max({uv_bounds[1].y, uv_bounds[3].y});
}

std::string PinholeCamera::to_json() const {
    std::string camera_json = Camera::to_json();
    // Add K and Kinv matrices to JSON
    // TODO: Implement proper JSON formatting for matrices
    return camera_json;
}

PinholeCamera PinholeCamera::from_json(const std::string &json_str) {
    nlohmann::json json_data = nlohmann::json::parse(json_str);

    ConfigDict config = ConfigDict::from_json(json_data);
    return PinholeCamera(config);
}

void PinholeCamera::compute_intrinsic_matrices() {
    // Compute intrinsic matrix K
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    // Compute inverse intrinsic matrix explicitly
    Kinv = Eigen::Matrix3d::Zero();
    Kinv(0, 0) = 1.0 / fx;
    Kinv(1, 1) = 1.0 / fy;
    Kinv(0, 2) = -cx / fx;
    Kinv(1, 2) = -cy / fy;
    Kinv(2, 2) = 1.0;
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

} // namespace pyslam

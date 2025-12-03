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

// std::pair<pyslam::MatNx2d, pyslam::VecNd> CameraUtils::project_points(pyslam::MatNx3dRef xcs,
//                                                                       pyslam::Mat3dRef K);

template <typename Scalar>
std::pair<MatNx2<Scalar>, VecN<Scalar>> CameraUtils::project_points(MatNx3Ref<Scalar> xcs,
                                                                    Mat3Ref<Scalar> K) {
    const int N = static_cast<int>(xcs.rows());
    MatNx2<Scalar> uv(N, 2);
    VecN<Scalar> z(N);

    // Work in row space to avoid a 3xN transpose/copy:
    const auto Q = xcs * K.transpose();

    // Depths
    z = Q.col(2);

    // Project (elementwise division)
    uv.col(0) = Q.col(0).array() / z.array();
    uv.col(1) = Q.col(1).array() / z.array();

    // Handle near-zero depths
    auto bad = (z.array() <= static_cast<Scalar>(1e-6));
    if (bad.any()) {
        for (int i = 0; i < N; ++i) {
            if (bad(i)) {
                uv(i, 0) = static_cast<Scalar>(-1.0);
                uv(i, 1) = static_cast<Scalar>(-1.0);
            }
        }
    }
    return {std::move(uv), std::move(z)};
}

// ------------------------------------------------------------------------

template <typename Scalar>
std::pair<MatNx3<Scalar>, VecN<Scalar>>
CameraUtils::project_points_stereo(MatNx3Ref<Scalar> xcs, Mat3Ref<Scalar> K, const Scalar bf) {
    const int N = static_cast<int>(xcs.rows());
    MatNx3<Scalar> projections(N, 3);
    VecN<Scalar> depths(N);

    // Work in row space to avoid a 3xN transpose/copy:
    // Q = [X Y Z] = xcs * K^T  (Nx3)
    const auto Q = xcs * K.transpose();

    // Depths
    depths = Q.col(2);

    auto u = Q.col(0).array() / depths.array();
    auto v = Q.col(1).array() / depths.array();
    auto ur = u.array() - bf / depths.array();

    for (int i = 0; i < N; ++i) {
        if (depths(i) > 1e-6) {
            projections(i, 0) = u(i);
            projections(i, 1) = v(i);
            projections(i, 2) = ur(i);
        } else {
            projections(i, 0) = -1.0;
            projections(i, 1) = -1.0;
            projections(i, 2) = -1.0;
        }
    }
    return std::make_pair(projections, depths);
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

// ------------------------------------------------------------------------

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
      depth_factor(other.depth_factor), depth_threshold(other.depth_threshold), K(other.K),
      Kinv(other.Kinv) {}

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
    }
    return *this;
}

void Camera::compute_intrinsic_matrices() {
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
    nlohmann::json json_data = nlohmann::json::parse(json_str);

    // Parse basic camera parameters
    type = static_cast<CameraType>(json_data["type"].get<int>());
    width = json_data["width"].get<int>();
    height = json_data["height"].get<int>();
    fx = json_data["fx"].get<double>();
    fy = json_data["fy"].get<double>();
    cx = json_data["cx"].get<double>();
    cy = json_data["cy"].get<double>();

    // Parse distortion coefficients
    if (!json_data["D"].is_null()) {
        std::vector<double> D_vec = json_data["D"].get<std::vector<double>>();
        D = D_vec;
    } else {
        D.clear();
    }

    fps = json_data["fps"].get<int>();
    bf = json_data["bf"].get<double>();
    b = json_data["b"].get<double>();
    depth_factor = json_data["depth_factor"].get<double>();
    depth_threshold = json_data["depth_threshold"].get<double>();
    is_distorted = json_data["is_distorted"].get<bool>();
    u_min = json_data["u_min"].get<double>();
    u_max = json_data["u_max"].get<double>();
    v_min = json_data["v_min"].get<double>();
    v_max = json_data["v_max"].get<double>();
    initialized = json_data["initialized"].get<bool>();

    // Compute field of view if not already set
    if (fovx == 0.0) {
        fovx = focal2fov(fx, width);
    }
    if (fovy == 0.0) {
        fovy = focal2fov(fy, height);
    }

    // Parse intrinsic matrices
    std::vector<double> K_vec = json_data["K"].get<std::vector<double>>();
    std::vector<double> Kinv_vec = json_data["Kinv"].get<std::vector<double>>();

    // Convert vectors to Eigen matrices (assuming 3x3 matrices)
    K = Eigen::Map<Eigen::Matrix3d>(K_vec.data());
    Kinv = Eigen::Map<Eigen::Matrix3d>(Kinv_vec.data());
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

PinholeCamera::PinholeCamera(const PinholeCamera &other) : Camera(other) {}

PinholeCamera &PinholeCamera::operator=(const PinholeCamera &other) {
    if (this != &other) {
        Camera::operator=(other);
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
MatNx2<Scalar> PinholeCamera::undistort_points_template(MatNx2Ref<Scalar> uvs) const {
    if (!is_distorted) {
        return uvs;
    }

    // Convert to OpenCV format
    cv::Mat uvs_mat(static_cast<int>(uvs.rows()), static_cast<int>(uvs.cols()), CV_64F,
                    const_cast<Scalar *>(uvs.data()));

    // Create intrinsic matrix for OpenCV
    cv::Mat K_cv = (cv::Mat_<Scalar>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // Create distortion coefficients
    cv::Mat D_cv;
    if (D.size() >= 4) {
        D_cv = (cv::Mat_<Scalar>(1, 4) << D[0], D[1], D[2], D[3]);
    } else {
        D_cv = cv::Mat::zeros(1, 4, CV_64F);
    }

    // Undistort points
    cv::Mat undistorted_uvs;
    cv::undistortPoints(uvs_mat, undistorted_uvs, K_cv, D_cv, cv::Mat(), K_cv);

    // Convert back to NumPy array
    MatNx2<Scalar> result(uvs.rows(), 2);
    for (int i = 0; i < undistorted_uvs.rows; ++i) {
        result(i, 0) = undistorted_uvs.at<Scalar>(i, 0);
        result(i, 1) = undistorted_uvs.at<Scalar>(i, 1);
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

// ===============================

// Helper function to create camera from JSON with proper type detection
CameraPtr create_camera_from_json(const nlohmann::json &camera_json) {
    if (camera_json.is_null()) {
        return nullptr;
    }

    try {
        // Check camera type from JSON
        if (camera_json.contains("camera_type")) {
            std::string camera_type = camera_json["camera_type"].get<std::string>();
            if (camera_type == "PinholeCamera") {
                return std::make_shared<PinholeCamera>(
                    PinholeCamera::from_json(camera_json.dump()));
            }
            // Add support for other camera types here as they are implemented
            else {
                throw std::runtime_error("Unsupported camera type: " + camera_type);
            }
        } else {
            // Fallback: assume PinholeCamera if no type specified
            return std::make_shared<PinholeCamera>(PinholeCamera::from_json(camera_json.dump()));
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to create camera from JSON: " + std::string(e.what()));
    }
}

} // namespace pyslam

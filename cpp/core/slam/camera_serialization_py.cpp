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
#include "camera_models.h"
#include <nlohmann/json.hpp>

#include "utils/serialization_numpy.h"

#include <Eigen/Dense>

// Define this before including numpy headers to suppress deprecation warnings
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace pyslam {

//=======================================
//         Numpy serialization
//=======================================

// Camera::state_tuple()
py::tuple Camera::state_tuple() const {
    const int version = 1;

    // Serialize all camera parameters
    return py::make_tuple(version, static_cast<int>(type), width, height, fx, fy, cx, cy,
                          D, // distortion coefficients vector
                          fps, bf, b, fovx, fovy, static_cast<int>(sensor_type), depth_factor,
                          depth_threshold, is_distorted, u_min, u_max, v_min, v_max, initialized,
                          K,   // Eigen::Matrix3d intrinsic matrix
                          Kinv, // Eigen::Matrix3d inverse intrinsic matrix
                          right_camera ? py::cast(right_camera->to_json()) : py::none(), T_right_left
    );
}

// Camera::restore_from_state()
void Camera::restore_from_state(const py::tuple &t) {
    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1) {
        throw std::runtime_error("Camera::restore_from_state: unsupported version " +
                                 std::to_string(version));
    }

    // Restore basic camera parameters
    type = static_cast<CameraType>(t[idx++].cast<int>());
    width = t[idx++].cast<int>();
    height = t[idx++].cast<int>();
    fx = t[idx++].cast<double>();
    fy = t[idx++].cast<double>();
    cx = t[idx++].cast<double>();
    cy = t[idx++].cast<double>();
    D = t[idx++].cast<std::vector<double>>();
    fps = t[idx++].cast<int>();
    bf = t[idx++].cast<double>();
    b = t[idx++].cast<double>();
    fovx = t[idx++].cast<double>();
    fovy = t[idx++].cast<double>();
    sensor_type = static_cast<SensorType>(t[idx++].cast<int>());
    depth_factor = t[idx++].cast<double>();
    depth_threshold = t[idx++].cast<double>();
    is_distorted = t[idx++].cast<bool>();
    u_min = t[idx++].cast<double>();
    u_max = t[idx++].cast<double>();
    v_min = t[idx++].cast<double>();
    v_max = t[idx++].cast<double>();
    initialized = t[idx++].cast<bool>();
    K = t[idx++].cast<Eigen::Matrix3d>();
    Kinv = t[idx++].cast<Eigen::Matrix3d>();
    right_camera.reset();
    T_right_left.setIdentity();
    if (t.size() > static_cast<size_t>(idx)) {
        if (!t[idx].is_none()) right_camera = create_camera_from_json(nlohmann::json::parse(t[idx].cast<std::string>()));
        ++idx;
        T_right_left = t[idx++].cast<Eigen::Matrix4d>();
    }
}

// PinholeCamera::state_tuple()
py::tuple PinholeCamera::state_tuple() const {
    const int version = 1;

    // Serialize the base Camera data as a nested tuple
    py::tuple camera_state = Camera::state_tuple();

    // PinholeCamera doesn't add any additional data beyond Camera
    return py::make_tuple(version, camera_state);
}

// PinholeCamera::restore_from_state()
void PinholeCamera::restore_from_state(const py::tuple &t) {
    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1) {
        throw std::runtime_error("PinholeCamera::restore_from_state: unsupported version " +
                                 std::to_string(version));
    }

    // Restore the base Camera data
    auto camera_state = t[idx++].cast<py::tuple>();
    Camera::restore_from_state(camera_state);
}

namespace {

py::tuple model_state_tuple(const Camera &camera) {
    return py::make_tuple(1, camera.state_tuple());
}

void restore_model_state(Camera &camera, const py::tuple &t, const char *model_name) {
    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1) {
        throw std::runtime_error(std::string(model_name) +
                                 "::restore_from_state: unsupported version " +
                                 std::to_string(version));
    }
    camera.restore_from_state(t[idx++].cast<py::tuple>());
}

} // namespace

py::tuple KannalaBrandt4Camera::state_tuple() const { return model_state_tuple(*this); }

void KannalaBrandt4Camera::restore_from_state(const py::tuple &t) {
    restore_model_state(*this, t, "KannalaBrandt4Camera");
    type = CameraType::KANNALA_BRANDT4;
}

py::tuple PinholeRadtan8Camera::state_tuple() const { return model_state_tuple(*this); }

void PinholeRadtan8Camera::restore_from_state(const py::tuple &t) {
    restore_model_state(*this, t, "PinholeRadtan8Camera");
    type = CameraType::PINHOLE_RADTAN8;
}

py::tuple DoubleSphereCamera::state_tuple() const { return model_state_tuple(*this); }

void DoubleSphereCamera::restore_from_state(const py::tuple &t) {
    restore_model_state(*this, t, "DoubleSphereCamera");
    type = CameraType::DOUBLE_SPHERE;
}

} // namespace pyslam

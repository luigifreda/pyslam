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
                          Kinv // Eigen::Matrix3d inverse intrinsic matrix
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

} // namespace pyslam
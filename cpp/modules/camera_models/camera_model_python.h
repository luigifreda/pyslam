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

// Factor modules cannot cast a derived camera to shared_ptr<CameraModel>.
// camera_models._shared_camera performs that cast where the type is registered
// and hands back an owning pointer.

#include "camera_model.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>

inline std::shared_ptr<camera_models::CameraModel> owning_camera(const pybind11::handle &obj) {
    if (!obj || obj.is_none()) {
        throw std::invalid_argument("A camera_models camera is required");
    }
    pybind11::capsule capsule = pybind11::module_::import("camera_models").attr("_shared_camera")(obj);
    auto *held = static_cast<std::shared_ptr<camera_models::CameraModel> *>(capsule);
    if (held == nullptr || !(*held)) {
        throw std::invalid_argument("A camera_models camera is required");
    }
    return *held;
}

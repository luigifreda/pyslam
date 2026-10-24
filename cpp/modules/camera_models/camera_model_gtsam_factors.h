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

// gtsam factors for camera_models::CameraModel. This header does not include g2o.
// The factors are the shared templates in cpp/core/slam/camera_projection_factors_gtsam.h.

#include "camera_model.h"
#include "camera_projection_factors_gtsam.h"

namespace camera_models {

using CameraProjectionFactor = pyslam::camera_model::CameraProjectionFactor<CameraModel>;
using CameraResectioningFactor = pyslam::camera_model::CameraResectioningFactor<CameraModel>;
using StereoCameraProjectionFactor = pyslam::camera_model::StereoCameraProjectionFactor<CameraModel>;
using StereoCameraResectioningFactor = pyslam::camera_model::StereoCameraResectioningFactor<CameraModel>;

} // namespace camera_models

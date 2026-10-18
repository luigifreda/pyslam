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

// cpp_core g2o edges for pyslam::Camera, instantiated from camera_projection_edges_g2o.h.
// This header does not include gtsam; the gtsam factors are in camera_factors_gtsam.h.
// Pinhole cameras keep the native g2o fast path.

#include "camera.h"
#include "camera_projection_edges_g2o.h"

namespace pyslam {

using EdgeCameraProjectXYZ = camera_model::EdgeCameraProjectXYZ<Camera>;
using EdgeCameraProjectXYZOnlyPose = camera_model::EdgeCameraProjectXYZOnlyPose<Camera>;
using EdgeStereoCameraProjectXYZ = camera_model::EdgeStereoCameraProjectXYZ<Camera>;
using EdgeStereoCameraProjectXYZOnlyPose = camera_model::EdgeStereoCameraProjectXYZOnlyPose<Camera>;

} // namespace pyslam

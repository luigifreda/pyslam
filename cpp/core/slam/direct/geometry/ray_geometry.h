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

#include "camera/camera.h"
#include "common/eigen_aliases.h"

#include <Eigen/Geometry>

namespace pyslam {

// Ray geometry on raw pixels, shared by depth measurement, the monocular initializer and the
// photometric modules. Pure geometry: no image access, no tracking state.

// Lift a raw pixel at camera-frame depth z. The pinhole bearing lies on z = 1.
// False when the ray is not safely in front of the camera.
bool lift_reference_point(const Camera &camera, const Vec2d &pixel, double depth, Vec3d &X_ref);

// SVO computeTau(): depth (camera z) uncertainty for a point at depth z on the reference ray
// through pixel_ref, given the current camera, for an angular error px_error_angle.
// Returns a negative value when the geometry is degenerate (no baseline, ray at infinity).
double depth_uncertainty_one_pixel(const Camera &camera, const Eigen::Isometry3d &T_cur_ref,
                                   const Vec2d &pixel_ref, double depth, double px_error_angle);

} // namespace pyslam

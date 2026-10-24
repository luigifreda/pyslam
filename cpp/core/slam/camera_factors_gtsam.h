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

// cpp_core gtsam factors for pyslam::Camera, instantiated from camera_projection_factors_gtsam.h.
// This header does not include g2o; the g2o edges are in camera_edges_g2o.h.
// Pinhole cameras keep the native gtsam fast path.

#include "camera.h"
#include "camera_projection_factors_gtsam.h"

namespace pyslam {

using CameraProjectionFactor = camera_model::CameraProjectionFactor<Camera>;
using CameraResectioningFactor = camera_model::CameraResectioningFactor<Camera>;
using StereoCameraProjectionFactor = camera_model::StereoCameraProjectionFactor<Camera>;
using StereoCameraResectioningFactor = camera_model::StereoCameraResectioningFactor<Camera>;

inline boost::shared_ptr<gtsam_factors::WeightedGenericStereoProjectionFactor3D>
make_stereo_camera_factor(const gtsam::StereoPoint2 &measured, const gtsam::SharedNoiseModel &noise,
                          gtsam::Key pose, gtsam::Key point, CameraPtr camera) {
    if (camera->projects_raw_keypoints())
        return boost::make_shared<StereoCameraProjectionFactor>(measured, noise, pose, point, camera);
    auto K = boost::make_shared<gtsam::Cal3_S2Stereo>(camera->fx, camera->fy, 0.0, camera->cx, camera->cy,
                                                      camera->b);
    return boost::make_shared<gtsam_factors::WeightedGenericStereoProjectionFactor3D>(measured, noise, pose,
                                                                                     point, K);
}

inline boost::shared_ptr<StereoCameraResectioningFactor>
make_stereo_camera_pose_factor(const gtsam::SharedNoiseModel &noise, gtsam::Key pose, CameraPtr camera,
                               const gtsam::StereoPoint2 &measured, const gtsam::Point3 &point) {
    return boost::make_shared<StereoCameraResectioningFactor>(noise, pose, camera, measured, point);
}

} // namespace pyslam

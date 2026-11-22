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
#include "slam_direct/direct_params.h"
#include "slam_direct/live_point.h"

#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Sensor measurements shared by every tracking mode. These turn a pixel into a metric
// 3D point from an RGBD depth image or a stereo pair; none of them estimates a pose.

// Smallest camera-frame depth a tracker projects or warps through.
constexpr double kMinTrackingDepth = 1e-2;

// Convert any 1/3/4-channel image to CV_8UC1. An empty input returns an empty Mat.
cv::Mat as_gray8(const cv::Mat &img);

// Convert the configured sensor units to meters; zero denotes an unusable depth sample.
double sample_depth(const cv::Mat &depth, const Vec2d &uv, double depth_factor);

// Gate for every new landmark: depth in (seed_min_depth, seed_max_depth), the configured range
// the depth filter and the epipolar search also use. Nonlinear models keep a positive optical-axis
// depth and apply the window to the ray length. An undistorted pinhole checks z.
bool metric_point_ok(const Camera &camera, const DirectSlamParams &params, const Vec3d &X);

// Depth images and rectified disparity store camera-frame z. The bearing lies on z = 1
// for a pinhole, so this matches unproject_point_3d there. Same lift as the photometric
// aligner, so map points and warp reference points agree for raw-pixel cameras.
bool lift_pixel_depth(const Camera &camera, const Vec2d &uv, double z, Vec3d &X);

// Rectified KITTI-style cameras store the baseline in Camera::bf and have no
// right camera. A calibrated rig uses Camera::triangulate_stereo_point.
// params.stereo_max_error bounds a calibrated rig's triangulation error.
bool triangulate_match(const Camera &camera, const DirectSlamParams &params, const Vec2d &left,
                       const Vec2d &right, Vec3d &X);

// Expected right-image pixel of a left-camera point Xc observed at uv, used to seed stereo LK.
Vec2d predict_right_pixel(const Camera &camera, const Vec3d &Xc, const Vec2d &uv);

// One sensor measurement per input pixel.
struct PixelMeasurement {
    bool ok = false;           // A metric 3D point was measured.
    Vec3d X = Vec3d::Zero();   // Left-camera point; valid only when ok.
    double sensor_depth = 0.0; // RGBD: the raw sample in meters, also when !ok (0 = no sample).
    Vec2d uv_right = Vec2d::Zero();
    bool has_right = false;    // Stereo: uv_right is the matched right pixel (implies ok).
};

// RGBD samples the depth image. Stereo (sensor STEREO with a right image) matches every
// pixel into the right image with pyramidal LK, seeded by right_init (or the left pixel when
// right_init is empty), and triangulates. Stereo matching is LK in every tracking mode: it
// is a sensor measurement, not pose tracking. Other inputs return entries with ok == false.
std::vector<PixelMeasurement> measure_pixels(const Camera &camera, const DirectSlamParams &params,
                                             const cv::Mat &gray, const cv::Mat &gray_right,
                                             const cv::Mat &depth, const std::vector<Vec2d> &pixels,
                                             const std::vector<Vec2d> &right_init = {});

// measure_pixels at every live point's current pixel (one entry per live point). Stereo
// uses the right image only for a STEREO camera, and seeds LK at the right pixel
// predicted from the landmark seen at Tcw.
std::vector<PixelMeasurement> measure_live_points(const Camera &camera, const DirectSlamParams &params,
                                                  const cv::Mat &gray, const cv::Mat &gray_right,
                                                  const cv::Mat &depth, const Eigen::Isometry3d &Tcw,
                                                  const std::vector<LivePoint> &live_points);

} // namespace pyslam

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

#include "slam_direct/depth_measurement.h"

#include "slam_direct/direct/direct_aligner.h"
#include "slam_direct/klt/klt_tracker.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace pyslam {

cv::Mat as_gray8(const cv::Mat &img) {
    if (img.empty()) {
        return cv::Mat();
    }
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img;
    } else if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("DirectSlam: unsupported image channel count");
    }
    if (gray.type() != CV_8UC1) {
        cv::Mat eight;
        gray.convertTo(eight, CV_8U);
        return eight;
    }
    return gray;
}

double sample_depth(const cv::Mat &depth, const Vec2d &uv, double depth_factor) {
    if (depth.empty()) {
        return 0.0;
    }
    const int u = static_cast<int>(std::lround(uv.x()));
    const int v = static_cast<int>(std::lround(uv.y()));
    if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows) {
        return 0.0;
    }
    double raw = 0.0;
    switch (depth.type()) {
    case CV_16UC1:
        raw = static_cast<double>(depth.at<uint16_t>(v, u));
        break;
    case CV_32FC1:
        raw = static_cast<double>(depth.at<float>(v, u));
        break;
    case CV_64FC1:
        raw = static_cast<double>(depth.at<double>(v, u));
        break;
    default:
        return 0.0;
    }
    if (!(raw > 0.0) || !std::isfinite(raw)) {
        return 0.0;
    }
    const double factor = depth_factor > 0.0 ? depth_factor : 1.0;
    return raw * factor;
}

bool metric_point_ok(const Camera &camera, const Vec3d &X) {
    if (!X.allFinite() || !(X.z() > 0.0)) {
        return false;
    }
    const double metric = camera.projects_raw_keypoints() ? X.norm() : X.z();
    return metric > 0.5 && metric < 80.0;
}

bool lift_pixel_depth(const Camera &camera, const Vec2d &uv, double z, Vec3d &X) {
    return lift_reference_point(camera, uv, z, X);
}

bool triangulate_match(const Camera &camera, const Vec2d &left, const Vec2d &right, double max_error,
                       Vec3d &X) {
    if (camera.right_camera) {
        return camera.triangulate_stereo_point(left, right, max_error, X) && metric_point_ok(camera, X);
    }
    if (!(camera.bf > 0.0) || std::abs(left.y() - right.y()) > 2.0) {
        return false;
    }
    const double disparity = left.x() - right.x();
    if (!(disparity > 0.1)) {
        return false;
    }
    const double depth = camera.bf / disparity;
    if (!(depth > kMinTrackingDepth)) {
        return false;
    }
    return lift_pixel_depth(camera, left, depth, X) && metric_point_ok(camera, X);
}

Vec2d predict_right_pixel(const Camera &camera, const Vec3d &Xc, const Vec2d &uv) {
    if (camera.right_camera) {
        const Vec3d Xr = camera.T_right_left.block<3, 3>(0, 0) * Xc + camera.T_right_left.block<3, 1>(0, 3);
        return camera.right_camera->project_point(Xr).first;
    }
    return Vec2d(uv.x() - camera.bf / Xc.z(), uv.y());
}

std::vector<PixelMeasurement> measure_pixels(const Camera &camera, const DirectSlamParams &params,
                                             const cv::Mat &gray, const cv::Mat &gray_right,
                                             const cv::Mat &depth, const std::vector<Vec2d> &pixels,
                                             const std::vector<Vec2d> &right_init) {
    std::vector<PixelMeasurement> measured(pixels.size());
    if (camera.sensor_type == SensorType::RGBD) {
        for (size_t i = 0; i < pixels.size(); ++i) {
            PixelMeasurement &m = measured[i];
            m.sensor_depth = sample_depth(depth, pixels[i], camera.depth_factor);
            m.ok = lift_pixel_depth(camera, pixels[i], m.sensor_depth, m.X) && metric_point_ok(camera, m.X);
        }
        return measured;
    }
    if (gray_right.empty() || pixels.empty()) {
        return measured;
    }
    // The right pixel comes from pyramidal Lucas-Kanade, not from a descriptor match.
    const LkTrack tracked =
        track_pyramidal_lk(gray, gray_right, pixels, right_init.empty() ? pixels : right_init, params);
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (!tracked.status[i]) {
            continue;
        }
        PixelMeasurement &m = measured[i];
        if (triangulate_match(camera, pixels[i], tracked.points_cur[i], params.stereo_max_error, m.X)) {
            m.ok = true;
            m.uv_right = tracked.points_cur[i];
            m.has_right = true;
        }
    }
    return measured;
}

} // namespace pyslam
